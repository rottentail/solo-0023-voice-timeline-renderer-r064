// Command chunked_render is an offline research harness, NOT production code.
//
// It exercises the question whether the whole-resident render pipeline of
// internal/render can be replaced by chunked processing while keeping output
// PCM bytes, source selection, and receipt facts identical. The production
// packages are imported read-only as the reference implementation; nothing
// under internal/ is modified.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"voice-timeline-renderer/internal/audiocodec"
	"voice-timeline-renderer/internal/domain"
)

// spanPlan is the experiment's description of one timeline span. Frame numbers
// are source-side frames exactly as a client would submit them.
type spanPlan struct {
	name             string
	sourceStartFrame int64
	sourceEndFrame   int64
	gainDB           float64
	crossfadeFrames  int64
}

type sourceDef struct {
	name       string
	rate       int
	channels   int
	frequency  float64
	amplitude  float64
	phase      float64
	frameCount int
}

type prepared struct {
	plan             spanPlan
	clip             *audiocodec.Clip
	gain             float32
	outputStartFrame int64
	outputEndFrame   int64
}

// referenceOutput is everything the legacy whole-resident pipeline produces
// that a chunked replacement must reproduce exactly.
type referenceOutput struct {
	wav            []byte
	prePeak        float64
	gainDB         float64
	postPeakDBFS   float64
	postRMSDBFS    float64
	digest         string
	totalFrames    int64
	outputChannels int
	spanRecords    []domain.ReceiptSpan
}

// facts retained across block boundaries by a correct streaming renderer.
// There is no full-length signal in here.
type retainedFacts struct {
	totalFrames    int64
	channels       int
	prePeak        float64 // max |mix sample| over the whole pre-normalization signal
	sampleCount    int     // total float32 samples, for pre-mix RMS if ever needed
	sumSquares     float64
	postPeak       float64 // measured on quantized int16 values, pass B
	postSumSquares float64 // block-local partial sums merged after the fact
	postSamples    int
	// postSumSquaresLegacy accumulates every quantized sample into one scalar
	// in strict frame order, which is exactly the summation order legacy
	// audiocodec.Measure uses on the whole decoded buffer.
	postSumSquaresLegacy float64
}

func main() {
	outDir := flag.String("out", "research/chunked_render/results", "directory for the JSON experiment report")
	flag.Parse()
	if err := run(*outDir); err != nil {
		fmt.Fprintf(os.Stderr, "experiment failed: %v\n", err)
		os.Exit(1)
	}
}

func run(outDir string) error {
	ctx := context.Background()
	report := &report{BlockSizes: blockSizes}

	// Primary signal: three spans, two crossfades, with a strict triple-overlap
	// region where all three spans contribute to the same output frames.
	triple := buildTripleOverlapCase(ctx)
	ref, err := renderReference(ctx, triple)
	if err != nil {
		return err
	}
	report.ReferenceDigest = ref.digest
	report.ReferenceTotalFrames = ref.totalFrames
	report.ReferencePrePeak = ref.prePeak
	report.ReferenceGainDB = ref.gainDB
	report.ReferencePostPeakDBFS = ref.postPeakDBFS
	report.ReferencePostRMSDBFS = ref.postRMSDBFS

	for _, blockFrames := range blockSizes {
		entry := blockResult{BlockFrames: blockFrames}

		// Correct two-pass scheme, pass B deliberately using a *different* block
		// size than pass A to prove the boundary choice carries no state.
		correct, facts, err := renderCorrectTwoPass(ctx, triple, blockFrames, blockFrames/2+1)
		if err != nil {
			return err
		}
		entry.CorrectBytesEqual = bytesEqual(correct, ref.wav)
		entry.CorrectDigestEqual = sha256Hex(correct) == ref.digest
		entry.FactsPrePeak = facts.prePeak
		entry.FactsPostPeakDBFS = amplitudeDBFS(facts.postPeak)
		entry.FactsPostRMSDBFS = amplitudeDBFS(math.Sqrt(facts.postSumSquares / float64(facts.postSamples)))
		entry.FactsPostRMSLegacyOrderDBFS = amplitudeDBFS(math.Sqrt(facts.postSumSquaresLegacy / float64(facts.postSamples)))
		entry.ReceiptFactsMatch = receiptFactsMatch(facts, ref)
		entry.MaxResidentMixSamples = int64(max(blockFrames, blockFrames/2+1)) * int64(ref.outputChannels)

		// Wrong scheme W1: single pass, normalize each block with its own local
		// peak before any global fact can exist.
		w1, w1Facts, w1Err := renderWrongLocalPeak(ctx, triple, blockFrames)
		if w1Err != nil {
			return w1Err
		}
		entry.W1BytesEqual = bytesEqual(w1, ref.wav)
		entry.W1DigestEqual = sha256Hex(w1) == ref.digest
		entry.W1PostPeakDBFS = amplitudeDBFS(w1Facts.postPeak)
		entry.W1SampleDiffs = countSampleDiffs(w1, ref.wav)
		entry.W1FramesStillEqual = frameCountOf(w1) == ref.totalFrames

		// Wrong scheme W2: blocks are mixed in isolation, crossfade frame index
		// restarts at every block boundary.
		w2, w2Facts, w2Err := renderWrongBlockIsolation(ctx, triple, blockFrames)
		if w2Err != nil {
			return w2Err
		}
		entry.W2BytesEqual = bytesEqual(w2, ref.wav)
		entry.W2DigestEqual = sha256Hex(w2) == ref.digest
		entry.W2PostPeakDBFS = amplitudeDBFS(w2Facts.postPeak)
		entry.W2SampleDiffs = countSampleDiffs(w2, ref.wav)
		entry.W2FramesStillEqual = frameCountOf(w2) == ref.totalFrames
		entry.W2DiffFrameRegions = diffRegions(w2, ref.wav, ref.outputChannels)

		report.Results = append(report.Results, entry)
	}

	// Control case: one span, no crossfade. W2 must become harmless there,
	// showing its failure is specifically the cross-boundary fade handling.
	control := buildNoCrossfadeCase(ctx)
	controlRef, err := renderReference(ctx, control)
	if err != nil {
		return err
	}
	for _, blockFrames := range []int64{1, 3, 7} {
		w2, _, w2Err := renderWrongBlockIsolation(ctx, control, blockFrames)
		if w2Err != nil {
			return w2Err
		}
		report.ControlNoFade = append(report.ControlNoFade, controlResult{
			BlockFrames:      blockFrames,
			W2BytesEqual:     bytesEqual(w2, controlRef.wav),
			FramesStillEqual: frameCountOf(w2) == controlRef.totalFrames,
			ReferenceDigest:  controlRef.digest,
			ObservedDigest:   sha256Hex(w2),
		})
	}

	// Cancellation demonstration: cancel mid pass A and confirm no durable
	// artifact facts survive; then a re-run from scratch reaches the reference.
	cancelOutcome, err := demonstrateCancellation(triple)
	if err != nil {
		return err
	}
	report.Cancellation = cancelOutcome
	report.RMSSummation = runRMSSummationStress()
	fuzzResults, err := runFuzz()
	if err != nil {
		return err
	}
	report.Fuzz = fuzzResults

	report.Summary = summarize(report)

	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return err
	}
	path := filepath.Join(outDir, "experiment_report.json")
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return err
	}
	printConsole(report, path)
	return nil
}

var blockSizes = []int64{1, 2, 3, 4, 5, 6, 8, 11, 16, 23, 32, 37}

// ---------------------------------------------------------------------------
// Signal construction
// ---------------------------------------------------------------------------

func buildTripleOverlapCase(ctx context.Context) experimentCase {
	sources := []sourceDef{
		{name: "srcA", rate: 48000, channels: 2, frequency: 220, amplitude: 0.5, phase: 0.0, frameCount: 20},
		{name: "srcB", rate: 24000, channels: 1, frequency: 330, amplitude: 0.8, phase: 0.7, frameCount: 20},
		{name: "srcC", rate: 48000, channels: 2, frequency: 440, amplitude: 0.35, phase: 1.3, frameCount: 32},
	}
	// Placement at the 48 kHz output rate (span B is a 24 kHz source, so its
	// 12 selected source frames resample to 24 output frames):
	//   span A: 16 out frames, output [0,16),  no fade
	//   span B: 24 out frames, crossfade 14    -> output [2,26),  fade [2,16)
	//   span C: 24 out frames, crossfade 12    -> output [14,38), fade [14,26)
	// The intersection [14,16) is a strict three-span overlap: frames 14 and 15
	// receive a weighted contribution from A, B and C simultaneously. The two
	// fade windows themselves overlap on [14,16). Both fade windows are wider
	// than several experiment block sizes, so every small boundary splits a
	// fade somewhere, including boundaries inside the triple region.
	spans := []spanPlan{
		{name: "spanA", sourceStartFrame: 0, sourceEndFrame: 16, gainDB: -2, crossfadeFrames: 0},
		{name: "spanB", sourceStartFrame: 2, sourceEndFrame: 14, gainDB: 1, crossfadeFrames: 14},
		{name: "spanC", sourceStartFrame: 4, sourceEndFrame: 28, gainDB: -4, crossfadeFrames: 12},
	}
	return buildCase(ctx, "triple_overlap", sources, spans, 48000, 2, -1.0)
}

func buildNoCrossfadeCase(ctx context.Context) experimentCase {
	sources := []sourceDef{
		{name: "srcSolo", rate: 48000, channels: 1, frequency: 180, amplitude: 0.4, phase: 0.2, frameCount: 24},
	}
	spans := []spanPlan{
		{name: "solo", sourceStartFrame: 0, sourceEndFrame: 18, gainDB: -3, crossfadeFrames: 0},
	}
	return buildCase(ctx, "no_crossfade_control", sources, spans, 48000, 1, -2.0)
}

type experimentCase struct {
	name           string
	wavs           map[string][]byte
	plans          []spanPlan
	sampleRate     int
	outputChannels int
	targetPeakDBFS float64
}

func buildCase(ctx context.Context, name string, defs []sourceDef, plans []spanPlan, sampleRate, channels int, target float64) experimentCase {
	wavs := make(map[string][]byte, len(defs))
	for index, def := range defs {
		wav, err := deterministicWAV(def)
		if err != nil {
			panic(err)
		}
		_ = ctx
		wavs[fmt.Sprintf("src%d", index)] = wav
	}
	return experimentCase{
		name:           name,
		wavs:           wavs,
		plans:          plans,
		sampleRate:     sampleRate,
		outputChannels: channels,
		targetPeakDBFS: target,
	}
}

func deterministicWAV(def sourceDef) ([]byte, error) {
	samples := make([]float32, def.frameCount*def.channels)
	for frame := 0; frame < def.frameCount; frame++ {
		seconds := float64(frame) / float64(def.rate)
		first := math.Sin(2*math.Pi*def.frequency*seconds+def.phase) * def.amplitude
		if def.channels == 1 {
			samples[frame] = float32(first)
			continue
		}
		samples[frame*2] = float32(first)
		samples[frame*2+1] = float32(0.75 * math.Sin(2*math.Pi*def.frequency*seconds+def.phase+0.4) * def.amplitude)
	}
	return audiocodec.EncodePCM16(def.rate, def.channels, samples)
}

// ---------------------------------------------------------------------------
// Preparation: mirrors internal/render.prepareSpans using exported codec calls.
// ---------------------------------------------------------------------------

func prepare(ctx context.Context, c experimentCase) ([]prepared, int64, error) {
	decoded := make(map[string]*audiocodec.Clip, len(c.wavs))
	keys := make([]string, 0, len(c.wavs))
	for key := range c.wavs {
		keys = append(keys, key)
	}
	// deterministic key order for stable reports
	sortStrings(keys)

	out := make([]prepared, 0, len(c.plans))
	cursor := int64(0)
	for index, plan := range c.plans {
		key := keys[index]
		clip, ok := decoded[key]
		if !ok {
			parsed, _, err := audiocodec.DecodeWAV(c.wavs[key])
			if err != nil {
				return nil, 0, err
			}
			clip = parsed
			decoded[key] = parsed
		}
		selected, err := audiocodec.SliceFrames(ctx, clip, plan.sourceStartFrame, plan.sourceEndFrame)
		if err != nil {
			return nil, 0, err
		}
		resampled, err := audiocodec.Resample(ctx, selected, c.sampleRate)
		if err != nil {
			return nil, 0, err
		}
		mapped, _, err := audiocodec.MapChannels(ctx, resampled, c.outputChannels)
		if err != nil {
			return nil, 0, err
		}
		frames := mapped.FrameCount()
		start := cursor
		if index > 0 {
			start -= plan.crossfadeFrames
		}
		end := start + frames
		cursor = end
		out = append(out, prepared{
			plan:             plan,
			clip:             mapped,
			gain:             float32(audiocodec.DBToLinear(plan.gainDB)),
			outputStartFrame: start,
			outputEndFrame:   end,
		})
	}
	return out, cursor, nil
}

// mixSpanIntoBlock replays, for one output block, the exact per-span loop of
// internal/render/mixer.go: outgoing spans are scaled in place, then the
// incoming weighted sample is added. Frame numbering is absolute, so block
// boundaries do not alter any weight or placement.
func mixSpanIntoBlock(block []float32, blockStart int64, span prepared, channels int) {
	frames := span.clip.FrameCount()
	for frame := int64(0); frame < frames; frame++ {
		outFrame := span.outputStartFrame + frame
		if outFrame < blockStart || outFrame >= blockStart+int64(len(block)/channels) {
			continue
		}
		incoming := float32(1)
		outgoing := float32(1)
		if span.plan.crossfadeFrames > 0 && frame < span.plan.crossfadeFrames {
			progress := (float64(frame) + 0.5) / float64(span.plan.crossfadeFrames)
			angle := progress * math.Pi / 2
			incoming = float32(math.Sin(angle))
			outgoing = float32(math.Cos(angle))
		}
		inStart := int(frame) * channels
		outStart := int(outFrame-blockStart) * channels
		for channel := 0; channel < channels; channel++ {
			value := span.clip.Samples[inStart+channel] * span.gain * incoming
			if span.plan.crossfadeFrames > 0 && frame < span.plan.crossfadeFrames {
				block[outStart+channel] *= outgoing
			}
			block[outStart+channel] += value
		}
	}
}

// ---------------------------------------------------------------------------
// Reference: the legacy whole-resident pipeline, one in-memory full buffer.
// ---------------------------------------------------------------------------

func renderReference(ctx context.Context, c experimentCase) (referenceOutput, error) {
	spans, totalFrames, err := prepare(ctx, c)
	if err != nil {
		return referenceOutput{}, err
	}
	channels := c.outputChannels
	mix := make([]float32, int(totalFrames)*channels)
	for _, span := range spans {
		block := mix
		mixSpanIntoBlock(block, 0, span, channels)
	}
	prePeak := peakAbsolute(mix)
	gain := 1.0
	if prePeak > 0 {
		gain = audiocodec.DBToLinear(c.targetPeakDBFS) / prePeak
	}
	if math.IsNaN(gain) || math.IsInf(gain, 0) || gain <= 0 {
		gain = 1
	}
	for index, sample := range mix {
		value := float64(sample) * gain
		if value > 1 {
			value = 1
		}
		if value < -1 {
			value = -1
		}
		mix[index] = float32(value)
	}
	wav, err := audiocodec.EncodePCM16(c.sampleRate, channels, mix)
	if err != nil {
		return referenceOutput{}, err
	}
	decoded, _, err := audiocodec.DecodeWAV(wav)
	if err != nil {
		return referenceOutput{}, err
	}
	postMetrics := audiocodec.Measure(decoded.Samples)
	receiptSpans := make([]domain.ReceiptSpan, 0, len(spans))
	for index, span := range spans {
		receiptSpans = append(receiptSpans, domain.ReceiptSpan{
			Index:            index,
			SourceStartFrame: span.plan.sourceStartFrame,
			SourceEndFrame:   span.plan.sourceEndFrame,
			OutputStartFrame: span.outputStartFrame,
			OutputEndFrame:   span.outputEndFrame,
			OutputFrames:     span.outputEndFrame - span.outputStartFrame,
			CrossfadeFrames:  span.plan.crossfadeFrames,
			GainDB:           span.plan.gainDB,
		})
	}
	return referenceOutput{
		wav:            wav,
		prePeak:        prePeak,
		gainDB:         audiocodec.LinearToDB(gain),
		postPeakDBFS:   postMetrics.PeakDBFS,
		postRMSDBFS:    postMetrics.RMSDBFS,
		digest:         sha256Hex(wav),
		totalFrames:    totalFrames,
		outputChannels: channels,
		spanRecords:    receiptSpans,
	}, nil
}

// ---------------------------------------------------------------------------
// Correct two-pass streaming scheme.
// Pass A mixes block by block and retains only scalar facts.
// Pass B re-mixes block by block with the now-known global gain and streams
// PCM16 bytes into a writer without ever holding the full signal.
// ---------------------------------------------------------------------------

func renderCorrectTwoPass(ctx context.Context, c experimentCase, blockA, blockB int64) ([]byte, retainedFacts, error) {
	spans, totalFrames, err := prepare(ctx, c)
	if err != nil {
		return nil, retainedFacts{}, err
	}
	channels := c.outputChannels

	// Pass A: mix + pre-normalization facts only.
	facts := retainedFacts{totalFrames: totalFrames, channels: channels}
	for blockStart := int64(0); blockStart < totalFrames; blockStart += blockA {
		size := min64(blockA, totalFrames-blockStart)
		block := make([]float32, int(size)*channels)
		for _, span := range spans {
			if span.outputEndFrame <= blockStart || span.outputStartFrame >= blockStart+size {
				continue
			}
			mixSpanIntoBlock(block, blockStart, span, channels)
		}
		for _, sample := range block {
			absolute := math.Abs(float64(sample))
			if absolute > facts.prePeak {
				facts.prePeak = absolute
			}
			// Match audiocodec.Measure's float32-rounded square.
			facts.sumSquares += measuredSquare(sample)
			facts.sampleCount++
		}
		if err := ctx.Err(); err != nil {
			return nil, retainedFacts{}, err
		}
	}

	gain := 1.0
	if facts.prePeak > 0 {
		gain = audiocodec.DBToLinear(c.targetPeakDBFS) / facts.prePeak
	}
	if math.IsNaN(gain) || math.IsInf(gain, 0) || gain <= 0 {
		gain = 1
	}

	// Pass B: re-mix, apply the global gain, quantize, and stream bytes.
	writer := newByteStreamer(c.sampleRate, channels, totalFrames)
	for blockStart := int64(0); blockStart < totalFrames; blockStart += blockB {
		size := min64(blockB, totalFrames-blockStart)
		block := make([]float32, int(size)*channels)
		for _, span := range spans {
			if span.outputEndFrame <= blockStart || span.outputStartFrame >= blockStart+size {
				continue
			}
			mixSpanIntoBlock(block, blockStart, span, channels)
		}
		// localSumSq is a per-chunk partial: it is merged into the global fact
		// only after the block is done. This is the natural streaming shape and
		// the source of the order-dependent RMS difference we want to expose.
		localSumSq := 0.0
		for _, sample := range block {
			value := float64(sample) * gain
			if value > 1 {
				value = 1
			}
			if value < -1 {
				value = -1
			}
			q := quantizeInt16(float32(value))
			writer.writeSample(q)
			decoded := decodedFloat16(q)
			abs := math.Abs(float64(decoded))
			if abs > facts.postPeak {
				facts.postPeak = abs
			}
			square := measuredSquare(decoded)
			localSumSq += square
			facts.postSumSquaresLegacy += square
			facts.postSamples++
		}
		facts.postSumSquares += localSumSq
	}
	return writer.bytes(), facts, nil
}

// ---------------------------------------------------------------------------
// Wrong scheme W1: single pass; each block normalized against its local peak.
// No global pre-peak exists at emission time, so the blocks cannot share the
// legacy gain. Frame count stays correct while PCM bytes diverge.
// ---------------------------------------------------------------------------

func renderWrongLocalPeak(ctx context.Context, c experimentCase, blockFrames int64) ([]byte, retainedFacts, error) {
	spans, totalFrames, err := prepare(ctx, c)
	if err != nil {
		return nil, retainedFacts{}, err
	}
	channels := c.outputChannels
	target := audiocodec.DBToLinear(c.targetPeakDBFS)
	writer := newByteStreamer(c.sampleRate, channels, totalFrames)
	facts := retainedFacts{totalFrames: totalFrames, channels: channels}
	for blockStart := int64(0); blockStart < totalFrames; blockStart += blockFrames {
		size := min64(blockFrames, totalFrames-blockStart)
		block := make([]float32, int(size)*channels)
		for _, span := range spans {
			if span.outputEndFrame <= blockStart || span.outputStartFrame >= blockStart+size {
				continue
			}
			mixSpanIntoBlock(block, blockStart, span, channels)
		}
		localPeak := peakAbsolute(block)
		localGain := 1.0
		if localPeak > 0 {
			localGain = target / localPeak
		}
		if math.IsNaN(localGain) || math.IsInf(localGain, 0) || localGain <= 0 {
			localGain = 1
		}
		for _, sample := range block {
			value := float64(sample) * localGain
			if value > 1 {
				value = 1
			}
			if value < -1 {
				value = -1
			}
			q := quantizeInt16(float32(value))
			writer.writeSample(q)
			decoded := decodedFloat16(q)
			abs := math.Abs(float64(decoded))
			if abs > facts.postPeak {
				facts.postPeak = abs
			}
			facts.postSumSquares += measuredSquare(decoded)
			facts.postSamples++
		}
	}
	return writer.bytes(), facts, nil
}

// ---------------------------------------------------------------------------
// Wrong scheme W2: blocks mixed in isolation; the crossfade frame counter is
// restarted at the start of every block, so a fade split by a boundary is
// re-evaluated from progress zero instead of using the absolute frame index.
// Placement arithmetic is unchanged, hence the frame count stays correct.
// ---------------------------------------------------------------------------

func renderWrongBlockIsolation(ctx context.Context, c experimentCase, blockFrames int64) ([]byte, retainedFacts, error) {
	spans, totalFrames, err := prepare(ctx, c)
	if err != nil {
		return nil, retainedFacts{}, err
	}
	channels := c.outputChannels

	// The global pre-peak is still observable here (this flaw is independent of
	// the normalization pass structure), so the gain matches the reference;
	// only crossfade weights at boundaries are wrong.
	globalMix := make([]float32, int(totalFrames)*channels)
	for _, span := range spans {
		mixSpanIntoBlock(globalMix, 0, span, channels)
	}
	prePeak := peakAbsolute(globalMix)
	gain := 1.0
	if prePeak > 0 {
		gain = audiocodec.DBToLinear(c.targetPeakDBFS) / prePeak
	}

	writer := newByteStreamer(c.sampleRate, channels, totalFrames)
	facts := retainedFacts{totalFrames: totalFrames, channels: channels}
	for blockStart := int64(0); blockStart < totalFrames; blockStart += blockFrames {
		size := min64(blockFrames, totalFrames-blockStart)
		block := make([]float32, int(size)*channels)
		for _, span := range spans {
			if span.outputEndFrame <= blockStart || span.outputStartFrame >= blockStart+size {
				continue
			}
			mixSpanIsolated(block, blockStart, span, channels)
		}
		for _, sample := range block {
			value := float64(sample) * gain
			if value > 1 {
				value = 1
			}
			if value < -1 {
				value = -1
			}
			q := quantizeInt16(float32(value))
			writer.writeSample(q)
			decoded := decodedFloat16(q)
			abs := math.Abs(float64(decoded))
			if abs > facts.postPeak {
				facts.postPeak = abs
			}
			facts.postSumSquares += measuredSquare(decoded)
			facts.postSamples++
		}
	}
	return writer.bytes(), facts, nil
}

// mixSpanIsolated is the flawed variant: the fade progress uses the frame
// index *within the current block*, restarting at each boundary.
func mixSpanIsolated(block []float32, blockStart int64, span prepared, channels int) {
	frames := span.clip.FrameCount()
	for frame := int64(0); frame < frames; frame++ {
		outFrame := span.outputStartFrame + frame
		if outFrame < blockStart || outFrame >= blockStart+int64(len(block)/channels) {
			continue
		}
		localFrame := frame
		if frame < span.plan.crossfadeFrames {
			// Naive streaming mistake: re-index the fade relative to block start.
			localFrame = outFrame - blockStart
			if localFrame < 0 {
				localFrame = 0
			}
		}
		incoming := float32(1)
		outgoing := float32(1)
		if span.plan.crossfadeFrames > 0 && frame < span.plan.crossfadeFrames {
			progress := (float64(localFrame) + 0.5) / float64(span.plan.crossfadeFrames)
			if progress > 1 {
				progress = 1
			}
			angle := progress * math.Pi / 2
			incoming = float32(math.Sin(angle))
			outgoing = float32(math.Cos(angle))
		}
		inStart := int(frame) * channels
		outStart := int(outFrame-blockStart) * channels
		for channel := 0; channel < channels; channel++ {
			value := span.clip.Samples[inStart+channel] * span.gain * incoming
			if span.plan.crossfadeFrames > 0 && frame < span.plan.crossfadeFrames {
				block[outStart+channel] *= outgoing
			}
			block[outStart+channel] += value
		}
	}
}

// ---------------------------------------------------------------------------
// Cancellation demonstration
// ---------------------------------------------------------------------------

func demonstrateCancellation(c experimentCase) (cancellationReport, error) {
	spans, totalFrames, err := prepare(context.Background(), c)
	if err != nil {
		return cancellationReport{}, err
	}
	channels := c.outputChannels
	tmpDir, err := os.MkdirTemp("", "chunked-render-cancel-*")
	if err != nil {
		return cancellationReport{}, err
	}
	defer os.RemoveAll(tmpDir)
	tmpPath := filepath.Join(tmpDir, "partial.wav")

	// Simulate the production staging discipline: a partial file may exist only
	// inside the private temp directory; nothing is renamed into the published
	// renders/ tree, and on failure the staging directory is removed.
	tmpFile, err := os.OpenFile(tmpPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o640)
	if err != nil {
		return cancellationReport{}, err
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	blockFrames := int64(3)
	emittedFrames := int64(0)
	emittedBytes := int64(0)
	canceled := false
	var cancelErr error
	for blockStart := int64(0); blockStart < totalFrames; blockStart += blockFrames {
		if err := ctx.Err(); err != nil {
			cancelErr = err
			break
		}
		size := min64(blockFrames, totalFrames-blockStart)
		block := make([]float32, int(size)*channels)
		for _, span := range spans {
			if span.outputEndFrame <= blockStart || span.outputStartFrame >= blockStart+size {
				continue
			}
			mixSpanIntoBlock(block, blockStart, span, channels)
		}
		// Append raw little-endian PCM16 for this block, as a streaming encoder
		// would, instead of buffering the whole output in memory.
		pcm := make([]byte, 0, int(size)*channels*2)
		for _, sample := range block {
			q := quantizeInt16(sample)
			pcm = append(pcm, byte(uint16(q)), byte(uint16(q)>>8))
		}
		if _, err := tmpFile.Write(pcm); err != nil {
			_ = tmpFile.Close()
			return cancellationReport{}, err
		}
		emittedBytes += int64(len(pcm))
		emittedFrames += size
		if blockStart == 6 {
			cancel()
			canceled = true
			cancelErr = ctx.Err()
			break
		}
	}
	if err := tmpFile.Close(); err != nil {
		return cancellationReport{}, err
	}
	partialExisted := fileExists(tmpPath)
	partialSize := int64(0)
	if info, statErr := os.Stat(tmpPath); statErr == nil {
		partialSize = info.Size()
	}
	// Failure cleanup: remove the unpublished staging artifact.
	if err := os.Remove(tmpPath); err != nil {
		return cancellationReport{}, err
	}

	// Re-run from scratch with an alive context and confirm byte-identical output.
	restarted, _, err := renderCorrectTwoPass(context.Background(), c, 5, 7)
	if err != nil {
		return cancellationReport{}, err
	}
	ref, err := renderReference(context.Background(), c)
	if err != nil {
		return cancellationReport{}, err
	}
	return cancellationReport{
		CancelObserved:               canceled,
		CancelError:                  fmt.Sprint(cancelErr),
		FramesEmittedBeforeCancel:    emittedFrames,
		PartialPCMBytesBeforeCleanup: emittedBytes,
		PartialFileObservedInStaging: partialExisted,
		PartialFileSize:              partialSize,
		TotalFrames:                  totalFrames,
		NoPublishedArtifact:          !fileExists(tmpPath),
		RestartDigestMatches:         bytesEqual(restarted, ref.wav),
	}, nil
}

// ---------------------------------------------------------------------------
// Small helpers: WAV streaming, quantization, metrics, report structures.
// ---------------------------------------------------------------------------

type wavStreamer struct {
	sampleRate int
	channels   int
	frames     int64
	data       []byte
}

func newByteStreamer(sampleRate, channels int, totalFrames int64) *wavStreamer {
	return &wavStreamer{
		sampleRate: sampleRate,
		channels:   channels,
		frames:     totalFrames,
		data:       make([]byte, 0, 44+int(totalFrames)*channels*2),
	}
}

func (w *wavStreamer) writeSample(q int16) {
	var pair [2]byte
	pair[0] = byte(uint16(q))
	pair[1] = byte(uint16(q) >> 8)
	w.data = append(w.data, pair[0], pair[1])
}

func (w *wavStreamer) bytes() []byte {
	return buildWAV(w.sampleRate, w.channels, w.frames, w.data)
}

// buildWAV reproduces internal/audiocodec.EncodePCM16's 44-byte header exactly,
// with the PCM payload supplied incrementally instead of allocated whole.
func buildWAV(sampleRate, channels int, frames int64, pcm []byte) []byte {
	blockAlign := channels * 2
	byteCount := 44 + len(pcm)
	result := make([]byte, 44)
	copy(result[0:4], "RIFF")
	putUint32LE(result[4:8], uint32(byteCount-8))
	copy(result[8:12], "WAVE")
	copy(result[12:16], "fmt ")
	putUint32LE(result[16:20], 16)
	putUint16LE(result[20:22], 1)
	putUint16LE(result[22:24], uint16(channels))
	putUint32LE(result[24:28], uint32(sampleRate))
	putUint32LE(result[28:32], uint32(sampleRate*blockAlign))
	putUint16LE(result[32:34], uint16(blockAlign))
	putUint16LE(result[34:36], 16)
	copy(result[36:40], "data")
	putUint32LE(result[40:44], uint32(len(pcm)))
	return append(result, pcm...)
}

// quantizeInt16 mirrors the branch structure of EncodePCM16's quantizer so the
// streamed bytes match the legacy encoder sample for sample.
func quantizeInt16(sample float32) int16 {
	var quantized int32
	switch {
	case sample <= -1:
		quantized = -32768
	case sample >= 1:
		quantized = 32767
	case sample < 0:
		quantized = int32(math.Round(float64(sample) * 32768))
	default:
		quantized = int32(math.Round(float64(sample) * 32767))
	}
	if quantized < -32768 {
		quantized = -32768
	}
	if quantized > 32767 {
		quantized = 32767
	}
	return int16(quantized)
}

// decodedFloat16 mirrors internal/audiocodec.DecodeWAV's asymmetric
// reconstruction, including the intermediate float32 rounding: negative codes
// divide by 32768 and non-negative codes by 32767, and the result is stored as
// float32 before any measurement. Post-normalization receipt facts are
// measured on these decoded float32 values, so a streaming renderer must
// reproduce this exact conversion; doing the division in float64 produces
// different receipt numbers even though the emitted PCM bytes are identical.
func decodedFloat16(q int16) float32 {
	if q < 0 {
		return float32(q) / 32768
	}
	return float32(q) / 32767
}

// measuredSquare mirrors audiocodec.Measure exactly: the loop variable is a
// float64 copy of each float32 sample, so the square is a float64 product
// (no intermediate float32 rounding) accumulated into a float64 sum. A
// float32-rounded square would produce a different RMS in the receipt.
func measuredSquare(value float32) float64 {
	number := float64(value)
	return number * number
}

func peakAbsolute(samples []float32) float64 {
	peak := 0.0
	for _, sample := range samples {
		absolute := math.Abs(float64(sample))
		if absolute > peak {
			peak = absolute
		}
	}
	return peak
}

const silenceFloorDBFS = -120.0

func amplitudeDBFS(value float64) float64 {
	if value <= 0 {
		return silenceFloorDBFS
	}
	measured := 20 * math.Log10(value)
	if measured < silenceFloorDBFS {
		return silenceFloorDBFS
	}
	return measured
}

func sha256Hex(data []byte) string {
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:])
}

func bytesEqual(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for index := range a {
		if a[index] != b[index] {
			return false
		}
	}
	return true
}

func frameCountOf(wav []byte) int64 {
	// data chunk payload starts at offset 44 in this encoder; frame count is
	// derived from the data size exactly as the legacy decoder would.
	dataBytes := int64(len(wav) - 44)
	channels := int64(wav[22]) | int64(wav[23])<<8
	return dataBytes / 2 / channels
}

func countSampleDiffs(observed, reference []byte) int {
	diffs := 0
	limit := len(observed)
	if len(reference) < limit {
		limit = len(reference)
	}
	for index := 44; index+1 < limit; index += 2 {
		if observed[index] != reference[index] || observed[index+1] != reference[index+1] {
			diffs++
		}
	}
	if len(observed) != len(reference) {
		diffs += int(math.Abs(float64(len(observed)-len(reference))) / 2)
	}
	return diffs
}

// diffRegions reports which output frame ranges contain differing samples,
// used to show W2's errors cluster around block-split crossfade frames.
func diffRegions(observed, reference []byte, channels int) []frameRegion {
	regions := []frameRegion{}
	totalFrames := int(frameCountOf(reference))
	inRegion := false
	start := 0
	for frame := 0; frame < totalFrames; frame++ {
		differs := false
		for channel := 0; channel < channels; channel++ {
			offset := 44 + (frame*channels+channel)*2
			if observed[offset] != reference[offset] || observed[offset+1] != reference[offset+1] {
				differs = true
				break
			}
		}
		if differs && !inRegion {
			inRegion = true
			start = frame
		}
		if !differs && inRegion {
			inRegion = false
			regions = append(regions, frameRegion{Start: int64(start), End: int64(frame)})
		}
	}
	if inRegion {
		regions = append(regions, frameRegion{Start: int64(start), End: int64(totalFrames)})
	}
	return regions
}

func receiptFactsMatch(facts retainedFacts, ref referenceOutput) bool {
	legacyOrderRMS := amplitudeDBFS(math.Sqrt(facts.postSumSquaresLegacy / float64(facts.postSamples)))
	if amplitudeDBFS(facts.postPeak) != ref.postPeakDBFS {
		return false
	}
	if legacyOrderRMS != ref.postRMSDBFS {
		return false
	}
	if facts.totalFrames != ref.totalFrames {
		return false
	}
	return facts.prePeak == ref.prePeak
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func min64(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

func putUint32LE(target []byte, value uint32) {
	target[0] = byte(value)
	target[1] = byte(value >> 8)
	target[2] = byte(value >> 16)
	target[3] = byte(value >> 24)
}

func putUint16LE(target []byte, value uint16) {
	target[0] = byte(value)
	target[1] = byte(value >> 8)
}

func sortStrings(values []string) {
	for i := 1; i < len(values); i++ {
		for j := i; j > 0 && values[j-1] > values[j]; j-- {
			values[j-1], values[j] = values[j], values[j-1]
		}
	}
}
