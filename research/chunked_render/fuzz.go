package main

import (
	"context"
	"fmt"
	"math/rand"
)

type fuzzResult struct {
	Seed              int64   `json:"seed"`
	OutputChannels    int     `json:"output_channels"`
	TotalFrames       int64   `json:"total_frames"`
	BlockSizesTested  []int64 `json:"block_sizes_tested"`
	CorrectEveryBlock bool    `json:"correct_two_pass_bytes_and_facts_every_block"`
	W1WrongEveryBlock bool    `json:"w1_local_peak_differs_every_block"`
	W2WrongEveryBlock bool    `json:"w2_block_isolation_differs_every_block"`
	ReferenceDigest   string  `json:"reference_digest"`
}

// runFuzz builds seeded, still fully deterministic timelines whose geometry is
// not hand-tuned: random source tones, random span lengths and random valid
// crossfades, including overlaps that put three spans on the same frame. It
// guards against the main experiment overfitting one handcrafted signal.
func runFuzz() ([]fuzzResult, error) {
	seeds := []int64{11, 29, 47, 83}
	results := make([]fuzzResult, 0, len(seeds))
	for _, seed := range seeds {
		c, err := randomizedCase(seed)
		if err != nil {
			return nil, err
		}
		ctx := context.Background()
		ref, err := renderReference(ctx, c)
		if err != nil {
			return nil, err
		}
		blocks := []int64{1, 2, 3, 5, 7, 13}
		correctEveryBlock := true
		w1Wrong := true
		w2Wrong := true
		for _, block := range blocks {
			correct, facts, err := renderCorrectTwoPass(ctx, c, block, block*2+1)
			if err != nil {
				return nil, err
			}
			if !bytesEqual(correct, ref.wav) || !receiptFactsMatch(facts, ref) {
				correctEveryBlock = false
			}
			w1, _, err := renderWrongLocalPeak(ctx, c, block)
			if err != nil {
				return nil, err
			}
			if bytesEqual(w1, ref.wav) {
				w1Wrong = false
			}
			w2, _, err := renderWrongBlockIsolation(ctx, c, block)
			if err != nil {
				return nil, err
			}
			if bytesEqual(w2, ref.wav) {
				w2Wrong = false
			}
		}
		results = append(results, fuzzResult{
			Seed:              seed,
			OutputChannels:    c.outputChannels,
			TotalFrames:       ref.totalFrames,
			BlockSizesTested:  blocks,
			CorrectEveryBlock: correctEveryBlock,
			W1WrongEveryBlock: w1Wrong,
			W2WrongEveryBlock: w2Wrong,
			ReferenceDigest:   ref.digest,
		})
	}
	return results, nil
}

func randomizedCase(seed int64) (experimentCase, error) {
	rng := rand.New(rand.NewSource(seed))
	const rate = 48000
	channels := 1 + int(seed%2) // alternate mono / stereo output
	sourceCount := 3
	defs := make([]sourceDef, sourceCount)
	sourceFrames := 36 + rng.Intn(12)
	for index := range defs {
		defs[index] = sourceDef{
			name:       fmt.Sprintf("fz%d", index),
			rate:       rate,
			channels:   channels,
			frequency:  160 + rng.Float64()*600,
			amplitude:  0.2 + rng.Float64()*0.6,
			phase:      rng.Float64() * 3.14,
			frameCount: sourceFrames,
		}
	}
	wavs := make(map[string][]byte, sourceCount)
	for index, def := range defs {
		wav, err := deterministicWAV(def)
		if err != nil {
			return experimentCase{}, err
		}
		wavs[fmt.Sprintf("src%d", index)] = wav
	}

	// Build spans with cumulative output positions. All sources share the
	// timeline rate, so output frames equal selected source frames. Every
	// crossfade is kept strictly inside both neighboring spans; the second
	// crossfade is chosen large enough to create triple-overlap frames.
	spanCount := 3
	plans := make([]spanPlan, spanCount)
	lengths := make([]int64, spanCount)
	for index := range lengths {
		lengths[index] = int64(14 + rng.Intn(10)) // 14..23 frames
	}
	prevEnd := int64(0)
	_ = prevEnd
	for index := range plans {
		startFrame := int64(rng.Intn(6))
		endFrame := startFrame + lengths[index]
		if endFrame > int64(sourceFrames) {
			endFrame = int64(sourceFrames)
		}
		crossfade := int64(0)
		if index > 0 {
			limit := lengths[index-1]
			if lengths[index] < limit {
				limit = lengths[index]
			}
			// keep at least one non-faded frame on each side
			if limit > 2 {
				crossfade = 2 + rng.Int63n(limit-2)
			} else {
				crossfade = limit
			}
		}
		plans[index] = spanPlan{
			name:             fmt.Sprintf("fzspan%d", index),
			sourceStartFrame: startFrame,
			sourceEndFrame:   endFrame,
			gainDB:           -6 + rng.Float64()*8, // -6..+2 dB
			crossfadeFrames:  crossfade,
		}
	}
	// Enlarge the final crossfade so a strict three-span overlap exists when
	// the geometry allows it; clamp to both neighbor span lengths minus one.
	if spanCount >= 3 {
		bound := plans[2].sourceEndFrame - plans[2].sourceStartFrame
		if plans[1].sourceEndFrame-plans[1].sourceStartFrame < bound {
			bound = plans[1].sourceEndFrame - plans[1].sourceStartFrame
		}
		if bound > 3 {
			plans[2].crossfadeFrames = bound - 1
		}
	}
	return experimentCase{
		name:           fmt.Sprintf("fuzz_%d", seed),
		wavs:           wavs,
		plans:          plans,
		sampleRate:     rate,
		outputChannels: channels,
		targetPeakDBFS: -1.5,
	}, nil
}
