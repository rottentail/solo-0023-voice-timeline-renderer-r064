package main

import (
	"math"
)

// rmsStressReport records a controlled floating-point experiment on the only
// receipt fact whose streaming computation has more than one valid shape:
// the RMS sum of squares. Peak is an order-insensitive max; the sum is not.
type rmsStressReport struct {
	// Sequence: one 1.0 followed by TinyRunCount copies of Tiny, repeated
	// GroupCount times. A running scalar accumulator that meets each tiny value
	// while the running sum is ~1.0 rounds the tiny value away (2^-53 is half
	// an ulp at 1.0 and rounds to even). A chunk whose boundary isolates a run
	// of tiny values sums them while the partial sum is still tiny, so they
	// survive and change the merged total.
	Tiny                       float64   `json:"tiny_value"`
	TinyRunCount               int       `json:"tiny_run_length"`
	GroupCount                 int       `json:"group_count"`
	SequenceLength             int       `json:"sequence_length"`
	SequentialSum              float64   `json:"legacy_single_scalar_sum"`
	RunningScalarChunkedSum    float64   `json:"running_scalar_sum_updated_per_sample"`
	ChunkMergedSums            []float64 `json:"chunk_local_partial_then_merged_sums"`
	ChunkSizes                 []int64   `json:"chunk_sizes"`
	ChunkDriftCount            int       `json:"chunk_sizes_with_sum_drift"`
	RunningScalarMatchesLegacy bool      `json:"running_scalar_matches_legacy_at_every_boundary"`
	SequentialRMS              float64   `json:"legacy_rms_linear"`
	SequentialRMSDBFS          float64   `json:"legacy_rms_dbfs"`
	MaxChunkRMSDeltaDBFS       float64   `json:"max_abs_rms_dbfs_delta_from_chunk_local"`
}

func runRMSSummationStress() rmsStressReport {
	const (
		groups     = 2
		tinyRun    = 4096
		tiny       = 1.0 / (1 << 53) // 2^-53: half an ulp of 1.0
		chunkCount = 8
	)
	sequence := make([]float64, 0, groups*(1+tinyRun))
	for group := 0; group < groups; group++ {
		sequence = append(sequence, 1.0)
		for index := 0; index < tinyRun; index++ {
			sequence = append(sequence, tiny)
		}
	}

	// Legacy shape: audiocodec.Measure walks the whole decoded buffer once and
	// accumulates into one scalar in frame order.
	sequentialSum := 0.0
	for _, value := range sequence {
		sequentialSum += value
	}

	// Streaming shape A: still one scalar, updated per emitted sample in the
	// same frame order. Blocks exist only as work units; no partial sum is ever
	// materialized, so this must equal the legacy sum at every boundary.
	chunkSizes := []int64{1, 2, 3, 7, 64, 4096, 4097, 8194}
	runningScalarChunkedSum := math.NaN()
	runningMatches := true
	for _, size := range chunkSizes {
		running := 0.0
		for start := 0; start < len(sequence); start += int(size) {
			end := start + int(size)
			if end > len(sequence) {
				end = len(sequence)
			}
			for _, value := range sequence[start:end] {
				running += value
			}
		}
		if math.IsNaN(runningScalarChunkedSum) {
			runningScalarChunkedSum = running
		}
		if running != sequentialSum {
			runningMatches = false
		}
	}

	// Streaming shape B: each block keeps its own partial sum of squares, and
	// partials are merged after the block finishes. This is the natural
	// "retain one fact per block" implementation, and it changes the float64
	// operation tree, hence the result.
	mergedSums := make([]float64, 0, chunkCount)
	driftCount := 0
	maxDeltaDB := 0.0
	legacyRMS := math.Sqrt(sequentialSum / float64(len(sequence)))
	for _, size := range chunkSizes {
		partials := make([]float64, 0)
		for start := 0; start < len(sequence); start += int(size) {
			end := start + int(size)
			if end > len(sequence) {
				end = len(sequence)
			}
			partial := 0.0
			for _, value := range sequence[start:end] {
				partial += value
			}
			partials = append(partials, partial)
		}
		merged := 0.0
		for _, partial := range partials {
			merged += partial
		}
		mergedSums = append(mergedSums, merged)
		if merged != sequentialSum {
			driftCount++
			chunkRMS := math.Sqrt(merged / float64(len(sequence)))
			delta := math.Abs(amplitudeDBFS(chunkRMS) - amplitudeDBFS(legacyRMS))
			if delta > maxDeltaDB {
				maxDeltaDB = delta
			}
		}
	}

	return rmsStressReport{
		Tiny:                       tiny,
		TinyRunCount:               tinyRun,
		GroupCount:                 groups,
		SequenceLength:             len(sequence),
		SequentialSum:              sequentialSum,
		RunningScalarChunkedSum:    runningScalarChunkedSum,
		ChunkMergedSums:            mergedSums,
		ChunkSizes:                 chunkSizes,
		ChunkDriftCount:            driftCount,
		RunningScalarMatchesLegacy: runningMatches,
		SequentialRMS:              legacyRMS,
		SequentialRMSDBFS:          amplitudeDBFS(legacyRMS),
		MaxChunkRMSDeltaDBFS:       maxDeltaDB,
	}
}
