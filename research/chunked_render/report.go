package main

import (
	"fmt"
	"strings"
)

type frameRegion struct {
	Start int64 `json:"start_frame"`
	End   int64 `json:"end_frame_exclusive"`
}

type blockResult struct {
	BlockFrames                 int64         `json:"block_frames"`
	CorrectBytesEqual           bool          `json:"correct_two_pass_bytes_equal"`
	CorrectDigestEqual          bool          `json:"correct_two_pass_digest_equal"`
	ReceiptFactsMatch           bool          `json:"correct_two_pass_receipt_facts_match"`
	FactsPrePeak                float64       `json:"retained_pre_peak"`
	FactsPostPeakDBFS           float64       `json:"retained_post_peak_dbfs"`
	FactsPostRMSDBFS            float64       `json:"retained_post_rms_dbfs_chunk_local_accumulation"`
	FactsPostRMSLegacyOrderDBFS float64       `json:"retained_post_rms_dbfs_legacy_order_accumulation"`
	MaxResidentMixSamples       int64         `json:"max_resident_mix_samples"`
	W1BytesEqual                bool          `json:"w1_local_peak_bytes_equal"`
	W1DigestEqual               bool          `json:"w1_local_peak_digest_equal"`
	W1PostPeakDBFS              float64       `json:"w1_post_peak_dbfs"`
	W1SampleDiffs               int           `json:"w1_differing_pcm_samples"`
	W1FramesStillEqual          bool          `json:"w1_frame_count_still_correct"`
	W2BytesEqual                bool          `json:"w2_block_isolation_bytes_equal"`
	W2DigestEqual               bool          `json:"w2_block_isolation_digest_equal"`
	W2PostPeakDBFS              float64       `json:"w2_post_peak_dbfs"`
	W2SampleDiffs               int           `json:"w2_differing_pcm_samples"`
	W2FramesStillEqual          bool          `json:"w2_frame_count_still_correct"`
	W2DiffFrameRegions          []frameRegion `json:"w2_diff_frame_regions"`
}

type controlResult struct {
	BlockFrames      int64  `json:"block_frames"`
	W2BytesEqual     bool   `json:"w2_bytes_equal_without_crossfade"`
	FramesStillEqual bool   `json:"frame_count_correct"`
	ReferenceDigest  string `json:"reference_digest"`
	ObservedDigest   string `json:"observed_digest"`
}

type cancellationReport struct {
	CancelObserved               bool   `json:"cancel_observed"`
	CancelError                  string `json:"cancel_error"`
	FramesEmittedBeforeCancel    int64  `json:"frames_emitted_before_cancel"`
	PartialPCMBytesBeforeCleanup int64  `json:"partial_pcm_bytes_emitted"`
	PartialFileObservedInStaging bool   `json:"partial_file_observed_inside_tmp_only"`
	PartialFileSize              int64  `json:"partial_file_size_bytes"`
	TotalFrames                  int64  `json:"total_frames"`
	NoPublishedArtifact          bool   `json:"no_published_artifact_after_cleanup"`
	RestartDigestMatches         bool   `json:"restart_from_scratch_matches_reference"`
}

type report struct {
	ReferenceDigest       string             `json:"reference_wav_digest"`
	ReferenceTotalFrames  int64              `json:"reference_total_frames"`
	ReferencePrePeak      float64            `json:"reference_pre_peak"`
	ReferenceGainDB       float64            `json:"reference_normalization_gain_db"`
	ReferencePostPeakDBFS float64            `json:"reference_post_peak_dbfs"`
	ReferencePostRMSDBFS  float64            `json:"reference_post_rms_dbfs"`
	BlockSizes            []int64            `json:"block_sizes"`
	Results               []blockResult      `json:"block_results"`
	ControlNoFade         []controlResult    `json:"no_crossfade_control"`
	Cancellation          cancellationReport `json:"cancellation"`
	RMSSummation          rmsStressReport    `json:"rms_summation_order_stress"`
	Fuzz                  []fuzzResult       `json:"seeded_random_signal_fuzz"`
	Summary               map[string]string  `json:"summary"`
}

func summarize(r *report) map[string]string {
	summary := map[string]string{}
	correctAll := true
	w1AllWrong := true
	w2AllWrong := true
	w2FadeFreeControl := true
	for _, entry := range r.Results {
		if !entry.CorrectBytesEqual || !entry.CorrectDigestEqual || !entry.ReceiptFactsMatch {
			correctAll = false
		}
		if entry.W1BytesEqual {
			w1AllWrong = false
		}
		if entry.W2BytesEqual {
			w2AllWrong = false
		}
	}
	for _, control := range r.ControlNoFade {
		if !control.W2BytesEqual {
			w2FadeFreeControl = false
		}
	}
	summary["correct_two_pass_byte_identical_every_boundary"] = boolLabel(correctAll)
	summary["w1_local_peak_rejected_every_boundary"] = boolLabel(w1AllWrong)
	summary["w2_block_isolation_rejected_every_boundary"] = boolLabel(w2AllWrong)
	summary["w2_invisible_without_crossfade_control"] = boolLabel(w2FadeFreeControl)
	summary["wrong_schemes_keep_frame_count_but_break_bytes"] = boolLabel(
		allFramesCorrectButBytesWrong(r),
	)
	summary["cancellation_leaves_no_published_fact_and_restart_matches"] = boolLabel(
		r.Cancellation.CancelObserved && r.Cancellation.NoPublishedArtifact && r.Cancellation.RestartDigestMatches,
	)
	chunkLocalDrifts := 0
	legacyOrderExact := true
	for _, entry := range r.Results {
		if entry.FactsPostRMSDBFS != r.ReferencePostRMSDBFS {
			chunkLocalDrifts++
		}
		if entry.FactsPostRMSLegacyOrderDBFS != r.ReferencePostRMSDBFS {
			legacyOrderExact = false
		}
	}
	summary["chunk_local_rms_drifts_at_boundary_count"] = fmt.Sprintf("%d/%d", chunkLocalDrifts, len(r.Results))
	summary["legacy_order_rms_exact_every_boundary"] = boolLabel(legacyOrderExact)
	summary["rms_stress_running_scalar_matches_legacy"] = boolLabel(r.RMSSummation.RunningScalarMatchesLegacy)
	summary["rms_stress_chunk_local_drift_boundary_count"] = fmt.Sprintf("%d/%d",
		r.RMSSummation.ChunkDriftCount, len(r.RMSSummation.ChunkSizes))
	fuzzCorrect := len(r.Fuzz) > 0
	fuzzW1 := len(r.Fuzz) > 0
	fuzzW2 := len(r.Fuzz) > 0
	for _, f := range r.Fuzz {
		if !f.CorrectEveryBlock {
			fuzzCorrect = false
		}
		if !f.W1WrongEveryBlock {
			fuzzW1 = false
		}
		if !f.W2WrongEveryBlock {
			fuzzW2 = false
		}
	}
	summary["fuzz_correct_two_pass_on_all_seeded_signals"] = boolLabel(fuzzCorrect)
	summary["fuzz_w1_rejected_on_all_seeded_signals"] = boolLabel(fuzzW1)
	summary["fuzz_w2_rejected_on_all_seeded_signals"] = boolLabel(fuzzW2)
	return summary
}

func allFramesCorrectButBytesWrong(r *report) bool {
	for _, entry := range r.Results {
		if !entry.W1FramesStillEqual || !entry.W2FramesStillEqual {
			return false
		}
		if entry.W1BytesEqual || entry.W2BytesEqual {
			return false
		}
	}
	return true
}

func boolLabel(value bool) string {
	if value {
		return "yes"
	}
	return "no"
}

func printConsole(r *report, path string) {
	var builder strings.Builder
	builder.WriteString(fmt.Sprintf("reference digest=%s frames=%d prePeak=%.6f gain=%.6f dB postPeak=%.4f dBFS postRMS=%.4f dBFS\n",
		r.ReferenceDigest, r.ReferenceTotalFrames, r.ReferencePrePeak, r.ReferenceGainDB,
		r.ReferencePostPeakDBFS, r.ReferencePostRMSDBFS))
	builder.WriteString(fmt.Sprintf("%-6s %-9s %-9s %-9s %-8s %-8s %-10s %-10s %-12s %-12s\n",
		"block", "correct", "W1bad", "W2bad", "W1diff", "W2diff", "W1frames", "W2frames", "rmsChunk", "rmsLegacy"))
	for _, entry := range r.Results {
		builder.WriteString(fmt.Sprintf("%-6d %-9t %-9t %-9t %-8d %-8d %-10t %-10t %-12.6f %-12.6f\n",
			entry.BlockFrames,
			entry.CorrectBytesEqual,
			!entry.W1BytesEqual,
			!entry.W2BytesEqual,
			entry.W1SampleDiffs,
			entry.W2SampleDiffs,
			entry.W1FramesStillEqual,
			entry.W2FramesStillEqual,
			entry.FactsPostRMSDBFS,
			entry.FactsPostRMSLegacyOrderDBFS,
		))
	}
	builder.WriteString("\nW2 differing frame regions (cross-boundary fade corruption):\n")
	for _, entry := range r.Results {
		builder.WriteString(fmt.Sprintf("  block=%-3d regions=%v\n", entry.BlockFrames, entry.W2DiffFrameRegions))
	}
	builder.WriteString("\nNo-crossfade control (W2 must be harmless):\n")
	for _, control := range r.ControlNoFade {
		builder.WriteString(fmt.Sprintf("  block=%-3d w2BytesEqual=%t framesCorrect=%t\n",
			control.BlockFrames, control.W2BytesEqual, control.FramesStillEqual))
	}
	builder.WriteString(fmt.Sprintf("\nCancellation: %+v\n", r.Cancellation))
	builder.WriteString("\nRMS summation-order stress (squares treated as generic float64 values):\n")
	builder.WriteString(fmt.Sprintf("  legacy sequential sum = %.20g\n", r.RMSSummation.SequentialSum))
	builder.WriteString(fmt.Sprintf("  running scalar sum    = %.20g (matches legacy=%t)\n",
		r.RMSSummation.RunningScalarChunkedSum, r.RMSSummation.RunningScalarMatchesLegacy))
	for index, size := range r.RMSSummation.ChunkSizes {
		builder.WriteString(fmt.Sprintf("  chunk=%-5d merged partial sum = %.20g drifts=%t\n",
			size, r.RMSSummation.ChunkMergedSums[index],
			r.RMSSummation.ChunkMergedSums[index] != r.RMSSummation.SequentialSum))
	}
	builder.WriteString(fmt.Sprintf("  max RMS delta from chunk-local merging = %.3e dBFS\n",
		r.RMSSummation.MaxChunkRMSDeltaDBFS))
	builder.WriteString("\nSeeded random-signal fuzz:\n")
	for _, f := range r.Fuzz {
		builder.WriteString(fmt.Sprintf("  seed=%-3d ch=%d frames=%-3d correct=%t w1Rejected=%t w2Rejected=%t\n",
			f.Seed, f.OutputChannels, f.TotalFrames, f.CorrectEveryBlock, f.W1WrongEveryBlock, f.W2WrongEveryBlock))
	}
	builder.WriteString("\nSummary:\n")
	for key, value := range r.Summary {
		builder.WriteString(fmt.Sprintf("  %s = %s\n", key, value))
	}
	builder.WriteString("\nfull report: " + path + "\n")
	fmt.Print(builder.String())
}
