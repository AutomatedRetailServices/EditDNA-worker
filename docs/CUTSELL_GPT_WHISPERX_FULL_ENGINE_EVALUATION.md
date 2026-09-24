# GPT Transcribe + WhisperX: full-engine RAW experiment

Status (2026-09-24): authorized full-engine run completed, MP4 produced, editorial acceptance FAILED. No production promotion or PR #25 merge.

User authorization: “ok entonces hagamos gpt + whisperx dentro del motor completo a ver como se comporta”. Experimental branch `feat/gpt-whisperx-video00`, based on evaluation baseline `53b05f32876ccb70e5c7f9ecbc6a725f4b01182b`.

## Scope and evidence contract

The existing `run_op("focused")` / universal validation / Flow B / Freeze / Boundary / render-QC chain remains the single editor. `CUTSELL_VALIDATION_ASR_PROVIDER=gpt-transcribe-whisperx` opts only the RAW entry point into the new provider. The normal provider and all production defaults remain unchanged. No selection, semantic-protection, Freeze, boundary or render authorities are changed.

`gpt-transcribe` supplies text from contiguous mono 16 kHz PCM chunks covering the complete source. Boundaries prefer measured quiet midpoints (no silence merging); target 30 seconds, maximum 40. Hard boundaries are disclosed. Chunking bounds Wav2Vec2 alignment memory. Unlike the earlier text-only comparison, GPT is called per chunk: this is a confound to report, not proof that chunking improved lexical accuracy. No prompt, keywords or Video00 text is injected.

WhisperX 3.8.6 aligns the returned words using its TorchAudio EN/ES models. The separate `/opt/cutsell-whisperx` environment avoids replacing the clean worker's Torch, Faster-Whisper, Numpy or MediaPipe dependencies. Both aligners and NLTK data are prepared during CPU image build, before a paid L4 invocation. No Whisper transcription or diarization runs in this sidecar. Interpolation is explicitly `ignore`; every GPT whitespace token must survive in order with finite, positive, ordered times within its audio chunk. Missing or invalid evidence fails the experiment, without another provider or guessed timestamps. Numeric tokens remain verbatim; wildcard alignment does not prove spoken numerical correctness.

The audit retains original provider text, normalized whitespace, chunk bounds, request IDs, language, model/package versions, GPU, timing counts, elapsed time and configuration fingerprint. The full result retains the canonical source SHA-256 and typed timed-ASR snapshot. The alignment child receives no API/AWS credentials. Ordinary project inference credentials come from the existing worker template; admin keys are rejected. API errors do not dump credential-bearing request objects. Unsupported language, HTTP or alignment failures are observable and fail closed.

For baseline parity, the existing post-render speech-safe microtrim option remains enabled. That separate cleanup component continues to use its existing Medium model; this experiment changes the primary transcription/timing evidence, not every optional ASR consumer in the application.

## Preflight and cost bounds

- 135 targeted provider, entry point, text comparison, dependency boundary, Modal persistence/image, timing reconciliation, and complete CleanCutBench/parity regression tests passed in 1.95 seconds. Initial wrapper tests required adding the missing boto3 test dependency in a scratch-only environment; no application dependency was changed.
- Distinct skeptical self-review in this session (not independent human release certification): reviewed the engine fingerprint keyword contract, token loss/reordering, timestamp interpolation, credentials, dependency isolation, bounded chunk coverage, default provider preservation, exact build identity and single paid invocation.
- Existing L4 full benchmark, `retries=0`, 5400-second bound, scale-to-zero and persisted-result deduplication. No overlapping full RAW run. No new persistent infrastructure.
- Credential gate before GPU; no production configuration changes. The temporary one-run push trigger is restored to manual-only immediately after registration.
- Workflow success is not editorial acceptance. Actual output, selected/discarded text, cut boundaries, protected meaning and QC must be reviewed. A blocked/invalidated render must not be presented as a finished edit.

## Source and comparison

Video00 source SHA-256: `b37059b1790cf3eb0447bb54595cc99f6668aadc5f65dafa9cdf6181621ef9f5` (366.997 seconds).

Earlier same-source text comparison: run `36040003462`, GPT 624 whitespace words, Medium 643 whitespace words (650 canonical normalized words); Medium had 17 zero-duration words, including 10 in the repeated-acne sentence. Those are ASR observations, not a full-engine GPT result. GPT lexical accuracy was not certified by listening. No claim of human audio review may be made from these tests.

Local source preflight produced 13 contiguous chunks. Twelve internal/terminal boundaries use silence or EOF; one internal boundary at 314.813 seconds is a bounded hard window and must be reviewed for a split word. This observed timestamp is QA evidence only, never runtime logic.

Full-engine run `36045181502`, job `107786869693`, exact code head `23845fa143b27e80cf972c85ed1812b228960ceb`: completed. Manual-only trigger restored at `212a5a4b8d74fe2f599ccb863efe77d8e60624de`.

The first registration (`36045050403`) was rejected as YAML before any job/GPU/provider call: a colon inside an unquoted job-if scalar. The folded-scalar correction was used for the actual run; this was not a second paid benchmark.

Qualification caveats to retain in the verdict: downstream Word.confidence carries the CTC alignment score (not GPT lexical confidence), so existing signal scoring receives a different confidence distribution. The historical full-engine Medium result #127 (`35945070839`, 145.97 s per current-state record) predates offline D-291.12 fixes; it is not an identical-code A/B control for this experiment.


## Observed full-engine result

- Source SHA-256 matches the earlier source exactly. Worker build SHA and 316-module package fingerprint match the workflow; package fingerprint `832f15d9aa5b2d3477046cf47f05b2e4b3fcac5633f0df78042a280ee62d3a7f`.
- Primary provider in the actual result: `gpt-transcribe+whisperx-3.8.6`; 13 chunks, 643 aligned words, 45 raw segments. Complete token identity/order retained; zero zero-duration words. No timing interpolation or fallback. One internal hard boundary at 314.813 s remains a qualification limitation.
- Runtime proof: NVIDIA L4, WhisperX 3.8.6, Torch/TorchAudio 2.8.0, Numpy 2.2.6, Transformers 4.57.6; Spanish `VOXPOPULI_ASR_BASE_10K_ES`. Last alignment pass 16.958 s; last entire ASR pass 45.732 s. These are LAST-PASS times, not total job ASR time.
- Existing engine completed in 460.941 s before post-render microtrim/upload. Selected 20 clips, discarded 21; final MP4 138.2 s (2:18.2). Microtrim performed zero cuts. Technical render QC PASS in one render, live Selection/Boundary contract verified and matching the reviewed plan. Source negation `No quiero...` and final advice are present in selected text. The complete acne sentence has a roughly 4.9-second span, not the prior Medium 0.44-second anomaly.
- Technical delivery status: `DELIVERABLE_PENDING_HUMAN_WATCH_LISTEN`. Perceptual review has 3 UNCERTAIN findings (one pause, two edge gestures), zero confirmed FAIL, and four unimplemented capabilities. This is NOT human/audio approval.
- Editorial acceptance: 8/11 pass, 3 fail. Kept abandoned stomach attempt (source 251.8445–253.5465), missed the required full gynecologist take (selected a different take at 82.76805–89.34905), and kept the percentage restatement (338.072–341.856).
- Historical Gold regression: 17/18 pass; `pimples_micro_2_present` missing. The historical 23-clip selection-lock manifest also failed; do not confuse that oracle mismatch with a mutation of the live frozen plan, whose contract verified successfully.
- Overall GitHub workflow conclusion FAILED: editorial/Gold checks above plus legacy Pacing V2 diagnostic extractors missing their optional blocks. The quality-ladder step printed a traceback. Architecture qualification passed. Do not relabel the workflow green because an MP4 exists.
- Residual timing concern: 4 source words exceed 1.5 s; `resolvía.` spans 180.000–185.423 s (5.423 s), in an unselected attempt. 21 words are shorter than 25 ms. Removing zero durations does not prove accurate phoneme boundaries.
- No direct audio listening was possible in this environment. Lexical correctness and the human Watch+Listen gate remain unverified. Current verdict: promising primary ASR/alignment integration, not qualified to replace the production provider or to certify automatic editorial quality.

## Observed duplicate ASR call and post-run correction

The run log contains TWO 13-request full-source passes. This is explained by the real engine: `final_boundary_authority._source_words()` calls the same provider on the full source again before Freeze. Thus this run made 26 primary transcription requests, not 13. The final `asr_provider_audit` is the last pass; its concatenated text matches the primary timed-ASR snapshot. Billing was not queried, so no exact dollar-cost claim is made.

After collecting this result, the experimental provider was corrected to reuse immutable successful evidence within the same provider/job, keyed by actual source SHA-256, source identity, requested language and config fingerprint. This avoids re-decoding/paying twice for identical source evidence. A changed source file or a deliberately extracted verification window is a cache miss; failures are never cached. Alignment confidence is now explicitly labeled as a CTC score. Cache behavior is fingerprinted and counted. 37 relevant offline tests passed after the change, including exact evidence reuse and invalidation when bytes change. No second paid run was launched; this correction is NOT represented as live-tested in run 36045181502.
