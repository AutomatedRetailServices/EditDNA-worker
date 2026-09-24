# GPT Transcribe + WhisperX: full-engine RAW experiment

Status (2026-09-24): implementation qualified offline; one authorized Video00 full-engine run pending. No production promotion or PR #25 merge.

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

Full-engine run/result: pending.
