# Uploaded source comparison — 2026-09-24

Run 36062782881 / job 107845430738 completed both authorized full-engine calls.
Test checkout: 89bb9c9bceb40264d5911677de790088b204232a.
Both engine package hashes: 71db0555a03f53c22db47247e40762daaa67e9882a1a38997110de4c425ea59e.
Source: 55.401 seconds, 12,429,383 bytes; SHA256 c9d892629f19d49058246e79bb57eb0af10bdf123416122307aa0bcd5c91cb14.
Same frozen worker template, flags, source and code; only primary ASR provider differs.
No Video00-specific acceptance oracle. No retries. Both Modal apps completed and stopped.

| Metric | Faster-Whisper Medium | GPT Transcribe + WhisperX |
| --- | --- | --- |
| Raw timed words | 168 | 157 |
| Zero-duration words | 10 | 0 |
| Selected fragments | 4 | 1 |
| Selected source duration | 16.792 s | 3.660 s |
| Rendered output | 16.000 s | None |
| Engine elapsed | 100.199 s | 73.037 s (no render) |
| Technical QC | PASS | Not attempted |
| Delivery | BLOCKED: repeated content | BLOCKED: lost content before freeze |

Medium's result contains adjacent repeated shopping-cart calls to action, flagged
PERCEPTUAL_REPEATED_AUDIENCE_CONTENT (similarity 0.898). Its legacy deliverable
boolean is true, but delivery_status and perceptual gate explicitly block it.
The MP4 is therefore supplied only as DIAGNOSTICO. Full local ffmpeg decode passed.
Artifact MP4 SHA256: cca193044685044dc51b90c5af0317396298b99ac958acedd6bee2d9861fd23d.

GPT ASR passed with two requests, 157 aligned words, no fallback and no timestamp
interpolation. Downstream selection retained only source 50.70015–54.36, beginning
“orange shopping cart, then they probably have sold out”. Coherence detected lost
semantic content; repair had no strategy for UNIQUE_FACT_LOST, so freeze/render
were blocked. This is not an API access error or failed alignment.

Lexical token disagreement is 9.4675%, not WER against a human reference.
Medium says “chocolate card” where GPT says “shopping cart”; they also differ in
the last take and trailing repeated text. No human listening occurred here, so
these are differences, not adjudicated transcription errors. No inference of
faster complete editing from GPT's shorter pre-render elapsed time.

Verdict: no provider promotion. GPT's temporal property improved, but the complete
pipeline produced no output; Medium produced a diagnostic output that also failed
editorial delivery. One call per provider cannot establish general superiority or
stability. Production remains Medium; no engine/editor code changed.

Evidence artifacts: uploaded-medium-reports (10835745425), uploaded-medium-A
(10835820321), uploaded-gpt-whisperx-reports (10835815565), and
uploaded-comparison-manifest (10835631236). Original ZIP digests verified locally.
User report embeds all original JSON evidence. Report generator:
benchmarks/report_uploaded_asr_comparison.py.


## Authorized repeat

Run 36064364263 / job 107850529335: same input and exact engine package hash,
new primary ASR requests, completed in 75.722 s. 157 words, zero zero-duration
words; identical selected source range 50.70015–54.36 and text. Again 15
discards, lost semantic content, NEEDS_HUMAN_REVIEW, no freeze or render.
Both GPT trials reproduce the selection failure; no universal reliability claim.
Reports artifact 10834793895, ZIP SHA256
6aac7f0017e905058ec4b7766c4261a065bf786a5dd8aec4064d0b328407ac17 verified.
No paid retry beyond the one newly authorized call. No editor code changed.
