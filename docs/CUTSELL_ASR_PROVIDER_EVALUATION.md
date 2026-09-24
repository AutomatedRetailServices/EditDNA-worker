# ASR provider evaluation and replay — isolated checkpoint

Branch: `feat/asr-provider-evaluation-and-replay`, based on `5f831bce`.
No production provider, canonical branch, Selection rule, or paid workflow has changed.

## Evidence

Five sequential RAW #127 attempts used the same source-media SHA, code SHA and
ASR-config fingerprint, yet produced five distinct ASR content hashes and
different source segment counts (48, 44, 54, 54, 54). This demonstrates an
upstream source of variation, not that ASR accounts for every downstream change.

## Implemented in this branch

- The real RAW validation entry point now constructs ASR through the existing
  `load_asr_provider_from_env` policy. Before this change it constructed
  `FasterWhisperASR(model_name=...)` directly, bypassing the deterministic
  configuration flag. With the flag unset, legacy decoding is unchanged.
- The existing ASR-only Modal harness accepts an explicit reviewed model choice:
  `medium` or `large-v3`; its default stays `medium`. This does not alter the
  edit pipeline or the production configuration.
- The same manually dispatched ASR-only harness can ask for an optional
  `gpt-transcribe` or Deepgram `nova-3` multilingual text candidate. Audio is
  extracted locally, sent only when requested, with a fixed endpoint and
  explicit credential/failure checks. Candidate text carries
  `selection_authority=False` and no word timestamps; it never feeds a cut.
- Tests cover opt-in behavior, model allowlist, fail-closed provider errors and
  English/Spanish candidate handling with fake HTTP.

## Gates still open

1. Run the actual repo test suite and worker CI on a full checkout. Syntax-only
   verification from this environment is not a substitute.
2. Inventory diverse English, Spanish, and mixed-language source files from the
   existing S3 folder. Establish independently reviewed spoken-word references,
   including negations, numbers, omissions and audio-adjacent speech.
3. Dispatch bounded **ASR-only** comparisons after verifying credentials and
   cost. A manual `workflow_dispatch` is required; no such paid run occurred
   in this checkpoint.
4. Select the transcription provider using measured errors and failure rates.
   `gpt-transcribe` does not provide the word-aligned evidence that Boundary
   requires here; independently validate a timed alignment (e.g. WhisperX)
   against audio before routing its text into the real editor.
5. Persist a typed source-hash/model/alignment snapshot in existing private
   storage and replay the editorial judge on exactly that snapshot. A cache
   must be scoped to the authorized tenant and use atomic creation rather
   than race-prone get/then/put. No such cache is live yet.
6. Cross-video acceptance on held-out source videos, then real rendered MP4
   Watch+Listen before changing production defaults.

A more accurate transcript by itself does not prove repeated BestTake decisions
or a clean edit. Production still uses Faster-Whisper `medium` on this branch.
