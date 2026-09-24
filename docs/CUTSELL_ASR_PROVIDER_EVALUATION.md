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

1. Run the actual repo test suite and worker CI on a full checkout. Here,
   `compileall` of changed Python files, YAML parsing, a manual `jq` payload
   check and four isolated HTTP-fake provider tests passed. The full repo
   test suite could not run without the repository checkout; these narrower
   checks are not a substitute.
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

## Video00 ASR-only run 36016458706 (2026-09-24)

Executed one isolated matrix run on source `Editdna longform validation/VIDEO-2026-07-30-09-18-03.mp4` at SHA `acac420178c055357d51761b55c1d05886633745`, using the same deterministic temperature 0.0 and NVIDIA L4. Both jobs passed structurally; no editor or renderer was invoked. GitHub Actions: https://github.com/AutomatedRetailServices/EditDNA-worker/actions/runs/36016458706 . The temporary branch-only `push` trigger was removed at `3ec5dea87ce738ede15f2bff84ab50de6f8be2ca`; the workflow is manual-only again. Production is unchanged.

| Model | Words | Segments | Elapsed | Observed ending |
| --- | ---: | ---: | ---: | --- |
| Faster-Whisper `medium` | 623 | 54 | 34.283 s | Contains conclusion, percentage and closing advice. One `No quiero sonar a conspiración` segment. |
| Faster-Whisper `large-v3` | 771 | 53 | 55.244 s | From 299.46 s onward, repeats `No quiero sonar a conspiración…` in **20** transcript segments, including near-zero-duration word timestamps at 366.9 s, and omits the actual conclusion and CTA. |

The `large-v3` output is unsafe as a transcription source on Video00 with this runtime/configuration. Workflow `success` means jobs ran, not that transcription is accurate. Do **not** switch production to `large-v3` on this evidence. The `medium` content hash under deterministic config (`asrcontent_dcc16eb97390b41df67327af`) matches one recorded RAW #127 result; a single new `medium` invocation does not establish repeatability. Before selecting a provider, validate against independently listened reference speech, rerun repeatability, and test held-out English and Spanish videos. The catastrophic repeat warrants a generic audio-grounded repetition/zero-duration-word reject gate before any new ASR is allowed to feed selection. This observation does not certify `medium` as fully accurate or solve downstream judge variability.

## Video00 external-provider attempt (2026-09-24)

- Run https://github.com/AutomatedRetailServices/EditDNA-worker/actions/runs/36018990582 (`8a23b432`): the GPT-Transcribe attempt sent the Video00 audio and received HTTP 403 Forbidden from `/v1/audio/transcriptions`; **no transcript was returned**. The matrix default `fail-fast` cancelled Deepgram in that run. HTTP 403 alone does not identify whether the issue is provider permission, key scope, model availability or account policy. Do not label GPT-Transcribe inaccurate from this outcome, and do not repeat the paid attempt until access is resolved.
- Run https://github.com/AutomatedRetailServices/EditDNA-worker/actions/runs/36019295301 (`b69fe9ce`): a preflight of the same RunPod template used by the Modal job found no `DEEPGRAM_API_KEY`; the Deepgram attempt stopped before any Modal GPU or provider request. **No Deepgram transcript exists.**
- The temporary branch-only push trigger was removed in `19533bf328bcb5b00acf69deab46d754d3ab5f2e`. The ASR workflow is manual-only again. Production and the canonical branch remain unchanged.

The external-provider comparison remains unperformed until credentials/access exist. The earlier medium-versus-large-v3 result remains the only successful new Video00 transcription comparison. Avoid interpreting these access failures as accuracy measurements.
