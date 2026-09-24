# ASR provider evaluation and replay — isolated checkpoint

Branch: `feat/asr-provider-evaluation-and-replay`, based on `5f831bce`.
No production provider, canonical branch, or Selection rule has changed. The isolated, authorized paid comparisons and temporary workflow triggers are recorded below.

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

## Correction: OpenAI project had model access; Video00 GPT-4o result

The earlier GPT-Transcribe 403 was **not evidence of missing prepaid credit**. A no-audio/no-GPU access probe, run https://github.com/AutomatedRetailServices/EditDNA-worker/actions/runs/36020072282 , using the same worker-template `OPENAI_API_KEY` returned HTTP 404 for `GET /v1/models/gpt-transcribe` but HTTP 200 for `GET /v1/models/gpt-4o-transcribe`. This identifies the accessible model ID for this project; the previous 403 body was not retained, so its exact server-side reason remains unproven. The earlier no-GPU probe 36019979313 failed at the RunPod request because Python urllib received 403 where the workflow's existing curl path works; the corrected probe used curl. No GPU ran in either probe.

The isolated Video00 test https://github.com/AutomatedRetailServices/EditDNA-worker/actions/runs/36020263828 (SHA `338bdd061eb1c8066f1259e9b26bb168ea09a40d`) successfully ran `gpt-4o-transcribe` as **text-only candidate** alongside the existing deterministic `medium` transcript. GPT-4o returned about 624 whitespace-separated words in 16.379 seconds of provider time. It preserved the negation, a complete conclusion, the hereditary 5–10% statement and the final advice, with no runaway sentence repetition. It produced suspect details to verify against original audio, including `resorción` where the existing medium reads `resorcina`, and a phrase after `2023` transcribed as `hay que votar`. One Video00 output does not establish accuracy or repeatability over other videos. The external text has no word-level alignment, is marked `selection_authority=False`, and was not fed to the editor. The `medium` result included 623 normalized words; its normalization differs from GPT's raw whitespace count, so those totals are not a word-error-rate comparison. Production stays on `medium`.

The temporary push trigger was removed at `743c74a1a6a3cc8f108c3dcb61ed198d85939fc9`. The stable manual workflow now names `gpt-4o-transcribe` (the accessible model) for future opt-in tests. Next: listen to the disputed words, verify enough held-out English and Spanish originals, measure provider repeatability, then test audio-grounded word alignment before allowing any provider into cutting authority.

## GPT-4o repeatability check on same Video00 bytes

Run https://github.com/AutomatedRetailServices/EditDNA-worker/actions/runs/36021387943 (SHA `8912cda5831a906596ae2a2590b8a1596ab5530b`) repeated the same isolated GPT-4o comparison on Video00, with deterministic `medium` alongside. Both GPT responses preserve the ending and contain one negation clause; however, GPT returned 624 versus 622 whitespace-delimited words and two different texts (character similarity about 0.9932). Editorially significant substitutions include `asintomática` versus `sintomática` (prefix changes meaning), `resorción` versus `resorciona` (both suspect against `medium`'s `resorcina`), and `hay que votar` versus `hay que borrar` near 2023. These are observed differences, **not** audio-verified winner labels. One repeat does not quantify a general instability rate; it falsifies a claim of byte-identical GPT output on this source.

Do not promote GPT-4o as the edit authority on this evidence, and do not infer accuracy from high whole-text similarity. Need an audio-reviewed reference for critical spans, then either pin and persist one reviewed source transcript with provenance and downstream replay or route unresolved critical words to review before destructive editing. WhisperX alignment can test word boundaries only after choosing the spoken words; it cannot decide `asintomática` versus `sintomática` from transcript similarity. Trigger restored to manual-only at `7901213267db3250f329c9a739ccfa888d448b09`. No full-video RAW and no production provider switch.

## Exact GPT-Transcribe access denial (synthetic probe, no GPU)

Run https://github.com/AutomatedRetailServices/EditDNA-worker/actions/runs/36022529122 (SHA `1b8d2e7fd095c434b5feeb39561a561e3e256125`) used the existing worker project key to POST less than one second of generated silence to `/v1/audio/transcriptions`, without Video00 or Modal. The response was HTTP 403, `error_type=invalid_request_error`, `error_code=model_not_found`, with the precise provider explanation that **the project associated with the key does not have access to `gpt-transcribe`**. Request id: `req_4cb4b10930174d15a319d558ba84067a`. This resolves the earlier ambiguity: the model exists in the provider's official docs and the endpoint is correct, while the worker's project cannot access it. The account balance was not measured by the probe and must not be asserted as the cause. Previous read-only model checks returned 404 for this ID and 200 for `gpt-4o-transcribe` using this same key.

Access remediation is an account/project decision outside this branch: inspect the usage tier and model availability in the API organization/project actually associated with the worker key (a balance in a different project, or promotional credits, does not establish this project's model access). If the eligible project has access, use its authorized scoped key; otherwise request model access from provider support citing the request id. Do not retry full Video00 on this model until the same no-original preflight succeeds. Temporary push trigger removed at `8a7a5630528f355d020c336f0ad288e41e18ca37`; no production key, model, canonical branch or editor output changed.


## GPT-Transcribe access resolved; authorized Video00 comparison

On 2026-09-24 the user authorized adding only `gpt-transcribe` to the worker
project model allowlist. Read-back verified that model was added and no existing
model was removed. The same worker-key synthetic probe then returned HTTP 200
at 17:59:28 UTC (run 36022529122, job 107763146014,
request `req_94b30788cdd24c74865e6827bc00673b`). No credentials changed in code.

The user subsequently authorized one Video00 ASR-only comparison of
`gpt-transcribe` versus Faster-Whisper `medium`. The adapter now accepts both
OpenAI model IDs; the new model uses `languages[]` only when a hint is supplied.
This run preserves the prior automatic language detection, deterministic
Whisper temperature 0, source video, MP3 comparator extraction and L4 runtime.
Whisper reads the original video; GPT receives its full mono 16 kHz, 64 kbps MP3
extraction, as in the prior GPT-4o comparisons. Thus this compares the existing
integration paths, not identical encoded input bytes or model-only latency.

Pre-run qualification: 44 targeted adapter/harness/Modal/dependency-boundary
tests passed. A separate skeptical self-review in the same session checked
opt-in routing, model-specific parameters, HTTP failure propagation, lack of
selection authority, no retries, one GPU job, fixed endpoint and source, and
unchanged production defaults. This is self-review, not independent release
certification. A commit-message/first-attempt guarded branch-only push trigger
launches this one authorized comparison and is removed immediately afterward.
No editor, renderer, production deployment or new persistent resource is involved.


## Video00 GPT-Transcribe versus Medium — completed 2026-09-24

Run https://github.com/AutomatedRetailServices/EditDNA-worker/actions/runs/36040003462
completed successfully on SHA `bb80b0d791108d426c6d7a5f8b53478912dda631`, job
`107769520233`. The single authorized comparison transcribed the full 366.997 s
Video00 using automatic language detection and the existing integration paths.
The temporary trigger was removed at `d47221f987e42aa23010b982affa7b59a021464a`;
the workflow is again manual-only. No production/canonical branch was changed.

| Observation | gpt-transcribe | Faster-Whisper medium |
| --- | --- | --- |
| Integration elapsed time | 12.578 s, including MP3 extraction and HTTP | 28.301 s, including model construction and decoding |
| Raw whitespace word count, same method | 624 | 643 |
| Word alignment | Not returned | 650 normalized words, 45 raw segments |
| Negation and closing CTA | Present | Present |
| Complete acne/resorcina sentence | One complete version, with incomplete retries retained | Three identical complete copies at 171.42–181.34, 181.34–191.30, 191.30–191.74 |
| Timing defect | No word timestamps to evaluate | Third copy: 15 words in 0.44 s, 10 zero-duration words; 17 zero-duration words overall |
| Phrase after 2023 | `hay que vomitar` | `hay que voltar` |

Both outputs say `era sintomática`; agreement is not verification of the
critical prefix against audio. Both return `resorcina` in the complete acne
sentence. GPT's opening reads `No secreto para nadie` versus Medium's
`No es secreto para nadie`. These disputed lexical details were not resolved
by direct listening: this session could retrieve and extract the original,
but native audio input was unavailable. The report embeds original audio
fragments at 140–151, 165–193 and 245–257 s for review; it does not claim an
independently listened reference or a word-error rate.

The original downloaded for review has SHA-256
`b37059b1790cf3eb0447bb54595cc99f6668aadc5f65dafa9cdf6181621ef9f5`,
38,700,219 bytes. The benchmark itself did not emit a source-media hash, so this
is review-copy provenance, not an assertion of a runtime hash measurement.
The official result artifact matches the JSON emitted in the job log; ZIP
SHA-256 `f13127810535a7b106fc0aea8676b644fe3e537c275042248972156144c66e1c`.
GPU: NVIDIA L4; faster-whisper 1.0.0; CTranslate2 4.8.2; temperature [0].

**Correction to the earlier GPT-4o paragraph:** rereading the actual result
from run 36020263828/job 107703070812 shows Medium had **650 normalized
words and 45 segments**, not the 623/54 copied from the separate earlier
Medium-versus-large-v3 run. Its complete raw timed transcript is byte-for-byte
equal as JSON to this new run's raw transcript. Both have content hash
`asrcontent_faca44334019ac1cc51dee1b` and config fingerprint
`asrcfg_664a49ba49f6baf6`. Thus the repetition was already present in that
GPT-4o comparison; this new pair does not demonstrate a new Medium variance.

Conclusion: GPT-Transcribe is now accessible and produces a structurally
better candidate on this test's repeated-sentence failure. Critical word
accuracy, GPT repeatability and cross-video performance remain unqualified.
Its text remains `selection_authority=False`; no word-alignment or production
promotion was performed. One successful request is not editorial acceptance.


## Deepgram credential recheck — 2026-09-24 18:32 UTC

After the user added Deepgram to the requested comparison, run
https://github.com/AutomatedRetailServices/EditDNA-worker/actions/runs/36041858862
(SHA `709760bbc9d18369a4013fe667f908a89154cc0f`, job `107775698719`)
checked the live `EditDNA-Worker-2` template. It reported
`DEEPGRAM_API_KEY present: False`. The new generic provider-credential gate
stopped before any Modal GPU or Deepgram request; **no Deepgram transcript
was produced and no paid ASR comparison was consumed**. Creating the account
does not itself install the key in the worker environment.

The branch-only launch trigger was removed at
`6b97dfc22389128f7acd68943f4c0c38d98abc95`. The workflow remains manual-only;
the credential gate is retained for future explicitly requested providers.
Offline checks exercised missing, blank, present and no-provider cases and
confirmed the key value is never printed. A separate self-review in the same
session checked the single-job guard, no retries and pre-GPU ordering.
The summary now runs only if Modal actually supplied an exit code, avoiding
misreporting a credential preflight stop as an attempted GPU failure.

Outstanding input: connect an authorized Deepgram API key as
`DEEPGRAM_API_KEY` in the existing worker template. Once available, the
requested Video00 comparison can use the existing `nova-3`/`language=multi`
adapter. Production still uses Medium; no winner is declared for Deepgram.


## Authorized full-engine GPT + WhisperX experiment (2026-09-24)

The user authorized one Video00 run through the complete existing engine using GPT text and WhisperX word alignment. Implementation, isolation, evidence and execution bounds are recorded in [CUTSELL_GPT_WHISPERX_FULL_ENGINE_EVALUATION.md](CUTSELL_GPT_WHISPERX_FULL_ENGINE_EVALUATION.md). This remains on `feat/gpt-whisperx-video00`; canonical/default ASR is unchanged.
