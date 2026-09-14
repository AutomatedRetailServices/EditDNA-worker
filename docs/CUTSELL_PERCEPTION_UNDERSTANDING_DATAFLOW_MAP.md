# CutSell Perception + Understanding Active Dataflow Map

**Status: INVESTIGATION ONLY. No engine behavior was changed to produce
this document. No RAW was launched, no paid provider was called.**
**Date:** 2026-09-06. **Head at time of writing:** `e00f854`
(`feature/runpod-pod-on-demand`).

Scope: this traces the CURRENT ACTIVE engine (real code, real call sites)
against `docs/CUTSELL_CANONICAL_ENGINE_ARCHITECTURE_D098.md`'s Perception/
Understanding layers (L1-L4) and their consumers (L5-L9). It is not a
second architecture audit — `docs/CUTSELL_SYSTEM_AUDIT_D096.md` remains
that. Every finding below is a direct read of the code on this head, not
an inference from documentation.

---

## 1. Active perception dataflow summary

One real production entrypoint carries every perception signal:
`flow_b.py::process_local_sources`. In call order:

1. `media_probe.probe_media` — ffprobe duration/fps/has_audio per source.
2. `asr_provider.transcribe` — ASR transcript segments (word timestamps).
3. `canonical_asr_evidence.build_canonical_asr_evidence` — deterministic
   re-segmentation + content/evidence hashes (diagnostic only today).
4. `local_performance.analyze_local_performance` — MediaPipe/OpenCV dense
   per-frame observations (face/pose/motion) AND `*_candidate` events
   (abrupt visual changes: body/hand reset, facial-expression shift,
   camera disengagement). Real dependency: `mediapipe==0.10.21` is
   installed in `Dockerfile.cutsell.worker` via
   `requirements.cutsell.worker.txt` — this is a live analyzer on the
   real worker image, not a stub.
5. `whole_video_analysis.safe_whole_video_analyze` — builds
   `WholeVideoContext`. The real production provider,
   `whole_video_local.RunPodLocalWholeVideoProvider`, sets `events=()`
   unconditionally (it only carries a compact per-source transcript
   summary) — the dense visual candidates from step 4 are NOT produced
   here despite this module's own docstring describing it as the shell
   "needed by dense local MediaPipe/OpenCV evidence."
6. `local_performance.merge_local_events_into_context` — merges step 4's
   `*_candidate` events into `whole_context.sources[].events` (this is
   where the docstring's promise actually gets fulfilled — one call
   later, from a different module).
7. `audio_silence.audio_silence_events` + `merge_audio_silence_into_context`
   — ffmpeg `silencedetect` dead-air intervals, merged into the same
   `whole_context.sources[].events` stream as `AUDIO_SILENCE`-kind events.
8. `silence_analysis.word_silence_gaps` — ASR-timing-derived pause gaps
   (separate from step 7's audio-measured silence), feeds `take_segmentation`.
9. `take_segmentation.segment_takes` — builds `CandidateTake`s from
   transcript + gaps.
10. `performance_confirmation.confirm_local_performance_events` — the
    ONLY promotion step: a `*_candidate` visual event is promoted to a
    confirmed `wrong_take` or `retry_setup` `TemporalEvent` only when a
    nearby later take is lexically similar (likely the same communication
    attempt) AND local reset/break visual evidence is present. Two
    visual-evidence families together confirm `wrong_take`; one alone
    is `retry_setup`. These confirmed events are added to
    `whole_context.sources[].events`.
11. `visual_analysis.safe_visual_analyze` — an LLM-based visual provider
    Protocol. **Never wired in production**: `brain_runtime.py`'s real
    `BrainRuntime` factory hardcodes `visual_provider=None`. So this
    step is always a no-op on the real path (`trace.complete("visual",
    status="not_requested", ...)`).
12. `local_performance.apply_local_performance_to_takes` — writes real
    MediaPipe-derived values (face_visibility, eye_contact,
    motion_stability, visual_fumble, expression_naturalness,
    gesture_naturalness, distraction_risk) onto each take's
    `MediaSignals` (`CandidateTake.signals`). Because step 11 never
    runs, `audio_quality`, `framing_quality`, `product_visibility`,
    `continuity`, `delivery_energy` NEVER leave their `MediaSignals`
    defaults (0.5/0.5/0.0/0.5/0.5) on any real run today.
13. `attempt_reconstruction.reconstruct_delivery_attempts` — reads
    `whole_context.sources[].events` (via its own private
    `_source_events` helper) for confirmed `wrong_take`/`retry_setup`/
    measured-pause evidence, and fuses/splits attempts accordingly
    (D-097.5's measured dead-air boundary; D-097.7's wrong_take
    pre-group credit both live here).
14. `pipeline.build_flow_b_draft` — orchestrates grouping → equivalence
    reconciliation → take-judge ranking → deterministic Best Take →
    claim-coverage → story validation → Freeze → (later) Boundary →
    Renderer → technical QC → perceptual Watch+Listen.

---

## 2. Shared attempt representation: **PARTIAL**

There is no single canonical attempt-evidence record every authority
reads. Three separate, independently-coded access patterns exist for the
same underlying evidence:

- `attempt_reconstruction.py::_source_events` / `_events_for_source`
- `boundary_engine_pass.py::_events_for_source`
- `perceptual_watch_listen.py::_events_for_source`

All three do the same thing — find `whole_context.sources[].events` for
one `source_asset_id` — implemented three times, not shared. This is
exactly the kind of low-risk consolidation D-098's Layer 3 (Canonical
Multimodal Attempt Evidence) exists to eventually replace, but a full
canonical record is not needed to fix it (see Section 6).

Separately, `CandidateTake.signals` (`MediaSignals`) is a real per-clip
multimodal quality record, carried unchanged through `DraftClip.signals`
all the way to render/QC layers, but it is populated by TWO different,
uncoordinated writers (`local_performance.apply_local_performance_to_takes`
and the never-invoked `visual_analysis.apply_visual_observations`) and
consumed by exactly ONE decision function (`take_judge.score_take`).

**Do downstream authorities read the same truth, or reconstruct their
own?**

| Module | Reads raw perception? | What it actually sees |
|---|---|---|
| `AttemptReconstructor` (`attempt_reconstruction.py`) | YES | `whole_context.sources[].events` (confirmed visual + measured audio silence) directly |
| `RecordingProcessRemoval` (dedup/cleanup, D-097 Priority E) | Partial | Operates on already-reconstructed attempts, not raw events |
| Retry-family / `IdeaClusterer` (`take_grouping.py`, `take_grouping_provider.py`) | **NO** | Text/lexical evidence only (`_restart_content`, opening-token overlap). `whole_video_context`/`TemporalEvent`/`.signals` never appear in either file. `reconcile_semantic_idea_equivalence` (where D-097.A/D-097.12 restart-evidence rules live) is called from `pipeline.py` WITHOUT `whole_video_context`, even though `whole_video_context` is a live local variable one call earlier in the same function (`safe_group_takes_by_sessions`) |
| `DeliveryScorer` / `BestTake` (`take_judge.py`) | YES | Both `whole_context` events (`delivery_cleanliness_evidence`: interior dead air + multimodal reset+break penalty) AND `MediaSignals` (`score_take`'s full weighted multimodal formula) |
| `RealizationResolver` (`realization_resolver.py`) | **NO** | Zero references to `whole_video`, `TemporalEvent`, or `.signals` — operates entirely on the semantic ledger (already-reduced scores/labels from upstream) |
| `DeterministicBestTakeAuthority`, `ClaimCoverageBestTake` | **NO** | Same as RealizationResolver — ledger/label consumers only |
| Story Validation (`final_story_coherence_validation.py`) | Indirect | Reads a propagated STRING TAG (`"whole_video_bad_take:wrong_take"`), not the raw event — one layer of abstraction removed from the original evidence |
| `BoundaryEngine` (`boundary_engine_pass.py`) | YES | Its own `_events_for_source` copy, used to avoid landing a physical cut inside a measured silence interval |
| Perceptual Watch+Listen (`perceptual_watch_listen.py`) | YES | Its own `_events_for_source` copy, post-render, for `reset_debris_at_edges` |

**Conclusion:** authorities do not reconstruct materially DIFFERENT
truths from the same raw media (there is one real evidence stream), but
they consume it at different removes — some read it directly (with
duplicate accessor code), some read a propagated label, and the entire
retry-family/grouping authority never receives it at all despite the
data being computed, merged, and sitting in scope one function call away.

---

## 3. Top existing multimodal signals already available

1. **Confirmed `wrong_take` / `retry_setup` visual evidence**
   (`performance_confirmation.py`) — corroborated (reset + disengagement/
   expression) or single-family visual evidence that a take was
   abandoned/interrupted, cross-checked against lexical similarity to a
   later take. Real, computed on every run with a video track, already
   consumed by `AttemptReconstructor`.
2. **Measured audio dead-air intervals** (`audio_silence.py`, ffmpeg
   `silencedetect`) — objective, not ASR-timing-derived. Consumed by
   `AttemptReconstructor` (measured-pause boundary, D-097.5),
   `take_judge` (interior dead-air penalty), `BoundaryEngine` (avoid
   cutting inside silence), and post-render QC/perceptual review.
3. **Per-take `MediaSignals`** (face visibility, eye contact, motion
   stability, visual fumble, expression/gesture naturalness, distraction
   risk) — real MediaPipe-derived values, consumed by `take_judge.
   score_take`'s full weighted ranking formula (the `watch_listen_
   baseline` branch, not just the `text_timing_baseline` text-only
   fallback, whenever MediaPipe successfully observed frames for that
   take).
4. **Dense visual `*_candidate` events** (body/hand reset, facial-
   expression shift, camera disengagement) — the raw material for #1,
   also directly interior-scanned by `take_judge.delivery_cleanliness_
   evidence`'s multimodal-reset penalty.

## 4. Top signals currently lost / diagnostic-only

1. **Retry-family/grouping evidence blindness** — the single biggest
   loss. `take_grouping.py`'s restart-evidence rules (same-opening-
   restart, safe-short-prefix-retry, and D-097.12's
   `incomplete_attempt_completed_by_retry`) are 100% lexical. Confirmed
   `wrong_take`/`retry_setup` visual evidence and measured audio silence
   both exist and are already in scope at the call site, but neither is
   passed into the grouping/equivalence-reconciliation functions. This
   is a plumbing gap (evidence exists, wrong authority never receives
   it), not a missing perception capability.
2. **MediaSignals fields declared but never populated on the real
   path**: `audio_quality`, `framing_quality`, `product_visibility`,
   `continuity`, `delivery_energy` all stay at their dataclass defaults
   because the only writer for them (`visual_analysis.
   apply_visual_observations`, an LLM-based provider) is never invoked —
   `brain_runtime.py` hardcodes `visual_provider=None`. `take_judge.
   score_take` includes real weight terms for all five
   (`0.12*audio_quality + 0.06*framing_quality + 0.05*product_visibility
   + 0.07*continuity + 0.07*delivery_energy`), so roughly 37% of that
   formula's weighted terms are silently running on a constant, not a
   measurement, on every real run today.
3. **Dense visual `*_candidate` events not corroborated into confirmed
   evidence unless a lexically-similar later take exists nearby**
   (`performance_confirmation.py`'s `max_retry_gap_sec=4.0` window). A
   real visual reset with no nearby textual retry (e.g. a long pause
   before a genuinely new idea) produces no confirmed event and is
   invisible to every downstream editorial consumer — only the raw
   `*_candidate` events remain, and nothing outside `local_performance.
   apply_local_performance_to_takes`'s aggregate `MediaSignals` update
   ever looks at them again.
4. **`canonical_asr_evidence`'s content/evidence hashes** — computed on
   every run, never read by any editorial decision; purely a stability-
   battery diagnostic (this one is explicitly documented as such in
   `flow_b.py`'s own comments, not a hidden gap).

---

## 5. D-097.12 stomach trace

The real Video00 persisted evidence for the stomach family
(`scratchpad/d097_{9,10,11}_raw/`, referenced in `docs/CUTSELL_DECISIONS.md`
D-097.9-11) is not present in this sandbox — this is a fresh container,
and re-fetching it would require the same S3 access already established
as unavailable here (D-097.13). Per this task's own instruction ("use the
existing D-097.12 stomach fixture ... as the concrete trace example when
possible" / "do NOT run Video00"), the trace below uses the real
CleanCutBench fixture
(`tests/test_cutsell_clean_cut_core_evaluation_suite.py::
test_incomplete_stomach_attempt_survives_arbiter_rejection_through_the_
full_chain`) plus the actual code path it exercises, since that fixture
is the offline-reproducible ground truth for this exact family.

**Evidence that exists in the fixture / would exist for the real
family:**

| Attempt | Text evidence | Visual/audio evidence available in principle | Consumed by |
|---|---|---|---|
| abandoned (`Tuve problemas estomacales ... me diagnosticaron con...`, incomplete) | ASR transcript, `complete_idea=False` | Would have confirmed `wrong_take`/`retry_setup` visual evidence IF the creator visibly reset before abandoning (real footage would carry this via `performance_confirmation.py`) and/or a measured audio-silence gap before the next attempt | `take_grouping.py` restart-evidence rules (text only); `AttemptReconstructor` (visual/audio evidence, if any) |
| aside (`... en 2023, no hay que preguntar.`, complete, independent) | ASR transcript | Same channels available | `take_grouping.py` (text only, correctly kept separate by the 2-token opening-window rule) |
| clean (`Tuve problemas de digestion ... nada severo ... tres meses con pastillas.`, complete) | ASR transcript | Same channels available | `take_judge.score_take` (would see real `MediaSignals` on the actual footage) |

**What made D-097.12's deterministic rule necessary?** The fixture's own
CleanCutBench test proves the arbiter (Gemini) declined to confirm the
abandoned/clean pair as the same idea in production (`oracle_pairs=
frozenset()` reproduces the worst observed case), and no EXISTING
restart-evidence rule (`same_opening_restart`, `_safe_short_prefix_retry`)
matched this specific shape (an incomplete attempt sharing only a 2-word
opener with its complete retry, differing after that). This is a genuine
LEXICAL coverage gap in the deterministic-evidence layer, closed by
adding a new rule at the SAME lexical layer.

**Was the relevant evidence already present but not available to the
right authority, or is a genuinely missing perception signal involved?**
**Both, at different levels:**
- At the layer that actually needed fixing (retry-family grouping), the
  gap is architectural: even if the real footage had confirmed visual
  `wrong_take` evidence for the abandoned attempt (plausible — an
  abandoned mid-attempt is exactly the pattern `performance_confirmation.
  py` is built to catch), `take_grouping.py` could not have used it
  regardless, because it never receives `whole_video_context` at all.
  D-097.12 could not have used multimodal evidence even if it existed,
  so a text-layer fix was the only option available in the current
  architecture — not a design mistake, a structural limit.
- Whether the real Video00 footage's `whole_video_context` actually
  contains a confirmed `wrong_take`/`retry_setup` event for this
  specific abandoned attempt is unverified in this sandbox (no persisted
  evidence file available); this is a factual gap, not an architectural
  one, and does not change the conclusion above.

---

## 6. Upstream perception vs downstream Watch+Listen map

For each `perceptual_watch_listen.py` v1 capability:

| Capability | Same evidence exists upstream? | Reused or reimplemented? | Post-render only? | Could upstream selection benefit from the same evidence? |
|---|---|---|---|---|
| `interior_dead_air_mp4` | YES — `audio_silence.py`'s ffmpeg `silencedetect` already produces the same class of evidence pre-render, consumed by `take_judge` and `AttemptReconstructor` | Reimplemented independently on the rendered MP4 (a fresh ffmpeg pass on the final file, not a reuse of the pre-render measurement) | YES (measures the ACTUAL render, correctly — pre-render dead air and post-render dead air are not guaranteed identical after Boundary trims) | Already does, via the pre-render `audio_silence` pathway; this capability's job is specifically to verify the RENDERED result, not to inform selection |
| `cut_adjacent_speech_energy_mp4` | Partial — `human_gold_decision_map`'s audio-feature primitives exist but were built for the Video00 ladder's correlation matching, not join-energy specifically | New measurement on rendered audio | YES (inherently — a join only exists after render) | No natural pre-render analogue (there is no "join" before Boundary executes it) |
| `reset_debris_at_edges_source_evidence` | YES — the exact same `whole_context.sources[].events` confirmed evidence `AttemptReconstructor` already reads pre-selection | Reused conceptually but reimplemented as a THIRD private `_events_for_source` (see Section 2) | YES, checks the RENDERED edge specifically | Already does, upstream, at `AttemptReconstructor` — this capability is deliberately a redundant physical-result check, not a new information source |
| `repeated_audience_content_transcript` | YES — transcript + `retry_similarity` (the same lexical-similarity primitive `performance_confirmation.py` already uses) | Reused function (`retry_similarity` imported from `performance_confirmation.py`), applied post-render | YES | Already informs upstream retry/duplicate handling via the same primitive; this is the intended redundant check, not a new capability |
| `facial_expression_post_line` (NOT_IMPLEMENTED) | Partial — `local_performance.py` produces `expression`/`facial_expression_shift_candidate` upstream, but nothing decodes POST-LINE expression on rendered frames | N/A | N/A | Upstream already has a coarser version (interior-take `expression_naturalness` in `MediaSignals`); a true post-line check needs new frame decoding on the render, not just reuse |
| `gesture_continuity_across_cut` (NOT_IMPLEMENTED) | Partial — `local_performance.py`'s hand/body landmarks exist upstream, but nothing tracks continuity ACROSS a physical join on the render | N/A | N/A | Same as above — upstream data exists in a different shape (per-take aggregate, not frame-level continuity across a specific join) |
| `clipped_phoneme_asr_realign` (NOT_IMPLEMENTED) | NO — no phoneme/word re-alignment on rendered audio exists anywhere in the codebase today | N/A | N/A | Would need new ASR re-alignment work, not a reuse of existing evidence |
| `framing_and_eye_contact` (NOT_IMPLEMENTED) | Partial — `MediaSignals.eye_contact`/`framing_quality` exist as fields, `eye_contact` gets a real value upstream, `framing_quality` never does (Section 4, item 2) | N/A | N/A | Upstream `eye_contact` already informs `score_take`; a rendered-result framing check would need new work since `framing_quality` itself is unpopulated everywhere |

No authority boundary was or is proposed to change here — this table is
descriptive only, consistent with D-098 Section 4's rule that upstream
perception informs editorial decisions while downstream Watch+Listen only
diagnoses and routes.

---

## 7. Top 3 Cut.ai-blocking gaps

### Gap 1 — Retry-family grouping cannot see multimodal retry/reset evidence

- **Root cause:** `take_grouping.py`/`take_grouping_provider.py`'s
  restart-evidence rules and `reconcile_semantic_idea_equivalence` never
  receive `whole_video_context`, even though it is a live variable in
  `pipeline.py` at the call site immediately before (used only for
  `context_text`/session partitioning).
- **Existing evidence available?** YES — confirmed `wrong_take`/
  `retry_setup` events and measured audio-silence intervals already
  exist in `whole_video_context.sources[].events` by the time grouping
  runs.
- **Current owner:** `take_grouping_provider.py` (the retry-family
  authority), which currently has no perception input at all.
- **Minimum architectural bridge:** thread `whole_video_context` (or a
  lightweight extraction of just the confirmed events, keyed by
  `source_asset_id`) as an OPTIONAL parameter into
  `reconcile_semantic_idea_equivalence` and the restart-evidence rule
  functions, used ONLY as one more piece of corroborating evidence
  alongside existing lexical rules (e.g., a confirmed `wrong_take` near
  an incomplete attempt's boundary raises confidence for a lexical
  restart-evidence match that is currently borderline) — never a
  standalone trigger, and never a change to what merges without
  multimodal corroboration today.
- **Expected Cut.ai effect:** fewer future bounded lexical-rule patches
  like D-097.12 (each new phrasing of "incomplete attempt + differently-
  worded complete retry" currently needs its own text rule); this is the
  gap most likely to remove a whole CLASS of future R-numbered patches,
  since it addresses the mechanism, not one phrasing.
- **Risk:** LOW if additive/corroborating only; the real risk is scope
  creep into "multimodal decides grouping," which D-098 and this
  document explicitly do not authorize.

### Gap 2 — Half of `MediaSignals`' ranking weight runs on constants, not measurements

- **Root cause:** `visual_provider` is hardcoded to `None` in
  `brain_runtime.py`, so `audio_quality`, `framing_quality`,
  `product_visibility`, `continuity`, `delivery_energy` never receive a
  real value; `take_judge.score_take` still weights all five as if they
  were measured.
- **Existing evidence available?** NO for these five fields
  specifically (no local/offline producer exists for them today — they
  were designed for an LLM visual provider that was never wired). The
  OTHER seven `MediaSignals` fields ARE real.
- **Current owner:** `take_judge.py` (DeliveryScorer/BestTake).
- **Minimum architectural bridge:** NOT "wire up an LLM visual
  provider" (that is new paid infrastructure, explicitly out of scope).
  The minimum bridge is local: `audio_quality` has a cheap local proxy
  already computed and unused for this purpose
  (`human_gold_decision_map`'s RMS/ZCR envelope primitives, or even
  `audio_silence.py`'s own noise-floor measurement); `continuity` and
  `product_visibility` could take a conservative local-only default that
  does not silently masquerade as a measurement (e.g. drop those two
  terms from the weighted sum until a real signal exists, rather than
  scoring every take identically on them). `framing_quality` and
  `delivery_energy` have no local proxy today and should stay
  explicitly excluded from the formula until one exists, rather than
  contributing a constant.
- **Expected Cut.ai effect:** more honest BestTake ranking on borderline
  good-vs-good contests (two clean retries of comparable text quality
  are currently ranked partly on frozen numbers); moderate effect size
  since `score_take`'s other seven real terms usually dominate obvious
  cases.
- **Risk:** LOW to remove unused constant terms; MEDIUM if a new local
  proxy is added, since it changes ranking outcomes and needs the same
  regression discipline as any DeliveryScorer change.

### Gap 3 — Confirmed visual retry/reset evidence is corroboration-gated to a narrow 4-second lexical-similarity window

- **Root cause:** `performance_confirmation.py`'s `max_retry_gap_sec=4.0`
  requires a nearby, lexically-similar LATER take before ANY visual
  `*_candidate` cluster is promoted to a confirmed event. A real reset
  with a longer natural pause before the retry, or a retry whose wording
  diverges enough to fall under `minimum_retry_similarity=0.58`, is never
  confirmed and becomes invisible everywhere downstream except the
  aggregate `MediaSignals` update.
- **Existing evidence available?** YES — the raw `*_candidate` events
  exist; only the promotion step is narrow.
- **Current owner:** `performance_confirmation.py` (the bridge between
  MediaPipe and editorial evidence).
- **Minimum architectural bridge:** none recommended without real
  Video00 evidence first. This document explicitly does NOT recommend
  widening the window or lowering the threshold — that is exactly the
  kind of "loosen a threshold because it seems restrictive" change this
  investigation is not authorized to propose without measuring how often
  it actually misses a real case. Flagged here as a candidate area for
  the NEXT investigation (measure false-negative rate against real
  RAWs), not a current recommendation.
- **Expected Cut.ai effect:** unknown until measured — this is why it
  ranks third, not first.
- **Risk:** widening it carelessly directly risks false-positive content
  deletion (exactly the harm D-020/editorial rules exist to prevent), so
  this gap should stay UNPROVEN rather than "obviously worth closing."

---

## 8. Minimum next implementation recommendation (NOT authorized by this document)

If the Product Owner authorizes ONE next bounded engineering task from
this investigation, Gap 1 is the strongest candidate: thread
`whole_video_context`'s already-computed confirmed events into
`reconcile_semantic_idea_equivalence` as an optional, purely
corroborating input to the EXISTING restart-evidence rules — not a new
Multimodal Attempt Record, not a new authority, not a rewrite of
`take_grouping.py`'s lexical rules. This is explicitly NOT authorized to
begin from this document; it requires separate Product Owner sign-off
per the anti-loop contract.

---

## 9. Confirmations

- **NO ENGINE BEHAVIOR CHANGED.** This document is the only file
  produced by this investigation; no `cutsell_worker/*.py` file was
  modified.
- **NO RAW / PROVIDER / S3 / INFRA WORK.** No Video00 RAW was launched,
  no paid provider was called, no AWS/S3 access was attempted, no
  infrastructure was touched.
