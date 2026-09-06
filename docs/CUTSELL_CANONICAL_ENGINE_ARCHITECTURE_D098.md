# CutSell Canonical Engine Architecture — Evolutionary Target Map (D-098)

**Status: CANON / DOCUMENTATION ONLY. No engine behavior was changed to
produce this document.**
**Date:** 2026-09-06. **Head at time of writing:** `69f244a`
(`feature/runpod-pod-on-demand`).

This document does not replace `docs/CUTSELL_SYSTEM_AUDIT_D096.md` or any
`D-097.x` decision. It is the evolutionary continuation of D-096's PROPOSED
TARGET ARCHITECTURE (Part 11), read together with everything D-097.x
proved on top of it. Nothing here authorizes new engineering. It exists so
the project can be read as ONE continuous architecture instead of a chain
of disconnected `Rxx` fixes.

**Precedence.** Where anything below appears to conflict with an accepted
D-096/D-097.x authority contract, the accepted contract wins and the
conflict is recorded under "Conflicts requiring Product Owner decision"
(end of this document) — nothing here silently overrides it.

---

## 0. One sentence

> OLD CANON (D-096) + ACCEPTED IMPLEMENTATION (D-097.x) + a PERCEPTION /
> UNDERSTANDING evolutionary layer = ONE continuous CutSell architecture —
> not a second system standing beside the first.

---

## 1. Core product doctrine (unchanged, restated)

CutSell remains **one canonical editing pipeline**. The engine is meant to
progressively become capable of:

1. seeing and hearing the RAW;
2. understanding creator recording behavior;
3. understanding relationships between attempts;
4. distinguishing audience-facing delivery from recording process;
5. selecting the best / minimum sufficient clean realization;
6. making physically natural cuts;
7. seeing and hearing its own rendered result;
8. learning from human-reviewed examples, regressions and evals.

The architecture below is ambitious on purpose. **Implementation stays
incremental and bounded** — see Section 6 (Anti-loop / execution contract).

---

## 2. Evolved canonical target architecture (20 layers)

Each layer is described at target scope, then immediately mapped to its
current real status in Section 3's compatibility table. A layer appearing
here is **not** an implementation order and **not** an authorization.

### Perception / Understanding (upstream of editorial decisions)

- **Layer 1 — Media Perception.** RAW audio/video produces evidence only
  (ASR, word timestamps, VAD, silence/pause duration, speech onset/offset,
  waveform/energy, cadence, breath, restart evidence, clipped-speech
  evidence; decoded frames, face/gaze/pose, gesture, framing, camera
  disengagement, visual reset evidence; ffmpeg/ffprobe media integrity).
  Perception produces evidence; it does **not** independently own
  editorial membership.
- **Layer 2 — Performance Understanding.** Fuses perception evidence into
  creator-behavior states (`AUDIENCE_DELIVERY`, `PRE_TAKE_SETUP`,
  `FALSE_START`, `ABANDONED_ATTEMPT`, `CLEAN_ATTEMPT`, `CORRECTION`,
  `CONTINUATION`, `POST_TAKE_RESET`, `RETAKE_EVENT`,
  `RECORDING_PROCESS`), each eventually carrying evidence, confidence, and
  a source time range.
- **Layer 3 — Canonical Multimodal Attempt Evidence.** One target shared
  attempt representation (identity, semantic, audio, vision, behavior,
  relationships, quality fields) so BestTake, Resolver, Boundary and QA
  stop each reconstructing their own private picture of the same attempt.
  Fields are additive/optional, not a required schema today.
- **Layer 4 — Attempt / Temporal Relationship Understanding.**
  `retry_of` / `continuation_of` / `corrects` / `supersedes` /
  `complements` / `duplicate_of` / `setup_for` / `same_editorial_function`
  relationships, target state: increasingly multimodal, not
  transcript-similarity alone. Evolves existing retry-family/grouping
  logic; does not replace it wholesale.

### Clean Raw / Cut.ai milestone (current active product area)

- **Layer 5 — Clean Raw Intelligence.** Separate audience-facing content
  from recording process; resolve false starts, fumbles, failed/abandoned
  attempts, obvious retries/duplicates, reset debris, accidental dead air;
  protect meaning, polarity/negation, numbers, diagnosis, corrections,
  valid complementary pieces, natural pauses/breaths. First goal: RAW →
  clean commercial video, not Human Gold.
- **Layer 6 — Realization / Best Take Intelligence.** No forced winner
  when no usable realization exists; one complete sufficient clean
  realization wins when sufficient; composite only when necessary;
  unusable failed retries are never restored merely to maximize claim
  coverage; delivery quality progressively becomes multimodal evidence;
  critical meaning is a safety constraint, not the editorial objective.
- **Layer 7 — Boundary Intelligence.** Membership answers WHAT survives;
  Boundary answers WHERE the physical cut occurs, using word timing,
  phoneme/audio alignment, VAD, breath, speech onset/offset, and
  (target) pose/gesture/performance state for a safe visual resting
  point. FFmpeg executes ranges; Boundary owns them. Preserves the
  existing single-Boundary-ownership doctrine.
- **Layer 8 — Downstream Perceptual Watch+Listen QA.** Sees/hears the
  ACTUAL rendered result. Role: DIAGNOSE, CLASSIFY, ROUTE — never edit.
  Verdict vocabulary PASS/FAIL/UNCERTAIN/ERROR. A full PASS requires the
  capabilities a gate actually needs to have been evaluated;
  NOT_IMPLEMENTED/UNCERTAIN/ERROR must never silently become a full PASS.
  A narrower `SUPPORTED_CHECKS_PASS`-style status may exist but must
  never be represented as full perceptual PASS.
- **Layer 9 — Cut.ai Commercial Parity (MILESTONE 1).** CutSell reliably
  converts RAW into a commercially clean edit matching or exceeding
  Cut.ai for recording-process removal, obvious retry resolution, clean
  take selection, basic redundancy removal, meaning preservation,
  commercial boundary quality, basic continuity. Cut.ai stays QA-only,
  never production input/prompt/timing/oracle.

### Human Gold milestone (downstream of Cut.ai parity)

- **Layer 10 — Editorial Function.** What job a realization performs
  (HOOK/SETUP/PROBLEM/SYMPTOM/DIAGNOSIS/EXPLANATION/PROOF/REFLECTION/
  CONCLUSION/CTA/etc.); two different statements may perform the same
  function.
- **Layer 11 — Delivery Sufficiency.** Does this realization ALONE
  sufficiently perform its editorial function — not just "does it contain
  every possible semantic claim".
- **Layer 12 — Information Gain / Editorial Utility.** Classify additional
  material as REQUIRED_NEW_INFORMATION / USEFUL_SUPPORT / REFRASE /
  REDUNDANT / LOW_VALUE_ELABORATION.
- **Layer 13 — Minimum Sufficient Editorial Set.** The Human Gold
  objective is the minimum material that correctly and effectively
  delivers the story, not maximum semantic coverage.
- **Layer 14 — Good-vs-Good Ranking.** Two complete realizations doing the
  same editorial job compete; pick one unless the second contributes
  genuinely required new information.
- **Layer 15 — Composite Intelligence.** Composite only when no single
  clean realization suffices, pieces are genuinely complementary and
  compatible, together they form the minimum sufficient realization, and
  no member is redundant.
- **Layer 16 — Multimodal Performance Ranking.** Future ranking evidence
  (fluency, eye contact, engagement, confidence, cadence, facial
  delivery, gesture quality, brevity, narrative fit) to choose between two
  GOOD takes. Human Gold stays QA-only, never production input.

### Engine growth / learning (roadmap)

- **Layer 17 — Decision Ledger.** Every important KEEP/DROP/COMPOSITE
  decision eventually explainable: what happened, why, evidence,
  confidence, supersession relationship, owning authority.
- **Layer 18 — Confidence + Abstention.** HIGH confidence → automatic;
  MEDIUM → bounded semantic/multimodal arbiter; LOW → conservative /
  unresolved / review. CutSell must not manufacture certainty.
- **Layer 19 — Human Feedback → Evaluation Memory.** Human-reviewed cases
  (stomach retry, pimples, Symptoms complementary content, diagnosis
  preservation, polarity/"No", duplicate conclusion, boundary defects)
  become persistent evaluation/regression cases, never hardcoded into
  production behavior.
- **Layer 20 — Future Learned Components.** Roadmap only, not current
  implementation authorization: retry classifier, recording-process
  classifier, performance-state classifier, take-quality ranker,
  editorial-redundancy classifier, composite-necessity classifier —
  gated on sufficient evaluation evidence existing first.

---

## 3. Old canon → evolved canon compatibility map

Classification key: **EXISTING** / **EXISTING + NEEDS CONSOLIDATION** /
**PARTIALLY IMPLEMENTED** / **MISSING / FUTURE**. `CURRENT ACTION` is one
of **PRESERVE** / **CONSOLIDATE LATER** / **FUTURE** — never "replace".

| Existing component (D-096/D-097.x) | Target layer(s) | Status | Proven | Missing | Current action |
|---|---|---|---|---|---|
| ASR + word timestamps, `human_gold_decision_map` audio features, ffmpeg/ffprobe media probes | L1 Media Perception (audio + media) | EXISTING | Feeds AttemptReconstructor, take_judge, live_render_qc, perceptual_watch_listen today | VAD as a first-class signal, systematic breath/cadence extraction | PRESERVE |
| Local multimodal "performance evidence" events (`whole_video_context.sources[].events`, reset/break kinds consumed by `perceptual_watch_listen._reset_debris_at_edges`) | L1 Media Perception (video) + L2 Performance Understanding | PARTIALLY IMPLEMENTED | Real production consumer exists (`_reset_debris_at_edges`); D-097.13 proved the checkpoint can read it | No canonical producer schema for face/gaze/pose signals; coverage is event-kind-specific, not a full performance-state fusion | CONSOLIDATE LATER |
| `AttemptReconstructor` (measured-pause boundary, D-097.5/.6 fixes) | L1→L2 boundary, L3 attempt identity | EXISTING | D-097.5/.6 RAWs proved measured dead-air pause boundary on real MP4s | Fusing visual/behavioral evidence into the same reconstruction decision | PRESERVE |
| RecordingProcessRemoval behavior (D-097 Priority E physical cleanup ownership, dedup) | L5 Clean Raw Intelligence | EXISTING | D-097 Priority E; RAW-proven physical dedup | Broader visual-reset-driven process detection | PRESERVE |
| Retry-family / `IdeaClusterer` grouping (`take_grouping.py`, `take_grouping_provider.py`, D-097.A restart-evidence rules incl. D-097.12's `incomplete_attempt_completed_by_retry`; D-100's `multimodal_corroborated_retry`) | L4 Attempt/Temporal Relationship Understanding | EXISTING | D-097.12 offline-proven stomach-family fix; D-097.A family completeness RAW-proven; D-100 closed D-099 Gap #1 offline (confirmed `wrong_take`/`retry_setup` evidence now optionally corroborates a weaker lexical link, gated by shared content + completeness asymmetry + boundary proximity) | Explicit `retry_of`/`corrects`/`complements` typed relationships beyond restart-evidence kinds; D-100's bridge is unproven on real Video00 media | CONSOLIDATE LATER |
| `DeliveryScorer` (cleanliness evidence: interior dead air, multimodal resets, negative controls) | L6 Realization/Best Take Intelligence | EXISTING | D-097 Priority C/D scoring proven in CleanCutBench | Full multimodal performance ranking (L16) as scoring input | PRESERVE |
| `BestTakeResolver` / `deterministic_best_take_authority.py` (`swap_enabled` deactivated per D-019) | L6 Realization/Best Take Intelligence | EXISTING | RAW-proven across D-097.1–.12 | — (SWAP explicitly out of scope, not a gap) | PRESERVE |
| `RealizationResolver` (`realization_resolver.py`, usable-first tiers, no-composite-with-failed-members, critical-veto-not-composite-forcing) | L6 Realization/Best Take Intelligence | EXISTING | D-097 Resolver Level-1 fixes, RAW-proven | — | PRESERVE |
| Story / Final Coherence Validation (`final_story_coherence_validation`, contradiction invariant, idea-coverage tracking, D-097.9 story-basis fix) | L6→L9 boundary (meaning protection) | EXISTING | D-097.9 R11 RAW-proven | Editorial-function/sufficiency reasoning (L10-11) is not yet part of this validator | PRESERVE |
| Selection Freeze | L5/L6 boundary marker | EXISTING | RAW-proven across the whole D-097.x thread | — | PRESERVE |
| `BoundaryEngine` (`boundary_engine_pass.py`, ONE post-Freeze pass, physical ownership contract, D-097.4 join-instant detector) | L7 Boundary Intelligence | EXISTING | D-097.4 RAW-proven frame-exact joins | Non-audio boundary evidence (visual reset points, gesture-safe cut points) | FUTURE |
| `Renderer` (`render_plan.py`, `live_render_qc.py`, gapless single-pass concat) | L7 Boundary Intelligence (execution) | EXISTING | D-097.2/.4 RAW-proven | — | PRESERVE |
| Technical post-render QC (`post_render_media_qc.py`, `live_render_qc.render_with_post_render_qc`) | L8 Downstream Watch+Listen (technical tier) | EXISTING | RAW-proven, D-097.4 probe fix | — | PRESERVE |
| `perceptual_watch_listen.py` (v1, advisory, per-capability, never auto-PASS) | L8 Downstream Watch+Listen (perceptual tier) | PARTIALLY IMPLEMENTED | 4 capabilities EVALUATED (interior dead air, cut-adjacent speech energy, reset debris via source evidence, repeated audience content); routes to BoundaryEngine/BestTakeResolver/Renderer | `facial_expression_post_line`, `gesture_continuity_across_cut`, `clipped_phoneme_asr_realign`, `framing_and_eye_contact` are all `NOT_IMPLEMENTED` today | CONSOLIDATE LATER |
| Clean Raw diagnostic checkpoint (`benchmarks/clean_raw_checkpoint.py`, D-097.13) | L8 Downstream Watch+Listen (checkpoint harness) | EXISTING | D-097.13: mechanism proven end-to-end on synthetic proxy media through the real render/QC/perceptual/membership-correlation authorities | Real Video00 stomach-family MP4 proof (this sandbox's AWS credentials are invalid for the persisted S3 bucket) | PRESERVE (proof pending, not the mechanism) |
| D-097.12 stomach-family grouping rule | L4 Attempt/Temporal Relationship Understanding | EXISTING | Offline/CleanCutBench-proven (fixture 55/55, full regression) | Real-media (RAW MP4) proof of the selection's rendered effect | PRESERVE |
| `benchmarks/video00_quality_ladder.py` (four-way region map, LEVEL 1/2/3 classification) | Cut.ai/Human Gold parity measurement (L9/L16 evidence) | EXISTING | RAW-proven, QA-only, architecturally enforced never-imported-by-production | — | PRESERVE |
| CleanCutBench (`test_cutsell_clean_cut_core_evaluation_suite.py`) | L19 Human Feedback → Evaluation Memory (present-day instance) | EXISTING | 55 fixtures, gates every RAW authorization | Systematic conversion of every future human-reviewed case into a fixture (partly already the practice, not yet a named process) | CONSOLIDATE LATER |
| Editorial function / sufficiency / minimum-set / good-vs-good / composite-necessity reasoning (L10-15) | Human Gold milestone | MISSING / FUTURE | Not implemented | Everything | FUTURE |
| Decision Ledger, confidence/abstention framework, learned/calibrated classifiers (L17-20) | Engine growth | MISSING / FUTURE | Not implemented | Everything | FUTURE |

Nothing in this table is marked obsolete, deleted, or renamed. Every row
with an `EXISTING` status names the real module/file already carrying it.

---

## 4. The two Watch+Listen roles (must never be conflated)

**A. Upstream perceptual understanding.** RAW → see + hear → perception
evidence → performance understanding → attempt understanding → informs
editorial decisions. This helps the motor edit. Today's real instance:
the local multimodal "performance evidence" events already consumed by
`AttemptReconstructor`/pre-group credit logic and by
`perceptual_watch_listen._reset_debris_at_edges` (as *evidence*, not as a
second editor).

**B. Downstream perceptual QA.** Rendered result → see + hear result →
detect defect → route to the owning authority. This supervises the
result. Today's real instances: `perceptual_watch_listen.py` (production
render path, advisory v1) and `benchmarks/clean_raw_checkpoint.py`
(D-097.13's bounded diagnostic-render harness, QA-only, reuses the same
authorities).

They may reuse the same underlying audio/frame/face/pose/semantic
analysis primitives. **Their authorities are different.** Upstream
perception may inform an editorial decision (e.g., the wrong_take
pre-group credit, D-097.7). Downstream Watch+Listen must never silently
rewrite selection — it only routes findings to Selection/BestTake for
wrong content, Boundary for bad physical cuts, or Renderer for A/V
defects, exactly as D-095 already specifies.

---

## 5. Delivery roadmap

**Milestone 1 — RAW → Cut.ai Commercial Parity (current).**
Progression: (A) perception evidence exists → (B) attempts reconstructed
correctly → (C) recording behavior understood → (D) retry relationships
stable → (E) clean realization selected → (F) meaning protected → (G)
physical boundaries clean → (H) rendered result inspected → (I) benchmark
Cut.ai parity demonstrated → (J) Cut.ai-level behavior demonstrated on
unseen RAWs.

Current position: D-097.12 proved (E)/(D) offline for the stomach family;
D-097.13 proved the (H) harness on synthetic media only; a real-media
proof of (H) for the stomach family, and (I)/(J) at large, remain
outstanding. The full Perception → Understanding dataflow (L1-L4) has not
yet been mapped end-to-end against this evolved architecture — that
mapping is the next investigation named in Section 7, not yet authorized.

**Milestone 2 — Cut.ai → Human Gold (downstream, not started).**
Progression: (A) editorial function → (B) delivery sufficiency → (C)
information gain → (D) minimum sufficient set → (E) good-vs-good ranking
→ (F) composite necessity → (G) multimodal performance ranking → (H)
rhythm/narrative fit → (I) Human Gold benchmark parity → (J) unseen-RAW
generalization. Nothing in Milestone 2 is authorized before Milestone 1
closes.

---

## 6. Anti-loop / execution contract (binding, restated)

- Architecture is not authorization. A capability appearing anywhere in
  this document does not authorize implementing it.
- Every real engineering authorization needs: one bounded capability, one
  owning authority or explicit dataflow, explicit positive tests,
  explicit negative controls, an integrated behavior test, observable
  evidence, regression protection, a compute/cost boundary, and a STOP
  condition.
- A newly discovered root cause does not automatically renew
  authorization. A new D-xxx number is not progress by itself. Progress
  is measured by CLOSED product capabilities (selection correct → no
  later authority reverses it → real diagnostic render → technical QC →
  perceptual/human verification as applicable → regression protected).
  Synthetic-only proof does not close a real-media capability — see
  D-097.13's status above.
- D-091 autonomous continuation must never override an explicit bounded
  task's STOP condition. This document's own authorization ends at
  Section 7; the next investigation named there is not begun by writing
  this document.

---

## 7. Conflicts requiring Product Owner decision

None identified while producing this document. No proposed target-layer
concept in Section 2 was found to contradict an accepted D-096/D-097.x
authority contract; every mapped row in Section 3 is additive
(CONSOLIDATE LATER / FUTURE) rather than substitutive. If a future
mapping pass finds a real conflict, it is recorded here rather than
silently resolved.

---

## 8. Current exact next engineering investigation (not authorized here)

**Perception + Understanding active dataflow map.** What perceptual/
audio/visual/semantic evidence already exists today, where is it stored
(e.g. `whole_video_context.sources[].events`, ASR word timestamps,
measured-pause data), and which active downstream authorities actually
consume it versus which target-layer consumers (L2-L4) do not exist yet.
This is a read-only investigation candidate, not an implementation, and
is **not authorized by this document** — it requires separate Product
Owner authorization before any code is touched.
