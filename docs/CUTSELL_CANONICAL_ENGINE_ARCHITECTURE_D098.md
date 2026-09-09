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

*Answered by `docs/CUTSELL_PERCEPTION_UNDERSTANDING_DATAFLOW_MAP.md`,
carried out under that separate authorization; its Section 7 gaps and
Section 8 recommendation feed D-107 below.*

---

## 9. Non-destructive editing + Multimodal BestTake doctrine (D-107)

Minimal formalization only — no layer is renumbered, no status in
Section 3's table changes, no new authority is created. This section
gives names to concepts the Product Owner's D-107 directive introduced so
future work has a stable vocabulary; it does not itself authorize
implementing any of them.

**Non-destructive editing doctrine (target, not yet built — no Figma/UI
work is authorized by naming it here).** Source RAW media is immutable.
The engine's output is an *edit decision* over source-media ranges, never
a mutation of the source. Every candidate realization carries a
**source-range handle** (source asset id + start/end) that survives past
Selection Freeze. Membership is **PRESELECTED** (currently part of the
winning edit) or **NOT_PRESELECTED** (excluded from this edit, but never
deleted, altered, or made unrecoverable) — replacing any framing where
"discarded" implies destroyed. A future editor surface built on this
doctrine (extending a clip, switching takes, restoring a NOT_PRESELECTED
take) is Layer 20 territory (Future Learned Components / roadmap), not
authorized here.

**Multimodal BestTake doctrine (target ranking philosophy for L6/L16).**
Among realizations that legitimately compete for the same semantic/
editorial job AND sufficiently preserve the intended message, BestTake's
job is to choose the realization with the best overall audience-facing
delivery — not maximum claims, not longest transcript, not highest
energy, not an automatic Hybrid-label winner, and not automatically the
visually-cleanest take if it loses required meaning. Canonical priority
order, highest first: (1) MEANING/MESSAGE SUFFICIENCY, (2) TAKE
USABILITY, (3) MULTIMODAL PERFORMANCE QUALITY, (4) EDITABILITY/BOUNDARY
QUALITY, (5) NARRATIVE/ENERGY FIT. This orders existing authorities
(Selection/meaning-safety above BestTake's own delivery ranking above
Boundary's physical execution) — it does not add a new one.

Two new named concepts inside tier (3), both currently **undefined in
code** (`MediaSignals` has no field for either — see D-107's signal
audit):
- **DELIVERY_ENERGY_FIT** — energy should be natural/appropriate to the
  content, not maximized. This reframes the *existing but always-default*
  `MediaSignals.delivery_energy` field's intended semantics (see D-098
  Section 3 Gap 2 / D-107): it is a fit measure, not a "more energy is
  better" score, whenever a real producer is ever wired for it.
- **PERFORMANCE_CONTINUITY** — stable body/face across a delivery, no
  abrupt error-reset behavior ("I messed up" visible recovery), a natural
  transition into the take's end. Distinct from the existing whole-clip
  aggregate `visual_fumble`/`motion_stability` fields: continuity is
  about *where in the clip* instability falls and whether it interrupts
  the delivery itself, not merely whether instability exists anywhere in
  the clip.

**ENTRY_QUALITY / DELIVERY_QUALITY / EXIT_QUALITY (target sub-take
localization — not implemented).** D-107's forensic confirms
`MediaSignals` and `local_performance.apply_local_performance_to_takes`
produce **whole-clip aggregates only**; no field anywhere localizes a
defect to the entry, interior delivery, or exit of a take. The only
existing entry/interior/exit split in the codebase is
`take_judge.delivery_cleanliness_evidence`'s fixed 0.35 s edge margin —
anchored to the take's raw `start`/`end`, not to a measured speech-end
timestamp. These three names give the target concept a vocabulary; they
require new perception work (Layer 1/2) to populate and are not
authorized by this document.

**Visual-defect ownership (clarifies, does not change, Section 4's
existing upstream/downstream split).**
- **CASE A — post-delivery edge defect** (e.g. a speaker breaks character
  strictly *after* a clean, complete delivery): owned by **BoundaryEngine**
  (Layer 7) — trim the edge; never lower the take's overall BestTake rank
  for a defect outside the delivery itself.
- **CASE B — delivery-overlapping defect** (e.g. a fumble *while*
  speaking the required content): owned by **BestTakeResolver /
  DeliveryScorer** (Layer 6) — the take becomes less usable as a whole
  realization of that idea.
- **CASE C — both present**: recorded independently by each owning
  authority. Neither authority silently compensates for the other's
  finding.

**Authority statement (restates, does not change, Section 4).** BestTake
owns take-level (whole-realization) quality. BoundaryEngine owns
edge-level physical-cut defects. Perceptual Watch+Listen verifies the
*rendered* result and routes findings to whichever of the two owns the
defect class — it never edits selection or boundaries itself.

No change to Section 2's 20 layers, Section 3's status table, or any
accepted D-096/D-097.x authority contract. See `docs/CUTSELL_DECISIONS.md`
D-107 for the empirical pimples-family forensic and signal audit that
motivated naming these concepts now, and for why no BestTake selection
code was changed to implement them in this task.

---

## 10. Behavior + Proposition Abstraction Doctrine, Confidence/Fallback,
and Anti-Rule-Proliferation (D-111)

**Status: additive doctrine, documentation only.** No layer is renumbered,
no status in Section 3's table changes, no new authority is created, no
`cutsell_worker/*.py` file was touched to write this section. This closes
the gap Section 9 left open: D-107 named the *ranking* vocabulary
(BestTake priority order, Performance Continuity, Delivery Energy Fit,
Entry/Delivery/Exit ownership); this section names the *generalization*
and *conflict-resolution* vocabulary the Product Owner's D-111 directive
introduced — how CutSell should keep learning from new creator behavior
without accreting an unbounded pile of one-off rules.

### 10.1 Behavior + Proposition Abstraction Doctrine

CutSell must generalize raw perception into two abstraction families
BEFORE editorial selection runs, rather than accumulating
benchmark-specific lexical/timing/motion/product/creator rules one
failure at a time:

- **BEHAVIOR STATES** (Layer 2's existing target vocabulary — restated,
  not expanded: `AUDIENCE_DELIVERY`, `PRE_TAKE_SETUP`, `FALSE_START`,
  `ABANDONED_ATTEMPT`, `CLEAN_ATTEMPT`, `CORRECTION`, `CONTINUATION`,
  `POST_TAKE_RESET`, `RETAKE_EVENT`, `RECORDING_PROCESS`, plus
  `NEW_AUDIENCE_BEAT` and `BREAKING_CHARACTER` named here for
  completeness). Many different physical behaviors (looking down,
  laughing, freezing, dropping hands, reaching for a phone, resetting
  posture, an expression change, a pause, saying "again", restarting a
  sentence, an abrupt movement) may all, depending on context, contribute
  EVIDENCE toward the SAME state. The target relationship is
  many-signals-to-one-state, not one-signal-to-one-rule.
- **PROPOSITION STATES** (new vocabulary, Layer 4/Layer 10 territory —
  see 10.2). The audience-facing idea/job/claim/feature a piece of speech
  is trying to deliver, distinct from its topic, product, or opening
  words.
- **ATTEMPT RELATIONSHIPS** — restates Layer 4's target relationship set
  with two additions for completeness: `retry_of`, `continuation_of`,
  `corrects`, `supersedes`, `complements`, `duplicate_of`, `setup_for`,
  `same_editorial_function` (all already named in Layer 4), plus
  `new_beat` (a distinct proposition — never grouped as a retry
  competitor) named explicitly here. Canonical meanings: RETRY =
  competing attempts at the SAME proposition/job; COMPLEMENT = additional
  audience-facing content that advances meaning and must not
  automatically compete for one winner; NEW_BEAT = a distinct
  proposition / distinct audience-facing job.
- **PERFORMANCE QUALITY** — Layer 6/16's existing target (multimodal
  ranking evidence), not expanded here.
- **CONFIDENCE** — see 10.4.

Production logic should prefer reasoning from these abstractions
(SIGNALS → GENERAL STATE → EDITORIAL DECISION) over inventing a new
permanent special-case heuristic from one observed failure. This
reaffirms, and gives a name to, the discipline every D-097.x/D-108/D-109/
D-110 bounded fix already followed one at a time (reuse existing
evidence and existing authorities; do not add a parallel decision path).

### 10.2 Proposition Identity Precedes Retry Identity

**New RAW → Cut.ai doctrine, required for Milestone 1, not deferred to
Human Gold (Layer 10-16).**

CRITICAL RULE:

- SAME PRODUCT ≠ SAME PROPOSITION
- SAME TOPIC ≠ SAME PROPOSITION
- SAME OPENER ≠ SAME PROPOSITION
- SAME SENTENCE STRUCTURE ≠ RETRY

Example (illustrative only — never hardcoded into engine logic): "This
32oz cup keeps drinks cold for 29 hours" (proposition: COLD_RETENTION),
"This 32oz cup doesn't dent easily" (proposition: DURABILITY), "This 32oz
cup fits any car cup holder" (proposition: CUP_HOLDER_COMPATIBILITY) share
the same product, the same opener, and similar sentence structure, but
are THREE distinct audience-facing propositions. `same_product=true` AND
`same_opening=true` AND `same_proposition=false` → **do not group as
retries.**

Before forming retry competitors, the engine must first ask "are these
attempts trying to deliver the same proposition/job?" and only THEN ask
"are these alternative attempts/retries of that proposition?" Conceptual
flow: `PRODUCT/TOPIC → PROPOSITION A → attempts/retries → BestTake`
(repeated per distinct proposition B, C, ...). Propositions must never be
collapsed merely because they share vocabulary.

This is a target discipline for the existing retry-family/grouping
authority (`take_grouping.py`/`take_grouping_provider.py`, Layer 4) — it
does not change that authority's code today, and it does not retract
D-108's already-implemented, narrower `blocked_pairs` veto (which acts on
a different, already-proven signal: recorded non-equivalence, not
proposition classification). A future proposition-identity implementation
would be new, separately-authorized engineering, evaluated the same way
every prior bounded fix was (positive tests, negative controls,
regressions, one real-media qualification).

### 10.3 Confidence / Conflict Resolution and the Multimodal Fallback Arbiter

Formalizes three target decision modes for Layer 18 (Confidence +
Abstention), extended with a named fallback consumer:

- **HIGH CONFIDENCE / EVIDENCE AGREEMENT** → deterministic/automatic
  decision (the normal case today: lexical rules, D-108's veto, D-109/
  D-110's directional rejection-respect, D-106's LEVEL_2 reclassification
  are all HIGH-confidence deterministic paths).
- **MEDIUM CONFIDENCE / SIGNAL CONFLICT** → a bounded **multimodal
  fallback arbiter** (target, NOT IMPLEMENTED — see 10.3.1).
- **LOW CONFIDENCE / INSUFFICIENT EVIDENCE** → abstain / preserve safely
  / route to review — never manufacture certainty.

No numerical threshold is introduced by this documentation task;
thresholds require calibration against real evaluation evidence, per the
anti-loop contract (Section 6).

#### 10.3.1 Multimodal Fallback Arbiter (target, not implemented)

The fallback is explicitly **not** the primary editor — it is invoked
only when the structured deterministic + multimodal-evidence system
cannot safely resolve a BOUNDED conflict on its own, and it receives only
the bounded finalists plus necessary context (never the whole family,
never unrestricted authority). Possible future trigger conditions (target
list, not an implementation): Hybrid/Gemini's winner disagrees with the
measured DeliveryScorer winner; the semantic winner has a strong measured
visual-performance defect while another sufficient competitor is cleaner;
top candidates remain very close; retry-vs-complement remains genuinely
unresolved; BestTake-vs-Boundary ownership remains ambiguous (D-107's
CASE C); multimodal signals conflict. Possible conceptual outcomes:
`BEST_TAKE_A`, `BEST_TAKE_B`, `EQUIVALENT`, `KEEP_BOTH_COMPLEMENTARY`,
`GOOD_TAKE_TRIM_EXIT`, `UNCERTAIN`. **Not implemented by this or any
prior task.**

#### 10.3.2 When Fallback Must Not Be Used

Do not escalate to the fallback arbiter when a stronger deterministic
answer already exists. Confirmed examples from this project's own
history: D-110's directional rule (a recorded strict replacement rejection
already says candidate Y is NOT a valid replacement for candidate X — a
later authority must respect that, never re-litigate it via a weaker
arbiter); polarity/negation-inversion safety; an authority conflict
already resolved by canonical safety evidence (D-109's own diagnosis);
deterministic media-integrity failures. Rule: CLEAR DETERMINISTIC ANSWER
→ use it; REAL CONFLICT → fallback may arbitrate; INSUFFICIENT EVIDENCE →
abstain/review.

### 10.4 Escalation Instead of Rule Proliferation (binding doctrine)

When existing deterministic and multimodal evidence cannot safely resolve
a case, the response is escalation through the three modes above, **never**
an automatic new permanent heuristic minted from that one benchmark
failure. A new benchmark failure is evidence for improving a GENERAL
capability (Behavior Understanding, Proposition Understanding, Attempt
Relationships, an existing authority's evidence intake), not automatic
justification for another lexical/motion/product/creator-specific
production rule. This restates, generalizes, and makes binding the
discipline already visible across D-097.x's many `R`-numbered fixes and
D-108/D-109/D-110's authority-collision fixes — each one reused existing
evidence/authorities rather than adding a parallel rule or a parallel
cleanup layer.

**Anti-rule-proliferation classification.** When a new creator/benchmark
failure appears, classify it before writing any code:

- A. Perception missing?
- B. Behavior-state understanding missing?
- C. Proposition identity wrong?
- D. Attempt relationship wrong?
- E. Existing evidence not reaching its owning authority (a plumbing gap
  — see the dataflow map's Gap 1)?
- F. Authority collision (two authorities independently re-deriving the
  same judgment with different evidence, as D-109/D-110 diagnosed and
  fixed)?
- G. BestTake quality issue?
- H. Boundary issue?
- I. Genuine unresolved multimodal conflict (10.3.1 territory)?

Only add a new deterministic production rule when it expresses a GENERAL
invariant supported by architecture/evidence — not a benchmark-specific
special case.

### 10.5 Safety / Architectural Invariants (extends Section 6, does not replace it)

Permanent deterministic rules remain appropriate for universal
invariants, restated and extended here for discoverability: meaning must
not invert; negation/polarity must survive; diagnosis/number/correction
safety; poor performance evidence does not by itself authorize deleting
unique/unreplaced meaning without a valid replacement (D-109/D-110); a
stronger, already-recorded prior replacement rejection cannot be silently
overridden by a weaker later authority re-asking the same question
(D-110's exact fix); same topic/product/opener does not prove same
proposition (10.2); source RAW remains immutable (Section 9); an
unsupported QA capability never silently becomes PASS (Section 2, Layer
8); QA reference videos (Cut.ai, Human Gold) never enter production
selection/prompts/timing; unique/complementary audience-facing meaning
cannot disappear merely because another take is visually cleaner (D-107's
BestTake priority order, meaning above cleanliness).

### 10.6 Hybrid / Gemini Authority (clarifies, does not change, existing authority)

Hybrid/Gemini may provide semantic interpretation, labels, equivalence
evidence, ranking evidence, and bounded arbitration. A model "winner" is
a strong NOMINATION — it is **not** absolute authority against: stronger
deterministic safety evidence, a prior replacement rejection (D-110),
strong measured multimodal disagreement, or a meaning/sufficiency
failure. This restates the doctrine D-109/D-110 already proved
empirically (a Gemini-labelled "winner" does not override a stricter
guard's recorded rejection) as a general principle, not a new authority.
No existing fast path is globally removed by naming this; any behavior
change still requires its own bounded implementation + evaluation proof,
per Section 6.

### 10.7 Human Gold Remains Downstream (restates Section 5, sharpened)

Cut.ai commercial parity (Milestone 1) comes first. "Same product but a
different feature/proposition" (10.2) is a Milestone-1 (RAW → Cut.ai)
requirement, not deferred Human Gold reasoning — proposition identity is
more basic than editorial-function sufficiency (Layer 10-11) or
good-vs-good ranking (Layer 14). Human Gold later adds editorial
function, delivery sufficiency, information gain, the minimum sufficient
set, good-vs-good refinement, composite necessity, and narrative/rhythm
refinement (Layers 10-16, unchanged from Section 2). Nothing in this
section authorizes starting Milestone 2 work.

### 10.8 Evaluation Learning Loop (restates Layer 19, sharpened)

Human/Product-Owner-reviewed cases (a stomach retry, a pimples family, a
complementary-symptom pair, a diagnosis-preservation case, a polarity/"No"
case, a duplicate conclusion, a boundary defect) should become evaluation
cases, regression fixtures, and confidence-calibration evidence — never
converted directly, one at a time, into a new production special-case
rule. Long-term target path (Layer 20, roadmap only):
`examples → eval dataset → confidence calibration → learned/ranked
components when justified by sufficient evaluation evidence`.

### 10.9 Current Implementation State (truthful snapshot at D-111)

- D-110's authority-collision fix (`hybrid_retry_winner_authority`
  respecting a same-run `complete_retry_identity_guard` rejection) is
  IMPLEMENTED offline; targeted + regression tests green.
- D-110's real-media qualification found the fix's own mechanism was
  **not triggered** on RAW `34123511687` — a second, uncoordinated
  authority earlier in the same take-level chain
  (`hybrid_retry_completion_integrity::_safe_failed_retry`) removed the
  same candidate first, via the identical doctrinal defect (an
  independently-derived retry-equivalence judgment that never consults
  the recorded guard rejection) at a different point in the chain. Not
  yet fixed; recorded, not implemented, per that task's own no-fix-loop
  scope.
- D-106's meaning-vs-parity QA distinction is ACTIVE and reconfirmed
  correct on real media (`34123511687`'s regression manifest: papillary
  meaning PASS, parity-only mismatch correctly non-gating).
- D-107's non-destructive doctrine (Section 9) is ACCEPTED; no editor
  surface exists yet.
- `MediaSignals` are PARTIALLY real (7 of 12 fields measured; 5 stay at
  frozen defaults — dataflow map Gap 2).
- Position-aware ENTRY/DELIVERY/EXIT perception (Section 9) is NOT
  IMPLEMENTED.
- The multimodal fallback arbiter (10.3.1) is NOT IMPLEMENTED.
- Explicit proposition-level engine representation (10.2) is NOT fully
  implemented — grouping remains lexical-only (dataflow map Gap 1).
- Upstream perception exists PARTIALLY (dataflow map Section 1-4).
- Downstream Watch+Listen remains PARTIAL/ADVISORY (4 of 8
  `perceptual_watch_listen.py` v1 capabilities EVALUATED, 4
  `NOT_IMPLEMENTED`).
- Cut.ai commercial parity (Milestone 1) is NOT YET achieved (D-110's
  qualification: overall LEVEL_1 28.744s this run, most of it in
  IdeaClusterer/RetryFamilyFormation on families unrelated to pimples).
- Human Gold implementation (Milestone 2) remains downstream, not
  started.

### 10.10 Likely Incremental Roadmap From Here (NOT an authorization)

Documented for continuity only — this ordering is a plausible sequence
given current evidence, not a commitment, and does not authorize any
step:

1. Current: D-110 real-media qualification is done (verdict B, MECHANISM
   AVAILABLE BUT NOT TRIGGERED); the next open item is Product-Owner
   authorization of a `hybrid_retry_completion_integrity` fix analogous
   to D-110, or a broader sweep of the take-level chain for the same
   collision shape.
2. Position-aware performance perception (ENTRY/DELIVERY/EXIT, Section
   9) — if evidence justifies it.
3. Expose reliable DeliveryScorer/MediaSignals/local-winner diagnostics.
4. Proposition identity / relationship consolidation where grouping
   over-collapses same-topic distinct beats (10.2, dataflow map Gap 1).
5. Multimodal BestTake v1 using only real measured evidence (Section 9).
6. Bounded multimodal fallback for genuine conflicts (10.3.1).
7. Boundary use of post-delivery visual reset evidence.
8. Full Video00 Cut.ai parity qualification.
9. Unseen-RAW Cut.ai-level generalization.
10. Only then: the Human Gold intelligence milestone (10.7).

This roadmap is not authorization for any of its own steps. Every step
still requires its own bounded-capability authorization per Section 6.

### 10.11 Anti-Loop Execution Contract (reaffirms Section 6)

Architecture is not authorization. Every behavior-changing implementation
still requires: one bounded capability, one owning authority, evidence,
positive tests, negative controls, regression protection, a cost
boundary, one real-media qualification when justified, and a STOP. A
newly discovered defect does not automatically create a fix, a RAW, a new
heuristic, or a provider call — see Section 6, unchanged.

No change to Section 2's 20 layers, Section 3's status table, Section 9,
or any accepted D-096/D-097.x/D-107/D-108/D-109/D-110 authority contract.
See `docs/CUTSELL_DECISIONS.md` D-111 for the decision-log entry
recording this section's doctrine.

---

## 11. Overlap — Dialogue/Pacing Transition Doctrine (D-129)

**Status: additive doctrine, documentation only.** No layer is renumbered,
no status in Section 3's table changes, no new authority is created, no
`cutsell_worker/*.py` file was touched to write this section. This names
a target USER-FACING editing capability distinct from the existing
internal perception term it is easily confused with.

### 11.1 Naming contract (binding, prevents future ambiguity)

- **`overlaps_delivery`** (existing internal perception term, UNCHANGED,
  never renamed) — a visual/performance event temporally intersects the
  required spoken DELIVERY span. Belongs to Layer 1/2 Perception, Layer 6
  BestTake CASE B (D-122/D-123/D-128), and Layer 7 Boundary ownership
  reasoning (D-107 Section 9's CASE A/B/C visual-defect split).
- **`Overlap`** (NEW, user-facing product term, UI label exactly
  `Overlap`) — tight/overlapping speech and audiovisual transitions
  BETWEEN two already-selected, already-frozen, already-boundary-safe
  consecutive clips, to produce faster, tighter, more conversational,
  more TikTok/UGC-native commercial pacing. This is an editing/pacing
  capability, not a perception signal, and not a BestTake competitor
  concept.
- **Canonical internal feature name (chosen, the only one used going
  forward):** `dialogue_overlap_enabled`. (`speech_transition_overlap_
  enabled` was the considered alternative; not used, to avoid ever being
  read as a third overlap-shaped term alongside `overlaps_delivery` and
  `Overlap`.)

### 11.2 Product intent

Overlap exists to make CutSell edits feel faster, tighter, more
conversational, and less pause-heavy, while preserving intelligibility.
Illustrative only (never hardcoded into engine logic): Clip A ("This cup
keeps drinks cold for 29 hours.") transitioning into Clip B ("And it fits
in your car cup holder.") without overlap leaves a pause-cut-pause gap
before B starts; with tight pacing, B's speech/visual transition begins
with little or no dead separation after A. Target transition vocabulary
for a future implementation (named for future use, not implemented now):
`HARD_CUT`, `TIGHT_CUT`, `J_CUT`, `L_CUT`, `MICRO_AUDIO_OVERLAP`.

### 11.3 Core invariant (binding)

**CLARITY BEFORE SPEED.** CutSell must never overlap intelligible dialogue
so aggressively that the audience cannot understand either phrase. Overlap
may tighten pacing; it may never sacrifice comprehension. This extends
Section 6/10.5's existing safety-invariant discipline — a new permanent
deterministic rule, not a benchmark-specific special case.

### 11.4 Authority and pipeline placement

Overlap is PHYSICAL/PACING behavior. It must NOT change semantic
membership, choose a different BestTake, merge distinct propositions,
create new speech, repair a bad retry family, override meaning safety,
override Selection Freeze, invent words, or hide a semantic mistake.
Canonical order (restates and extends Section 5's Milestone-1 progression
and Layer 7's existing Boundary-then-render placement — inserts a named
target sub-stage, does not renumber the 20 layers):

```
Selection / BestTake (Layer 6)
        v
Selection Freeze
        v
Boundary (Layer 7 -- safe clip start/end, dead-edge removal, ENTRY/EXIT)
        v
Dialogue / Pacing Transition  <-- NEW target sub-capability, Layer 7's
        v                          own execution boundary, NOT a new layer
Renderer (Layer 7 execution / render_plan.py)
```

Overlap occurs strictly AFTER semantic selection is frozen and AFTER
Boundary has established safe edges — it is a target consumer of
Boundary's output, never a replacement for it. Boundary remains
responsible for safe clip start, safe clip end, dead-edge removal, and
ENTRY/EXIT cleanup, unchanged. Overlap controls how two already-safe
neighboring clips interact in time; it must never independently cut
through required speech.

### 11.5 UI contract (target, not implemented)

- **Label:** `Overlap`. **Type:** toggle.
- **OFF:** normal tight/hard transitions; no intentional dialogue overlap.
- **ON:** CutSell MAY use speech-transition overlap where safe and useful
  — this is a permission, not a command. The engine still decides WHERE
  overlap is safe; per-join clarity/safety continues to govern every
  join even when the toggle is ON. "Overlap ON" never means "force
  overlap at every cut."
- The UI must never expose internal engine terminology: `overlaps_
  delivery`, `CASE B`, `delivery_event_count`, or any other Perception/
  BestTake-internal field name. Optional helper copy (illustrative, not
  final): "Faster, tighter speech transitions."

### 11.6 Future Pacing/Dialogue Transition Intelligence (named, not authorized)

A future implementation may consider: speech end/start timing, word
timestamps, breath/pause, semantic phrase completion, neighboring clip
pacing, sentence cadence, audio intelligibility, visual transition
continuity, and narrative energy. No threshold or algorithm is authorized
by naming these inputs here, per Section 6's anti-loop contract.

---

## 12. iOS Native Swift Foundation — Product/Platform Doctrine (D-129)

**Status: additive product/platform doctrine, documentation only.** This
section does not change any engine layer, does not authorize any Swift/
Xcode/camera/upload/UI implementation, and does not change
`cutsell_worker/*.py`. It formalizes iOS as a required, parallel product
foundation and records an HONEST inventory of what already exists in
THIS repository, since `mobile/ios/` was found to already contain real
Swift source — this section documents that truthfully rather than
treating iOS as greenfield.

### 12.1 Honest current-repo inventory (verified by direct inspection, this task)

`mobile/ios/` contains an XcodeGen-driven iOS app skeleton: `project.yml`
(no committed `.xcodeproj`, no `Package.swift` — Xcode project generation
is XcodeGen-based, `xcodegen generate`, unverified in this session) and
27 Swift source files under `mobile/ios/CutSell/` (~3,790 lines total).
`project.yml` already declares `NSCameraUsageDescription`,
`NSMicrophoneUsageDescription`, `NSPhotoLibraryUsageDescription`, and
`NSPhotoLibraryAddUsageDescription` Info.plist strings, and a
configurable `CutSellAPIBaseURL` (defaults to `http://127.0.0.1:8000` in
`APIClient.swift`, overridable via `UserDefaults`, `https`/`http` scheme
validated). `KeychainStore.swift` persists the auth session via Keychain,
not `UserDefaults`/plaintext. A targeted scan of the Swift sources for
embedded API keys/secrets found none.

**This inventory is a truthful file/responsibility mapping, not a
functional or quality claim.** No build was attempted, no simulator or
physical-device run was performed, and no decision-log (`D-xxx`) entry
prior to this one records any iOS build, test, or QA pass. Per this
document's own honesty requirement: **iOS status is SOURCE PRESENT / NOT
BUILD-VERIFIED / NOT DEVICE-VERIFIED / NO TESTFLIGHT** — never
characterized as a proven, working, or shippable app from this inventory
alone.

Apparent (filename/import-based, unverified) responsibility mapping
against Section 12.7's target module list:

| Target module | Apparent existing file(s) |
|---|---|
| Camera | `CameraCaptureView.swift`, `CameraController.swift` (front/back `AVCaptureDevice.Position`, start/stop), `CameraPreview.swift` |
| MediaImport | `NewCutView.swift`, `PickedMediaTransfer.swift` (imports `PhotosUI`) |
| MediaMetadata | `VideoPreparation.swift` |
| Upload | `MultipartUploadManager.swift`, `BackgroundPartUploader.swift`, `UploadResumeStore.swift`, `OverlayUploadManager.swift` |
| ProcessingStatus | `ProcessingView.swift`, `NotificationCenterModel.swift` |
| Playback | `DraftPlaybackView.swift` (imports `AVKit`), `VisualTimelineView.swift`, `TimelineAssets.swift` |
| Editor | `DraftEditorView.swift`, `DraftEditorViewModel.swift`, `EditorExtrasView.swift` (imports `PhotosUI`) |
| ExportShare | `FinishedExportActionsView.swift` (imports `PhotosUI`) |
| Networking | `APIClient.swift`, `KeychainStore.swift` |
| Models | `Models.swift`, `JSONValueHelpers.swift` |
| Diagnostics | `AppState.swift`, `PendingCutStore.swift`, `ProjectsView.swift` (no dedicated diagnostics/logging module apparent) |
| App shell | `CutSellApp.swift` |

No mapping above claims a target capability (Section 12.4's minimum
vertical slice) is functionally COMPLETE, correct, or tested — only that
source purporting to address it exists. Whether it satisfies Section 12.4
item-by-item requires the real-device QA this section requires below
(12.8), not authorized by this document.

### 12.2 iOS product principle

CutSell must validate the engine against footage produced by the actual
mobile capture environment users will use — the worker must not assume
every real iPhone RAW looks like the Video00 fixture set (Section 12.5).
iOS foundation work should therefore begin, and continue, BEFORE the
entire engine/product is considered beta-complete, running IN PARALLEL
with engine development. It does not replace or block current RAW →
Cut.ai engine work (Milestone 1, Section 5) — see Section 12.10.

### 12.3 iOS technology direction

Canonical native direction: **Swift / SwiftUI**, with Apple-native media
frameworks as appropriate (conceptually: `AVFoundation`, `Photos`/
`PhotosUI`, `AVKit`/`AVPlayer`, `URLSession`/background transfer APIs,
`FileManager`/local media storage) — consistent with, not a departure
from, the frameworks the existing `mobile/ios/CutSell/` sources already
import per Section 12.1. No new implementation is authorized by naming
this direction.

### 12.4 iOS minimum vertical slice (target milestone, not a completion claim)

The first iOS milestone is a REAL-DEVICE INGESTION / DELIVERY HARNESS,
not the complete polished editor. Minimum required capabilities: (1)
launch native app shell; (2) camera permission; (3) microphone
permission; (4) Photo Library permission/access; (5) record video; (6)
front camera; (7) back camera; (8) stop recording; (9) retake/delete
local take before upload; (10) import existing video from Photos; (11)
vertical 9:16 capture compatibility; (12) preserve real source
orientation metadata; (13) detect/log codec, container, duration,
resolution, FPS, variable-frame-rate when detectable, audio format/sample
rate, orientation, front-camera mirroring metadata; (14) upload RAW to
CutSell backend; (15) upload progress; (16) retry failed upload safely;
(17) background/interrupted upload strategy; (18) receive processing/job
status; (19) receive completed CutSell render; (20) play result
on-device; (21) save/export result; (22) share result through native iOS
share mechanisms when appropriate. Section 12.1's table names apparent
existing source per item; none of the 22 is asserted DONE by this
document.

### 12.5 iOS media reality contract

The engine must eventually be tested against real iPhone media including
HEVC/H.265 (where produced by device/settings), H.264, MOV/MP4, variable
frame rate, high-resolution capture, front-camera mirroring, rotation/
orientation metadata, interrupted recordings, long recordings, large
files, different iPhone generations, and different supported iOS
versions. The worker must not assume all phone RAWs look like Video00
fixtures (extends the existing Video00 QA-reference doctrine in
`CLAUDE.md`'s quality-ladder section — those references stay QA-only and
this does not change that).

### 12.6 iOS → engine ingestion contract

The iOS layer should deliver immutable original-source identity to the
engine. Required conceptual upload metadata: asset id, local/source
filename, duration, codec, container, resolution, fps/frame-rate
metadata, orientation, audio metadata, capture/import origin, and source
timestamp metadata where applicable. Source RAW remains immutable under
this document's Section 9 (D-107) non-destructive-editing doctrine and
D-111's restated invariant (Section 10.5) — this section does not change
that doctrine, it names iOS as one of the producers that must honor it.

### 12.7 iOS non-destructive editor compatibility

The future iOS editor must remain compatible with CutSell's canonical
**PRESELECTED / NOT_PRESELECTED** source-range model (Section 9) — the AI
result is an editable draft, never a destructive mutation of source
media. Future editor behavior should be able to support trim, extend
source handles, restore source range, remove, reorder, split, captions,
overlays, audio controls, and undo/redo, and future alternate-take/manual
controls ONLY if product scope later authorizes them. **This document
does NOT reintroduce automatic SWAP into Clean Cut V1** — D-019's
KEEP/DISCARD-only doctrine and `CLAUDE.md`'s SWAP-out-of-scope decision
are unchanged; `draft_edits.py`'s manual editor-layer `swap_take` remains
the distinct, already-carved-out product layer CLAUDE.md already names.

Target future modular Swift package structure (conceptually): `CutSellApp`,
`Camera`, `MediaImport`, `MediaMetadata`, `Upload`, `ProcessingStatus`,
`Playback`, `Editor`, `ExportShare`, `Networking`, `Models`,
`Diagnostics` — Section 12.1's table already maps most of these onto
existing files informally within one target, without Swift Package
Manager module boundaries. If SPM modules are used in a future
implementation, module boundaries should stay aligned to this list; not
every module needs to become a separate package prematurely — the goal
is maintainable native architecture, not package proliferation.

### 12.8 iOS Overlap UI requirement

The future iOS editing/settings surface must support a toggle labeled
`Overlap` (Section 11.5), wired to the canonical internal feature name
`dialogue_overlap_enabled` (Section 11.1) — never exposing the internal
performance term `overlaps_delivery`. Not implemented by this document.

### 12.9 iOS privacy, security, and observability requirements (target)

**Privacy/permissions.** Camera, Microphone, and Photos usage require
correct Apple permission declarations and user-facing reasons; `project.
yml` already declares placeholder strings for all three (Section 12.1) —
final approved user-facing copy is NOT invented by this document.

**Auth/security (target).** Authenticated upload requests; no API/
provider secrets embedded in the iOS client (Section 12.1's scan found
none today); signed/temporary upload access where appropriate; the
backend owns provider credentials (consistent with this repo's existing
provider-abstraction doctrine); TLS only (`APIClient.swift` already
validates `https`/`http` scheme, defaulting to a local dev URL); a local
temporary-media cleanup policy; no silent upload of media without a
user-initiated action/product flow.

**Observability (target).** The real-device ingestion harness should log
diagnostically: device model, iOS version, capture/import source, codec,
resolution, fps, orientation, duration, file size, audio format, upload
start/end, upload retry, backend job id, processing duration, and render
playback success/failure — without logging private media content
unnecessarily.

### 12.10 iOS real-device QA requirement (gates beta, not Milestone 1)

Before TestFlight beta closure, CutSell must be proven on physical
iPhones. Simulator-only success is insufficient for camera, microphone,
encoding, orientation, background upload, large-file handling, and
playback/export. This requirement is downstream of, and does not block,
the current engine milestone (Section 12.2/12.11).

### 12.11 iOS parallel delivery roadmap (not an authorization for any phase)

- **Phase 0** — canonical architecture/docs (this section).
- **Phase 1** — Swift app shell + real camera/import/upload/playback
  vertical slice. Per Section 12.1, SOURCE EXISTS in this repo mapping to
  most of Section 12.4's items by file name; build/device verification of
  this phase is NOT yet recorded in any decision-log entry.
- **Phase 2** — real-device ingestion QA against the CutSell engine.
- **Phase 3** — editable mobile draft/timeline.
- **Phase 4** — Overlap toggle (Section 11) + pacing controls and other
  editor settings.
- **Phase 5** — signing/provisioning/TestFlight readiness.

No phase is implemented, started, advanced, or authorized by this
document.

### 12.12 Engine roadmap relationship

The current engine milestone remains unchanged: **RAW → CLEAN RAW →
CUT.AI COMMERCIAL PARITY** (Section 5, Milestone 1). The iOS track runs
IN PARALLEL and does not block D-128 (multimodal fallback) or any other
authorized engine work merely because the complete iOS editor is
unfinished. Before beta is considered physically/product-complete,
real-iPhone capture/import/upload/playback validation (Section 12.10) is
required.

### 12.13 Canonical product architecture map

This does not create a second engine — it places the existing 20-layer
engine (Section 2) inside its real product context:

```
iOS CAPTURE / IMPORT
        v
IMMUTABLE RAW INGESTION  (Section 9 / 12.6 -- source-range handle)
        v
CUTSELL PERCEPTION / UNDERSTANDING  (Layers 1-4)
        v
CLEAN CUT / BESTTAKE / BOUNDARY  (Layers 5-9)
        v
PACING / DIALOGUE TRANSITION  (Section 11 -- Layer 7's execution boundary)
        v
RENDER  (Layer 7 execution)
        v
IOS EDITABLE DRAFT / PLAYBACK / EXPORT  (Section 12.7)
```

### 12.14 No implementation authorization (binding, restates Section 6)

Architecture is not authorization. This section does NOT authorize:
creating an Xcode project, Swift code, camera implementation,
`AVFoundation` work, UI work, Overlap engine behavior, transition-timing
changes, backend API changes, TestFlight submission, Apple signing,
provider calls, or a RAW run. Every implementation requires its own
bounded task, per Section 6's anti-loop/execution contract.

---

No change to Section 2's 20 layers, Section 3's status table, Section 9,
Section 10, or any accepted D-096/D-097.x/D-107/D-108/D-109/D-110/D-111/
D-128 authority contract. See `docs/CUTSELL_DECISIONS.md` D-129 for the
decision-log entry recording Sections 11-12's doctrine.

---

## 13. Parallel Multimodal Perception + Watch+Listen Multimodal
## Understanding (D-148)

**Status: additive doctrine, documentation only.** No layer is renumbered,
no status in Section 3's table changes, no new authority is created, no
`cutsell_worker/*.py` file was touched to write this section. D-098
remains foundational; D-111 onward remain additive refinements. This
section CONSOLIDATES them into a clearer PERCEPTION → UNDERSTANDING →
EDITORIAL REASONING reading of the same architecture, motivated directly
by the real-media finding in `docs/CUTSELL_DECISIONS.md` D-147: provider
semantic comparison, even when `family_complete_context=true`, produced
two independently "complete" windows that disagreed with each other on
the pimples/espinillas family's winner. That is direct evidence that
provider semantic comparison is ONE evidence source among several this
architecture already names (Layer 1-4) — it must never become the SOLE
perception/understanding layer for the RAW. Nothing below authorizes
implementing anything; every capability named here is subject to Section
6's anti-loop/execution contract exactly like every prior section.

### 13.1 Updated canonical top-level pipeline (restates, does not renumber, Sections 2/5)

This is a clearer READING of the existing 20 layers plus the existing
Section 11 pacing sub-stage — no layer number changes, no new authority
is created, no existing file's ownership moves:

```
RAW
  v
PARALLEL MULTIMODAL PERCEPTION            (Layer 1, Section 13.2 -- four
  v                                        tracks, conceptually concurrent)
WATCH+LISTEN MULTIMODAL UNDERSTANDING /
FUSION                                    (Layer 1->2 boundary, upstream
  v                                        role -- Section 13.3; NOT Layer 8's
                                           downstream role, see 13.3.4)
BEHAVIOR STATE                            (Layer 2, Section 13.5; restates
  v                                        D-111 Section 10.1, unchanged)
PROPOSITION IDENTITY                      (Layer 4/10, Section 13.6;
  v                                        restates D-111 Section 10.2 verbatim)
ATTEMPT RELATIONSHIPS                     (Layer 4, Section 13.7; restates
  v                                        D-111 Section 10.1's relationship
                                           vocabulary)
CONFIDENCE / CONFLICT                     (Layer 18, Section 13.13; restates
  v                                        D-111 Section 10.3/10.4)
STABLE FAMILY FORMATION                   (Layer 4, Section 13.8; the
  v                                        existing take_grouping.py/
                                           take_grouping_provider.py/
                                           hybrid_session_cleanup.py
                                           authority, evidence intake widened
                                           per 13.8 -- code unchanged here)
BESTTAKE                                  (Layer 6, Section 13.10; restates
  v                                        D-107 Section 9's priority order)
SELECTION FREEZE                          (Layer 5/6 boundary, unchanged)
  v
BOUNDARY                                  (Layer 7, Section 13.11; restates
  v                                        D-107 Section 9's CASE A/B/C split)
DIALOGUE / PACING TRANSITION              (Layer 7 execution sub-stage,
  v                                        Section 13.12; restates D-129
                                           Section 11.4's existing placement)
RENDERER                                  (Layer 7 execution, unchanged)
  v
DOWNSTREAM WATCH+LISTEN QA                (Layer 8, Section 13.3.4/13.9;
                                           restates Section 4's role B)
```

Then, and ONLY after stable RAW → Cut.ai commercial parity (Section 5
Milestone 1, Section 13.14):

```
  v
HUMAN GOLD EDITORIAL REFINEMENT           (Layers 10-16, Section 13.15;
                                           unchanged, still not started)
```

### 13.2 Parallel Multimodal Perception (Layer 1, four tracks)

Canonizes Layer 1 (Media Perception) as four conceptually PARALLEL
tracks. This restates and organizes evidence Layer 1 already names — it
does not add a new layer or move any file's ownership. **No canonical
requirement is established that the transcript (Track A) must finish
before visual or audio perception (Tracks B/C) starts** — they are
independent perception work over the same RAW and may execute
concurrently wherever technical dependencies permit (Section 13.16).

- **Track A — Speech / Language.** Consumes RAW audio. Produces ASR
  transcript, word timestamps, phrase/sentence spans, speech boundaries,
  language/text evidence. Existing owner: `asr.py`,
  `canonical_asr_evidence.py` (D-052 Part A's provider-neutral canonical
  ASR evidence + deterministic fingerprinting). **IMPLEMENTED** (Section
  3's existing L1 row).
- **Track B — Audio Perception.** Consumes RAW audio. Produces, where
  actually implemented today: speech activity / silence / dead-air /
  pause detection, audio continuity, local audio usability, timing
  evidence. Existing owner: `audio_silence.py`, `silence_analysis.py`,
  `audio_boundary_completion.py`. **PARTIAL** — these are genuine
  waveform-derived signals (energy/VAD-based silence and pause
  measurement), not a stub, but they do NOT constitute real semantic or
  prosodic audio understanding (tone, emphasis, emotional affect from the
  audio signal itself). A future capability may add real semantic/
  prosodic perception; per Section 13.4, this is **DESIGNED_NOT_
  IMPLEMENTED** today and must never be claimed as already existing.
- **Track C — Visual / Performance Perception.** Produces face/gaze
  evidence, expression evidence, body/hand motion, camera disengagement,
  resets, fumbles, performance continuity, entry/delivery/exit events,
  scene/performance evidence. Existing owners: `visual_analysis.py`
  (contract/dataclasses, `NoopVisualProvider` default), `visual_openai.py`
  (a REAL `OpenAIVisualProvider` adapter that samples frames and asks
  `gpt-4o-mini` to score `MediaSignals`' 10 visually-scored fields),
  `local_performance.py` (`apply_local_performance_to_takes`),
  `speech_visual_microtrim.py`. **PARTIAL** per Section 3's existing L1/L2
  row and Section 10.9's existing finding (`MediaSignals` 7 of 12 fields
  measured in practice; 5 stay at frozen defaults) — not re-audited
  field-by-field in this task; restated from the existing, not-yet-
  superseded D-107 forensic. Position-aware ENTRY/DELIVERY/EXIT
  localization (Section 9) remains **NOT IMPLEMENTED**, unchanged.
- **Track D — Media / Timing.** Produces source identity, duration, fps,
  orientation, technical stream facts, scene/timing metadata, source-safe
  timing references. Existing owner: ffmpeg/ffprobe media-integrity
  probes already named in Section 3's L1 row, `canonical_asr_evidence.py`'s
  fingerprinting, and (target, iOS-side) Section 12.6's ingestion
  metadata contract. **IMPLEMENTED** for the engine-side technical facts;
  the iOS-side contract (Section 12.6) remains **DESIGNED_NOT_
  IMPLEMENTED** per Section 12's own status.

### 13.3 Watch+Listen Multimodal Understanding (upstream fusion layer)

**Purpose:** understand what is happening in the RAW BEFORE editorial
selection runs — the explicit upstream fusion point Section 4's "Role A"
already named informally. This section gives it an explicit name and
conceptual output (13.3.1) without creating a new authority or a new
Layer number.

It combines: ASR/transcript, word timing, real audio evidence, visual/
performance evidence, timing evidence, behavioral evidence, and media
context (Tracks A-D, Section 13.2) into evidence usable by Layers 2-6.

#### 13.3.1 The Structured RAW Understanding Map (conceptual, no schema implementation required)

Per bounded source span / attempt / beat, the target conceptual output
may contain: `source_asset_id`, `source_start`, `source_end`;
`transcript`, `word_timings`; `speech_state`, `audio_state`,
`visual_state`; `behavior_state`; `entry_state`, `delivery_state`,
`exit_state`; `audience_delivery_span`; `proposition_candidate_id`,
`proposition_relation`; `attempt_relation`; `meaning_sufficiency`;
`retry_candidate`, `correction_candidate`, `continuation_candidate`,
`complementary_candidate`, `new_audience_beat_candidate`;
`performance_usability`, `audio_usability`, `visual_usability`,
`editability`; `confidence`, `conflict_flags`; `evidence_provenance`.

This restates and generalizes Layer 3's existing "Canonical Multimodal
Attempt Evidence" target (Section 2) — the Structured RAW Understanding
Map is Layer 3's target shape, named explicitly and given a full
candidate field list. **No production schema is implemented by naming
this** (Layer 3's own Section 3 status remains what it already was:
target, not built). `whole_video_context.sources[].events`
(`TemporalEvent`/`SourceVideoContext`, `whole_video_analysis.py`) is
today's real, narrower, partial instance of a few of these fields
(behavioral/reset events with a source time range) — it is not
retroactively claimed to already BE the full map.

#### 13.3.2 Authority principle (binding, restates and generalizes Section 4 + D-111 Section 10.6)

**PERCEPTION PROPOSES EVIDENCE. STRUCTURED EDITORIAL AUTHORITIES DECIDE.**
Watch+Listen Multimodal Understanding must NOT mean "one multimodal LLM
watches the entire RAW and decides the edit." It may produce evidence,
hypotheses, relations, confidence, and conflict flags (the Structured RAW
Understanding Map's own fields, 13.3.1). Structured editorial authorities
— Proposition Identity, Family Formation, BestTake, Boundary, Freeze —
remain explicit, separately owned, and unchanged by this section. This
generalizes D-111 Section 10.6's existing "Hybrid/Gemini authority"
doctrine (a model "winner" is a strong nomination, never absolute
authority) from one provider to the whole upstream understanding layer:
a fused multimodal understanding output is likewise a strong evidence
proposal, never a bypass of the authorities that decide membership,
retry topology, or BestTake.

#### 13.3.3 Real audio honesty (binding — audit result, not aspiration)

"Watch+Listen" means genuine listening only when real audio is actually
provided to and processed by a real audio-analysis component. **Transcript
+ timestamps alone are NOT listening.** Audit result (Section 13.2 Track
B): real waveform-derived audio evidence (silence/dead-air/pause/
continuity) IS genuinely computed today — **PARTIAL**, not absent. Real
semantic or prosodic audio understanding (tone, stress, emotional
affect, or meaning derived from the audio SIGNAL rather than the ASR
TEXT) is **DESIGNED_NOT_IMPLEMENTED** — named as a future Track B
capability in Section 13.2, never claimed as already existing. Every
semantic judgment in today's engine (Hybrid/Gemini editorial judging,
`semantic_idea_equivalence.py`) operates on ASR TEXT, not on the raw
audio waveform — this is TRACK A evidence dressed as understanding, and
must be described that way, never as "the engine listened to the audio
and understood it."

#### 13.3.4 Two Watch+Listen roles (restates and sharpens Section 4 — binding, never conflate)

1. **Upstream Watch+Listen Multimodal Understanding** (this section,
   13.3): RAW → see + hear → perception evidence → Structured RAW
   Understanding Map (13.3.1) → informs editorial decisions (Layers 2-6).
   Today's real instance remains what Section 4 already named: the local
   multimodal "performance evidence" events consumed by
   `AttemptReconstructor`/pre-group credit logic and by
   `perceptual_watch_listen._reset_debris_at_edges`.
2. **Downstream Watch+Listen QA** (Layer 8, Section 4's existing "Role
   B"): rendered result → see + hear the ACTUAL output → diagnose,
   classify, route. It must NEVER silently edit — it routes findings to
   Selection/BestTake for wrong content, Boundary for bad physical cuts,
   or Renderer for A/V defects, exactly as Section 4 and D-095 already
   specify. Today's real instances remain `perceptual_watch_listen.py`
   (v1, advisory, 4 of 8 capabilities EVALUATED, 4 `NOT_IMPLEMENTED`, per
   Section 3's existing row) and `benchmarks/clean_raw_checkpoint.py`.

They may reuse the same underlying audio/frame/face/pose/semantic
analysis primitives (e.g. the same `OpenAIVisualProvider`-shaped adapter
could in principle serve both). **Their authorities remain different** —
this is unchanged from Section 4 and is restated, not modified, here.

### 13.4 Behavior State contract (restates D-111 Section 10.1, unchanged vocabulary)

Canonical states: `AUDIENCE_DELIVERY`, `PRE_TAKE_SETUP`, `FALSE_START`,
`ABANDONED_ATTEMPT`, `CLEAN_ATTEMPT`, `CORRECTION`, `CONTINUATION`,
`NEW_AUDIENCE_BEAT`, `POST_TAKE_RESET`, `RECORDING_PROCESS`,
`BREAKING_CHARACTER` — all already named in Section 2 Layer 2 and D-111
Section 10.1; no new state is added here. Many different physical
manifestations (looking down, laughing, freezing, dropping hands,
reaching for a phone, resetting posture, an expression change, a pause,
saying "again", restarting a sentence, an abrupt movement) may all
contribute EVIDENCE toward the SAME state — the target relationship
remains many-signals-to-one-state (D-111 Section 10.1), never
one-signal-to-one-rule. **Terminology note (clarifies, does not rename):**
the existing code-level term for the physical retry-manifestation
instance is `RETAKE_EVENT` (Layer 2, Section 2); the relationship-level
concept "this attempt is a RETRY of that one" belongs to the Attempt
Relationship vocabulary (Section 13.7), not to the Behavior State
vocabulary — a behavior state describes what is happening AT a moment; an
attempt relationship describes how TWO spans relate to each other. No
rename of either existing term is made by drawing this distinction.

### 13.5 Proposition Identity contract (restates D-111 Section 10.2 verbatim, unchanged)

**PROPOSITION IDENTITY PRECEDES RETRY IDENTITY.** Canonical invariant,
unchanged: SAME TOPIC ≠ SAME PROPOSITION; SAME PRODUCT ≠ SAME PROPOSITION;
SAME OPENER ≠ SAME PROPOSITION; SAME SENTENCE STRUCTURE ≠ RETRY. Watch+
Listen Multimodal Understanding (13.3) may supply SUPPORTING EVIDENCE
toward a proposition-identity judgment (e.g. visual/behavioral context
corroborating that two takes address different audience-facing jobs even
when their opening words match) — Proposition Identity itself remains a
structured reasoning stage (Layer 4/10), not something Watch+Listen
decides unilaterally, per 13.3.2's authority principle. This is a
Milestone-1 (RAW → Cut.ai) requirement, not deferred Human Gold
reasoning, unchanged from D-111 Section 10.7.

### 13.6 Attempt Relationship contract (restates D-111 Section 10.1's vocabulary)

Canonical relationship vocabulary: `RETRY`, `CORRECTION`, `CONTINUATION`,
`COMPLEMENTARY`, `NEW_AUDIENCE_BEAT`, `DISTINCT_PROPOSITION`,
`UNCERTAIN` — or the exact repo-conventional equivalents already named in
D-111 Section 10.1 (`retry_of`, `continuation_of`, `corrects`,
`supersedes`, `complements`, `duplicate_of`, `setup_for`,
`same_editorial_function`, `new_beat`) and already given concrete,
implemented form as D-145's own 5-way relation outcome
(`SAME_PROPOSITION_RETRY` / `_CONTINUATION` / `_COMPLEMENTARY` /
`DISTINCT_PROPOSITION` / `UNCERTAIN`, `docs/CUTSELL_DECISIONS.md` D-145).
**`UNCERTAIN` is a valid, first-class outcome — do not force family
formation merely because a relationship could not be confidently
classified.** This is a naming consolidation across three prior
directives (D-111, D-145), not a new vocabulary and not a code change.

### 13.7 Family Formation role (widens evidence intake doctrine; code unchanged here)

Family Formation (`take_grouping.py`, `take_grouping_provider.py`,
`hybrid_session_cleanup.py`, `semantic_idea_equivalence.py`) should
consume transcript semantics + proposition evidence (13.5) + behavior
evidence (13.4) + multimodal timing/performance evidence where relevant —
**it must NOT depend solely on textual/provider comparative judgments.**
The existing semantic provider (Hybrid/Gemini) remains useful; it is ONE
evidence/authority component among several this architecture already
names, not the sole one. This is the direct architectural generalization
of D-144's root cause (SEMANTIC_PROVIDER_VARIANCE / FAMILY_FORMATION_
VARIANCE) and D-147's real-media proof that provider semantic comparison
alone can disagree with itself even under `family_complete_context=true`
— naming the target evidence-breadth requirement, not implementing a
fix. No `cutsell_worker/*.py` file's actual evidence intake changes by
naming this.

### 13.8 D-145 / D-146 / D-147 compatibility (strengthens, does not replace, the existing doctrine)

Preserved unchanged: **NO COMPLETE FAMILY CONTEXT → NO AUTHORITATIVE
COMPARATIVE WINNER** (D-145).

D-147 strengthens this doctrine with a real-media-proven second clause:

**COMPLETE FAMILY CONTEXT + MULTIPLE COMPLETE WINDOWS DISAGREE → NO
AUTHORITATIVE COMPARATIVE WINNER.**

Therefore, future semantic authority requires BOTH family completeness
AND decision consistency/consensus sufficient for authority, or explicit
abstention — **the final consensus implementation is NOT defined here**;
that belongs to Phase A.2 (D-149, Section 13.19) and Phase B, both
separately authorized.

#### 13.8.1 Complete-Window Disagreement / `COMPLETE_CONTEXT_CONFLICT` (new named state, doctrine only)

Canonizes the newly proven structural state D-147 evidenced on real
media: `family_complete_context=true` does NOT imply semantic certainty.
The possible state is named **`COMPLETE_CONTEXT_CONFLICT`** — meaning all
relevant competitors were present in at least one single request, but
independent legitimate comparisons disagree with each other. **Required
high-level policy: conflict must remain conflict.** A `COMPLETE_CONTEXT_
CONFLICT` must NOT be merged into a winner merely because each individual
request was, on its own, complete. No detector or gate implementing this
state is built by this section — D-149 (Section 13.19) is the named next
step, not implemented here.

### 13.9 Provider role (restates and generalizes D-111 Section 10.6)

Provider output is evidence. **Provider consistency itself is evidence**
— D-147's own finding (the same provider/model disagreeing with itself
across two complete windows in one run) IS a conflict signal in its own
right, per the Confidence/Conflict ladder (Section 13.13). A provider
label must never become unquestioned ontology for proposition, retry,
family topology, or BestTake when other structured or multimodal evidence
conflicts — restated from D-111 Section 10.6, now generalized explicitly
to the provider's OWN cross-request consistency, not only to disagreement
with a different authority.

### 13.10 BestTake role (restates D-107 Section 9's priority order, unchanged)

Preserve the canonical BestTake ordering, highest first: (1) MEANING/
MESSAGE SUFFICIENCY, (2) TAKE USABILITY, (3) MULTIMODAL PERFORMANCE
QUALITY, (4) EDITABILITY/BOUNDARY QUALITY, (5) NARRATIVE/ENERGY FIT.
Multimodal Understanding (13.3) provides evidence into tiers (2)-(3); it
does not directly become the winner. Hybrid/Gemini/OpenAI nomination
remains non-absolute, per 13.9 and D-111 Section 10.6.

### 13.11 Boundary role (restates D-107 Section 9's CASE A/B/C split, unchanged)

Preserve: a defect AFTER required spoken delivery → BoundaryEngine
(Layer 7); a defect DURING required delivery → BestTake/DeliveryScorer
evidence (Layer 6). Watch+Listen (13.3) should improve the EVIDENCE
supporting this distinction (e.g. a cleaner entry/delivery/exit
localization, Section 9's named-not-implemented target) — it does not
merge Boundary's authority into perception, and Boundary's ownership of
physical timing is unchanged.

### 13.12 Dialogue/Pacing role (restates D-129 Section 11, unchanged placement)

Preserve: Selection/BestTake → Freeze → Boundary → Dialogue/Pacing
Transition → Renderer (D-129 Section 11.4, unchanged). Pacing cannot
reopen Proposition Identity, Attempt Relationships, Family Formation, or
BestTake — it operates strictly on already-frozen, already-boundary-safe
membership, exactly as D-129 Section 11.4 already requires.

### 13.13 Confidence / Conflict policy (restates D-111 Section 10.3/10.4, unchanged)

Preserve the canonical decision ladder: HIGH CONFIDENCE → structured
automatic decision; MEDIUM CONFIDENCE / CONFLICTING EVIDENCE → bounded
arbitration where separately authorized (10.3.1's fallback arbiter,
still NOT IMPLEMENTED); LOW CONFIDENCE → abstain / preserve / review.
Provider-vs-provider disagreement (13.9) and complete-window-vs-complete-
window disagreement (13.8.1) are both real conflict signals under this
ladder. Per D-111 Section 10.4, a rare case discovered this way must
escalate through this ladder — **never** become a new regex, marker, or
timestamp-specific permanent rule.

### 13.14 RAW → Cut.ai Level-1 contract (restates Section 5 Milestone 1, unchanged)

Level 1 (RAW → Cut.ai commercial parity) requires: stable perception
(Section 13.2), stable understanding (Section 13.3), correct attempts
(Layer 3), correct proposition relations (13.5), stable retry/family
formation (13.7-13.8), meaning-safe BestTake (13.10), Boundary (13.11),
Pacing (13.12), and a commercially valid render. Unchanged from Section 5.

### 13.15 Human Gold Level-2 contract (restates Section 5 Milestone 2 + Section 10.7, unchanged)

Level 2 (Cut.ai parity → Human Gold refinement, Layers 10-16) may add:
minimal sufficient editorial set, narrative compression, preferred
realization, stronger context/energy fit, advanced composite judgment,
tighter aesthetic choices, advanced commercial editorial taste.
**Human-Gold-only preferences do NOT block Level 1** — unchanged from
Section 5/10.7; Milestone 2 is not started and not authorized by this
section.

### 13.16 Parallel execution design (documentation only, no orchestration code)

```
RAW
 |-- ASR / transcript / word timings        (Track A, 13.2)
 |-- audio perception                       (Track B, 13.2)
 |-- visual/performance perception          (Track C, 13.2)
 `-- media/scene/timing probe               (Track D, 13.2)
        |
       JOIN
        v
Watch+Listen Multimodal Understanding (13.3)
```

Benefits (documented, not implemented): independent perception work can
execute concurrently; less end-to-end latency; the transcript does not
become privileged before visual/audio evidence exists merely by pipeline
ordering; evidence arrives symmetrically into Understanding. This is a
target execution shape for a future orchestration implementation — no
`cutsell_worker/*.py` scheduling/concurrency code is added or changed by
this section.

### 13.17 Bounded multimodal fallback distinction (restates D-098 Section 10.3.1/10.3.2, sharpened)

Distinguish, clearly and permanently:

- **Upstream Watch+Listen Multimodal Understanding** (13.3): broad RAW
  understanding/evidence production, feeding Layers 2-6 generally.
- **Bounded Multimodal BestTake Fallback** (D-127 through D-141,
  `multimodal_besttake_fallback.py`/`multimodal_besttake_arbiter.py`,
  Phase 1 SHADOW-ONLY per D-128, still not activated): narrow arbitration
  between a small number of already-identified legitimate finalists when
  structured evidence is still unresolved after Layers 2-6 have already
  run.

They are **NOT the same system role.** The fallback is a narrow,
downstream-of-structured-reasoning consumer of a BOUNDED conflict; Watch+
Listen Understanding is broad, upstream, and feeds the structured
reasoning itself. D-127 through D-141 remain preserved CLOSED, unchanged,
not reactivated by this section.

### 13.18 Non-destructive editing contract (restates D-107 Section 9, unchanged)

Preserve: RAW immutable; PRESELECTED / NOT_PRESELECTED membership
(never "deleted"); recoverable source ranges and handles; manual editor
recovery / alternate-take restoration remains a future editor-surface
capability (Layer 20 / Section 12.7 territory), not implemented here. No
perception, understanding, or fusion stage named in this section
physically deletes RAW footage — Watch+Listen Multimodal Understanding
(13.3) is strictly an evidence producer, same as every existing
perception component in Section 3's table.

### 13.19 Anti-rule-proliferation doctrine (promotes D-111 Section 10.4 to top-level, unchanged in substance)

**CutSell must NOT learn thousands of video-specific error rules.** It
generalizes through Behavior State (13.4), Proposition Identity (13.5),
Attempt Relationship (13.6), multimodal performance evidence (13.2-13.3),
and Confidence/Conflict (13.13) — restated as top-level canonical
doctrine here, not a new rule. Hard, permanent rules remain appropriate
ONLY for genuine invariants: meaning preservation, negation/polarity
safety, number safety, source identity, authority ordering, family
completeness (D-145), complete-context conflict (13.8.1), fail-open
defaults, and truthful QA status reporting (Layer 8) — restates D-111
Section 10.5, unchanged, with `COMPLETE_CONTEXT_CONFLICT` (13.8.1) added
to the invariant list as a direct consequence of D-147's real-media proof.

### 13.20 Current implementation status matrix (audit, strict — distributed signals ≠ completed fusion)

| Component | Status | Basis |
|---|---|---|
| ASR | IMPLEMENTED | `asr.py`, `canonical_asr_evidence.py` — feeds AttemptReconstructor, take_judge, live_render_qc, perceptual_watch_listen today (Section 3) |
| Word timing | IMPLEMENTED | ASR word-level timestamps, consumed by Boundary/AttemptReconstructor (Section 3, Layer 7) |
| Audio perception (silence/dead-air/pause) | PARTIAL | `audio_silence.py`, `silence_analysis.py` — genuine waveform-derived signal, narrow scope |
| Real semantic/prosodic audio understanding | DESIGNED_NOT_IMPLEMENTED | No component analyzes tone/prosody/affect from the raw audio signal (13.3.3) |
| Visual perception | PARTIAL | `visual_openai.py`'s `OpenAIVisualProvider` is a real GPT-4o-mini frame-scoring adapter; `NoopVisualProvider` is the contract default; production wiring/coverage not re-verified in this task (13.2 Track C) |
| Position-aware performance (ENTRY/DELIVERY/EXIT) | DESIGNED_NOT_IMPLEMENTED | D-107 Section 9 — named, no producer exists |
| Behavior understanding | PARTIAL | Local multimodal "performance evidence" events consumed by `AttemptReconstructor`/pre-group credit and `perceptual_watch_listen._reset_debris_at_edges`; no full behavior-state fusion producer (Section 3) |
| Proposition identity | PARTIAL | D-145's 5-way relation vocabulary is implemented in the family-formation-adjacent design layer (semantic_idea_equivalence.py's pairwise template + D-145's own design); grouping's PRIMARY signal remains lexical, per D-111 Section 10.9's existing finding ("dataflow map Gap 1") |
| Attempt relationships | PARTIAL | Restart-evidence kinds + D-100's `multimodal_corroborated_retry` exist; explicit typed `retry_of`/`corrects`/`complements` beyond those kinds does not (Section 3) |
| Family formation | EXISTING, PROVIDER-EVIDENCE-DEPENDENT | `take_grouping.py`/`take_grouping_provider.py`/`hybrid_session_cleanup.py` — D-144/D-147 proved the provider-comparison component of this authority can be unstable even when family-complete (13.7-13.8) |
| Parallel perception orchestration | DESIGNED_NOT_IMPLEMENTED | Section 13.16 is a documented target shape; no scheduling/concurrency code exists for it |
| Unified multimodal fusion | DESIGNED_NOT_IMPLEMENTED | No single component produces the Structured RAW Understanding Map (13.3.1); today's evidence is distributed across ASR/audio/visual/behavior components consumed piecemeal by different downstream authorities — **distributed signals do NOT equal completed fusion** |
| Structured RAW Understanding Map | DESIGNED_NOT_IMPLEMENTED | Conceptual field list only (13.3.1); `whole_video_context.sources[].events` is a narrower, partial, already-real precursor, not the full map |
| BestTake | EXISTING | `deterministic_best_take_authority.py`, `realization_resolver.py`, `take_judge.py`'s DeliveryScorer — RAW-proven across D-097.x (Section 3) |
| Boundary | EXISTING | `boundary_engine_pass.py` — RAW-proven frame-exact joins (D-097.4, Section 3) |
| Dialogue/Pacing | EXISTING (Phase 1) | `dialogue_pacing_transition.py` (D-142) — HARD_CUT/TIGHT_CUT executable; J_CUT/L_CUT/MICRO_AUDIO_OVERLAP require renderer extension (Section 3-adjacent, D-142) |
| Upstream Watch+Listen | PARTIAL | 13.3.4 item 1 — real but narrow (reset-debris/pre-group-credit evidence only), no full Structured RAW Understanding Map producer |
| Downstream Watch+Listen | PARTIAL | `perceptual_watch_listen.py` v1 — 4 of 8 capabilities EVALUATED, 4 `NOT_IMPLEMENTED` (Section 3, unchanged) |

### 13.21 What remains partial / unimplemented (summary, not a task list)

Unified multimodal fusion (the Structured RAW Understanding Map, 13.3.1);
real semantic/prosodic audio understanding (13.3.3); position-aware
ENTRY/DELIVERY/EXIT performance localization (D-107 Section 9); explicit
typed attempt relationships beyond restart-evidence kinds (Section 3's
existing Gap); parallel perception orchestration (13.16); the inter-
complete-window agreement check D-147 proved necessary (13.8.1, D-149
below); the bounded multimodal fallback arbiter (10.3.1, still not
implemented); and all of Human Gold (Layers 10-16, Level 2, Section
13.15). None of these are authorized for implementation by this section.

### 13.22 Exact next engine sequence (NOT authorized here)

**D-149 — Semantic Authority Phase A.2 Observability.** Add detection for
COMPLETE-WINDOW vs COMPLETE-WINDOW disagreement (13.8.1), because D-147
proved the existing `partial_window_conflict` observability (D-146 Phase
A) misses it by construction. Then, Phase B's semantic authority gate
must require more than `family_complete_context=true` — it must also
account for inter-complete-window disagreement/conflict (13.8). Neither
is implemented here.

Separately, before Family Formation is declared architecturally complete,
the missing unified upstream Watch+Listen Multimodal Understanding
capability (13.3.1's Structured RAW Understanding Map) must be
implemented and qualified — not implemented or scheduled by this section.

---

No change to Section 2's 20 layers, Section 3's status table, Section 9,
Section 10, Section 11, Section 12, or any accepted D-096/D-097.x/D-107/
D-108/D-109/D-110/D-111/D-128/D-129/D-141 through D-147 authority
contract. See `docs/CUTSELL_DECISIONS.md` D-148 for the decision-log
entry recording this section's doctrine.

---

## 14. Language / Transcript Spine — Canonical Design (D-165)

**Status: additive doctrine, docs/design/forensic only. No engine
behavior change, no RAW, no provider call, no BestTake/Family/Pacing
change.** No layer is renumbered, no Section 3 status changes, no
`cutsell_worker/*.py` file was touched to write this section.

**Numbering note (source-of-truth reconciliation, binding):** the task
that produced this section referred to itself as "D-164"; by the time it
began, `docs/CUTSELL_DECISIONS.md` already carried a D-164 entry (Watch+
Listen BestTake Evidence real-media qualification, committed `07ff11e`,
strictly earlier in the same engineering session). Per this document's
own precedence rule (live Git/live docs state over a task's stated
expectation), this section and its decision-log entry are recorded as
**D-165**, not D-164. D-164's own entry is untouched.

### 14.1 Core principle

**THE TRANSCRIPT IS THE LINGUISTIC SPINE. WATCH+LISTEN IS THE MULTIMODAL
CORROBORATION/UNDERSTANDING LAYER. Neither replaces the other.** This
section canonizes the LANGUAGE side of the architecture Section 13
already named for the PERCEPTUAL side, using the exact same authority
principle (13.3.2): **PERCEPTION PROPOSES EVIDENCE. STRUCTURED EDITORIAL
AUTHORITIES DECIDE.** The Language Spine is Track A (13.2) organized into
an explicit typed hierarchy instead of the ASR transcript being read
ad hoc by dozens of independent consumers (14.9).

Updated conceptual pipeline (restates, does not renumber, Section 13.1):

```
RAW
  v
PARALLEL MULTIMODAL PERCEPTION (Section 13.2, four tracks)
  v
LANGUAGE / TRANSCRIPT SPINE  +  PERCEPTUAL SPINE      <- this section
  v                              (Section 13.3)          names the left
WATCH+LISTEN MULTIMODAL UNDERSTANDING / FUSION            branch
  v
STRUCTURED EDITORIAL REASONING
  (Proposition -> Attempt Relationships -> Family Formation
   -> BestTake -> Boundary -> Pacing -> Renderer, unchanged)
```

### 14.2 Canonical language hierarchy

```
WORD -> PHRASE -> SENTENCE/UTTERANCE -> ATTEMPT -> PROPOSITION
     -> RELATION -> FAMILY -> SELECTED REALIZATION
```

Every level below is a **target conceptual contract**, in the same sense
Section 13.3.1's "Structured RAW Understanding Map" is conceptual — no
schema is implemented by naming it. Each level's real, already-existing
code counterpart is named explicitly (14.10's inventory) so this is never
mistaken for "nothing exists yet."

- **`LanguageWord`** — text, start, end, ASR confidence, speaker (if
  available), punctuation association, `source_asset_id`. Real
  counterpart today: `contracts.Word` (text/start/end/confidence — no
  speaker/punctuation field yet) plus `canonical_asr_evidence.py`'s
  per-word normalization/fingerprinting. Word timestamps remain the base
  language time index, reusing the exact same source-relative timeline
  D-155/D-156 already proved (no duplicate clock — 14.15).
- **`LanguagePhrase`** — a bounded sub-utterance span (pause/punctuation/
  syntax/timing bounded), tagged `partial_thought` / `restart` /
  `filler_segment` / `clause_boundary` / `continuation_boundary`. **No
  dedicated type exists today** — `take_segmentation.py`'s
  `_speech_units` (gap-based splitting, `split_gap_sec=0.75`) is the one
  real, narrower instance of phrase-like segmentation, and it is
  immediately consumed and discarded inside `segment_takes` rather than
  surfaced as its own reusable object. This is the single largest
  concrete gap in the hierarchy (14.11).
- **`LanguageUtterance`** — a meaningful spoken unit, distinguishing
  grammatical sentence completion from *editorial* utterance completion
  (ends on restart/abandonment/new-beat/meaning-completion even with
  imperfect punctuation). Real counterpart: `take_segmentation._ends_
  sentence`/`_looks_complete_idea`/`_grammatically_open_tail`/`_trails_
  off`, whose combined verdict becomes `CandidateTake.complete_idea`
  (`contracts.py`). This IS a genuine editorial-utterance judgment
  already, just expressed as one boolean on the take object rather than
  a distinct typed layer.
- **`LanguageAttempt`** — one attempt to communicate an editorial idea or
  part of one; states `PRE_TAKE_SETUP`/`FALSE_START`/`ABANDONED_ATTEMPT`/
  `CLEAN_ATTEMPT`/`CORRECTION`/`CONTINUATION`/`POST_TAKE_RESET`/
  `RECORDING_PROCESS` (Section 13.4's Behavior State vocabulary, restated
  here for the language side). Real counterpart: `attempt_
  reconstruction.py`'s `reconstruct_delivery_attempts`/`_merge_attempt`
  (mints `attempt_id`, the CANONICAL semantic identity per
  `canonical_identity.py`'s ID ownership table — content/membership-
  anchored, never timestamp-anchored) plus `raw_understanding_map.py`'s
  `BehaviorHypothesis` (D-155, 8 allowed labels, provenance-tagged). The
  Language Spine supplies the textual/timing evidence into this
  judgment; Watch+Listen (Section 13.3) supplies the behavioral/
  performance evidence — exactly the fusion split this task's directive
  named, already real today at the evidence level, not yet unified under
  one typed attempt object (Section 2 Layer 3's still-target "Canonical
  Multimodal Attempt Evidence").
- **`PropositionCandidate`** — the claim/informational unit/editorial job
  an attempt represents, per D-111's binding invariant restated in
  Section 13.5: **PROPOSITION IDENTITY PRECEDES RETRY IDENTITY**; same
  topic/product/opener/sentence-structure is NEVER sufficient alone.
  Real counterpart: `semantic_idea_equivalence.py`'s bounded arbiter +
  `editorial_slot_resolution_install.py`'s `_SEMANTIC_EQUIVALENCE_
  POLICY`/`_SLOT_RULES` (the actual "same intended idea is an editorial-
  function question, not a union-of-facts test" rule the provider prompt
  already encodes) + `semantic_claims.py`'s clause-level `Claim` objects
  (a finer-grained factual unit than a proposition, reused for coverage
  protection, not proposition identity itself). **Known, already-
  documented gap:** `canonical_identity.py` mints `semantic_idea_id` and
  `retry_family_id` from the SAME group key today — PROPOSITION identity
  and FAMILY identity are conflated in the real architecture (D-050
  Phase 3's own finding, explicitly deferred to D-050B/C, still deferred
  here).
- **`RelationEvidence`** — canonical relations `RETRY`/`CORRECTION`/
  `CONTINUATION`/`COMPLEMENTARY`/`NEW_AUDIENCE_BEAT`/
  `DISTINCT_PROPOSITION`/`UNCERTAIN` (D-145's 5-way outcome plus D-158's
  `UNCERTAIN`, Section 13.6, unchanged). Real counterpart: this is the
  **most mature, most fully implemented layer already** —
  `attempt_relationship_authority.py`'s `resolve_final_attempt_relation`
  (the full agree/conflict/fail-open truth table between the pre-existing
  semantic/deterministic decision and real Watch+Listen relation
  evidence) plus `watch_listen_relation_discovery.py`'s own discovery
  half, both RAW-proven (D-158/D-161/D-162). `UNCERTAIN` is already a
  first-class, never-forced outcome exactly as this task requires.
- **`FAMILY`** — a set of legitimate competing realizations of the same
  proposition/editorial job. Language Spine helps FORM the candidate set;
  Watch+Listen and structured authority VALIDATE it; transcript alone is
  never the sole family authority (Section 13.7, restated, unchanged).
  Real counterpart: `take_grouping.py`/`take_grouping_provider.py`/
  `hybrid_session_cleanup.py` (unchanged, D-158/D-161's own evidence-
  widening doctrine already governs this — Section 13.7 is not modified
  by this section).
- **`SELECTED REALIZATION`** — BestTake's eventual winner among competing
  realizations. Language Spine contributes meaning completeness,
  coverage, duplication, semantic equivalence, factual consistency, slot
  relevance; Perceptual Spine contributes delivery usability, fumble,
  breaking character, entry/delivery/exit quality, performance
  continuity (exactly D-163's own scope, Section 14.16 — unchanged,
  unmodified by this section).

### 14.3 Transcript normalization contract

Audit result: real, already-implemented canonical normalization exists
in `canonical_asr_evidence.py` (`_normalize_word_text`, `canonicalize_
transcript_words`, `compute_canonical_equivalence_hash`, `normalize_
transcript_segments`) and `take_segmentation.py`'s completion heuristics
— case/punctuation/whitespace normalization is real and deterministic.
**Never normalized away today, correctly:** negation, numbers, diagnosis/
factual terms (protected by `semantic_claims.py`'s clause-level claim
extraction and `polarity_safety.py`'s explicit polarity/negation guard —
these operate on the RAW clause text, downstream of normalization, never
letting canonicalization erase a meaning-changing word). Filler tokens,
hesitation markers, and partial words are NOT currently canonicalized
into one shared normalized-word vocabulary — each of the ~58 downstream
consumers named in 14.9 does its own ad hoc `.lower()`/tokenization,
which is exactly the fragmentation this task's directive anticipated.
Named entities have no dedicated canonical treatment today (out of
current scope; not required for Milestone 1 per 14.13).

### 14.4 Dead-air / silence integration

Confirmed: pause/attempt-boundary reasoning already uses REAL measured
audio-silence evidence, not transcript-gap inference alone — `attempt_
reconstruction.py`'s `_measured_pause_at_transition` (D-097.5/.6's own
measured dead-air pause boundary fix, RAW-proven) and `audio_silence.py`/
`silence_analysis.py`/`audio_boundary_completion.py` (Section 13.2 Track
B). Not every pause is treated as a new attempt — `_tiny_nonterminal_
continuation` and `_short_incomplete_suffix` (`attempt_reconstruction.py`)
already exist specifically to prevent that overreach. This capability
already satisfies the directive's own requirement; nothing new is
proposed here.

### 14.5 Filler / fumble contract

**Lexical filler** (um/uh/like) has no single canonical detector today —
it is handled inconsistently, if at all, inside individual consumers'
own text heuristics (14.9). **Editorial fumble** (restart/self-
correction/abandoned sentence/wrong word/repeated phrase) IS well
covered, but by BEHAVIOR-level modules, not a language-level fumble
type: `lexical_self_correction.py`, `internal_self_correction.py`,
`frustrated_restart.py`, `micro_restart_cleanup.py`,
`trailing_retry_restart.py` each independently detect a slice of this
space via their own text pattern matching — a duplicate-consumer
instance of the exact fumble/filler distinction this task asks for,
never unified under one shared typed evidence layer.

### 14.6 Retry / continuation / correction / complementary language evidence

All four relations are ALREADY implemented with real evidence beyond
"same opener/topic/keyword alone" (which is explicitly and correctly
rejected — Section 13.5/13.6, `_SEMANTIC_EQUIVALENCE_POLICY`):

- **Retry** — `same_opening_abandoned_start`/`incomplete_attempt_
  completed_by_retry` deterministic restart evidence
  (`take_grouping_provider.py`/`semantic_idea_equivalence.py`, proven
  live this session's own D-164 RAW: `docs/CUTSELL_DECISIONS.md` D-164
  Section "GLOBAL D-163 SUMMARY" merges list) plus D-100's `wrong_take`/
  `retry_setup` corroboration.
- **Continuation** — `_tiny_nonterminal_continuation` (language/timing
  evidence) + `recording_meta_continuation.py` (behavior evidence);
  incomplete-A + completing-B is already represented WITHOUT forcing
  retry competition, per this task's own requirement.
- **Correction** — `RELATION_CORRECTION` (D-145's 5-way outcome) +
  `semantic_claims.py`'s `negation_role` field (`FACTUAL_NEGATION` vs
  `CONTRASTIVE_HINDSIGHT_NEGATION`, D-066) — before/after meaning is
  preserved via the claim's own `text`/`content_tokens`, never flattened
  into duplicate takes.
- **Complementary** — `RELATION_COMPLEMENTARY` (D-145) +
  `hybrid_complementary_delivery_guard.py`/`hybrid_semantic_
  complementary_rescue.py`/`post_selection_complementary_family_
  stabilizer.py` — distinct useful information under one slot is kept out
  of retry competition, per this task's own requirement, though (14.9)
  by three separately-evolved modules rather than one shared authority.

### 14.7 Conclusion structure

**No explicit typed conclusion-slot evidence exists today.** The real,
working equivalent is `realization_resolver.py`'s `UNIQUE_CONCLUSION`
marker family (`claim_coverage_best_take.py`'s `_UNIQUE_CONCLUSION_
MARKERS`, `_clause_has_any`) plus `editorial_slot_resolution_install.py`'s
prompt rule: *"A later conclusion/restatement after an already complete
conclusion is normally a competing realization of the CONCLUSION slot
unless it advances the story with a genuinely different required
proposition."* This rule is real, RAW-proven (it governs the same-family
grouping this session's own D-164 RAW exercised), but it lives as
free-text guidance INSIDE a provider prompt string, never as a structured
field any code reads or writes. The directive's target shape (setup /
main conclusion statement / supporting continuation / closing line / CTA
transition, generalized, never Video00-hardcoded) is not implemented.

### 14.8 CTA structure

Partial, mixed evidence. `contracts.SemanticRole.CTA` is a REAL typed
enum value, wired live into `pipeline.build_flow_b_draft`'s `DraftClip.
role` field (via `flow_b.py`'s `semantic_labels = semantic.labels`) — but
this is the LEGACY composer/Sales-funnel semantic layer (`semantic_
openai.py`/`composer.py`), not Clean Cut Core V1's active idea-first
path; per CLAUDE.md's own binding rule ("Do not force rigid sales-funnel
logic during Clean Cut"), `DraftClip.role` defaults to `SemanticRole.
OTHER` for essentially every clip on the active path today (confirmed:
no call site in `pipeline.py`'s Clean Cut V1 flow populates real HOOK/
CTA labels from a live classifier). The only LIVE CTA-adjacent evidence
in the active path is textual, inside `realization_resolver.py`'s own
CTA-ordering comment/logic (`comprehensive` closing-CTA anchor handling,
`round9_orphan_prefix_integrity.py`-adjacent) and the prompt-level rule
already covered in 14.7. **No CTA candidate/pre-CTA-setup/duplicate-CTA/
partial-CTA/complete-CTA typed vocabulary exists**, and none is
implemented here — this is a real, named gap for a future task.

### 14.9 Editorial slot evidence

`contracts.SemanticRole` (`HOOK`/`PROBLEM`/`FEATURES`/`BENEFITS`/`PROOF`/
`STORY`/`CTA`/`OTHER`) is a real, typed 8-value enum, already wired end-
to-end into `DraftClip.role` — but it is a DORMANT/legacy field on the
active Clean Cut V1 path (14.8), not live editorial-context evidence for
today's Selection/BestTake/Family decisions. The directive's proposed
minimum slot abstraction (HOOK/SETUP/PROBLEM/FEATURE/PROOF/CONCLUSION/
CTA/OTHER) is therefore NOT a new vocabulary to invent — it is
`SemanticRole` itself, minimally extended (SETUP/CONCLUSION are the only
two missing values) and RE-ACTIVATED as evidence/context only, never a
forced funnel (this task's own explicit constraint, and CLAUDE.md's own
binding rule, both satisfied by treating it as advisory context, never a
membership decision). Not implemented here.

### 14.10 Confidence / ambiguity and language provenance

Confidence is already categorical/evidence-based, never one arbitrary
weighted master score, at every real layer inspected: `Word.confidence`
(ASR, optional float), `AttemptRelationHypothesis.confidence`
(`CONFIDENCE_SUPPORTED`/`WEAK`/`UNKNOWN`, `watch_listen_understanding.
py`), `semantic_idea_equivalence.py`'s per-merge `confidence` field
(reported, not weighted into a composite), and D-145's discrete 5-way
relation outcome. This already satisfies the directive's own "no
arbitrary weighted master score" requirement — nothing new proposed.

Provenance vocabulary already exists and is directly reusable:
`raw_understanding_map.py`'s `PROVENANCE_VISUAL_SIGNAL`/
`PROVENANCE_DETERMINISTIC_RULE`/`PROVENANCE_MULTIMODAL_FUSION`/
`PROVENANCE_UNKNOWN` (no `PROVENANCE_AUDIO_SIGNAL` producer exists for
behavior hypotheses today — the same Audio Honesty finding D-163
documented). A future V1 Language Spine should ADD `ASR`,
`TRANSCRIPT_NORMALIZATION`, `WORD_TIMING`, `PHRASE_SEGMENTATION`,
`SEMANTIC_PROVIDER`, `DETERMINISTIC_LANGUAGE_RULE`,
`WATCH_LISTEN_CORROBORATION` to this SAME shared vocabulary rather than
inventing a parallel one — not implemented here.

### 14.11 Current code inventory (factual matrix)

| Capability | Module(s) | Current output | Consumers | Status | Duplication | Missing piece |
|---|---|---|---|---|---|---|
| ASR | `asr.py` (`FasterWhisperASR`) | `Word` list w/ confidence | `canonical_asr_evidence.py`, `take_segmentation.py` | EXISTING | none | speaker diarization |
| Word timing | `contracts.Word`, `canonical_asr_evidence.py` | start/end/confidence per word | nearly every downstream module | EXISTING | none (single source) | punctuation association per word |
| Transcript normalization | `canonical_asr_evidence.py` | canonical words, content hash, equivalence hash | fingerprinting/dedup only today | EXISTING | none at this layer | not reused by the ~58 modules in 14.9's own text-comparison helpers |
| Phrase segmentation | `take_segmentation._speech_units` | gap-split sub-spans (internal only) | consumed then discarded by `segment_takes` | PARTIALLY IMPLEMENTED | n/a | no reusable `LanguagePhrase` type |
| Sentence/utterance segmentation | `take_segmentation._looks_complete_idea`/`_ends_sentence`/`_grammatically_open_tail` | `CandidateTake.complete_idea` (bool) | `attempt_reconstruction.py`, judge/resolver chain | EXISTING | none | editorial-utterance boundary is a single bool, not a typed span |
| `complete_idea` | `take_segmentation.py` | boolean on `CandidateTake` | Resolver/BestTake usability gates | EXISTING | none | — |
| Attempt representation | `attempt_reconstruction.py` (`reconstruct_delivery_attempts`, `_merge_attempt`), `raw_understanding_map.py` (`BehaviorHypothesis`) | fused `CandidateTake` w/ `attempt_id`; 8-label behavior hypothesis | pre-group credit, `take_judge.py`, Resolver | EXISTING | 2 parallel representations (attempt_id vs BehaviorHypothesis) not yet unified | Layer 3's still-target "Canonical Multimodal Attempt Evidence" |
| Proposition representation | `semantic_idea_equivalence.py`, `editorial_slot_resolution_install.py` (prompt policy) | merge/no-merge decision + confidence + reason | `take_grouping_provider.py` | EXISTING | conflated with family id (`retry_family_id`, see 14.2) | typed `PropositionCandidate` object |
| Semantic similarity | ~58 modules, each own tokenizer (14.9) | ad hoc boolean/float | scattered | EXISTING, HEAVILY FRAGMENTED | **~58 independent implementations** | one canonical normalized-token/overlap function |
| Retry evidence | `take_grouping_provider.py` restart-evidence rules, D-100 corroboration | merge decision + `accepted_by` reason | Family Formation | EXISTING | none | — |
| Continuation evidence | `attempt_reconstruction._tiny_nonterminal_continuation`, `recording_meta_continuation.py` | boolean/candidate | AttemptReconstructor | EXISTING | 2 modules, same concept | unified typed relation |
| Correction evidence | D-145 `RELATION_CORRECTION`, `semantic_claims.negation_role` | relation label / claim field | `attempt_relationship_authority.py` | EXISTING | none | — |
| Complementary evidence | D-145 `RELATION_COMPLEMENTARY` + 3 `hybrid_*complementary*`/`post_selection_complementary_family_stabilizer.py` modules | relation label / guard decisions | Family/Resolver | EXISTING | **3 separately-evolved modules** | consolidation |
| Conclusion evidence | `realization_resolver.UNIQUE_CONCLUSION`, prompt text only | marker match / prompt rule | Resolver, provider prompt | PARTIALLY IMPLEMENTED | n/a | typed conclusion-slot object |
| CTA evidence | `contracts.SemanticRole.CTA` (dormant on active path), prompt text | enum value (mostly `OTHER` in practice) | legacy composer path only | PARTIALLY IMPLEMENTED / DORMANT | n/a | typed, live CTA evidence on the active path |
| Slot evidence | `contracts.SemanticRole` (8 values, 2 missing: SETUP/CONCLUSION) | `DraftClip.role` | legacy composer path only | PARTIALLY IMPLEMENTED / DORMANT | n/a | re-activation as advisory context on the active path |
| Family Formation | `take_grouping.py`/`take_grouping_provider.py`/`hybrid_session_cleanup.py` | `TakeGroup` | BestTake | EXISTING | none (single authority, D-158/D-161 already widen its evidence intake) | — |
| BestTake language evidence | `take_judge.score_take` (`completeness` term), `claim_coverage_best_take.py` | scalar score component / coverage boolean | `deterministic_best_take_authority.py` | EXISTING | none | — |
| `RawUnderstandingMap` | `raw_understanding_map.py` (D-155, CLOSED) | per-source behavior-hypothesis map | D-163's `watch_listen_besttake_evidence.py` | EXISTING | none | — |
| `WatchListenUnderstanding` | `watch_listen_understanding.py` (D-157, CLOSED) | per-span entry/delivery/exit usability | D-163 guard, D-158/D-161 relation authority | EXISTING | none | — |

### 14.12 Duplicate transcript consumers (identified, not fixed)

Direct code inspection (not estimated) found **58 modules** under
`cutsell_worker/` independently tokenizing/normalizing `CandidateTake.
text` for their own content-overlap/similarity comparison (`grep -l
"_content_tokens\|_tokens(text)\|\.lower()\.split()\|content_tokens"`),
and **3 modules** independently reimplementing sentence/clause splitting
(`semantic_claims.py`, `take_segmentation.py`, `incomplete_unique_
bridge_completion_rescue.py`). This is the single largest concrete
fragmentation finding in this audit — most of `cutsell_worker/`'s
`hybrid_*`/`round*`/`final_*`/`post_selection_*` retry-integrity family
of modules (visible in the file listing: `hybrid_alternate_integrity.py`,
`hybrid_cross_group_retry_integrity.py`, `final_draft_retry_integrity.py`,
`round8_retry_reconciliation.py`, `round9_orphan_prefix_integrity.py`,
`round11_semantic_retry_cleanup.py`, and ~20 more of the same shape) each
reimplement a small, slightly different content-overlap heuristic rather
than consuming one canonical Language Spine tokenizer/normalizer. **What
should be computed once and reused:** canonical word normalization
(already centralized in `canonical_asr_evidence.py`, just not reused by
these 58 consumers), a canonical content-token/overlap function, and a
canonical phrase/clause boundary detector. No implementation is proposed
in this task.

### 14.13 Implementation gap — smallest truthful conclusion

**B. LANGUAGE SPINE PARTIALLY EXISTS; TYPED HIERARCHY + NORMALIZATION
MUST BE BUILT** — with a significant "fragmentation" (A-shaped)
component that is the dominant migration cost, not a missing-capability
(C/D-shaped) one. Every underlying EVIDENCE capability the hierarchy
needs already exists and is real, working, RAW-proven code: ASR + word
timing (`asr.py`), transcript normalization/fingerprinting
(`canonical_asr_evidence.py`), editorial-utterance completion
(`take_segmentation.py`), attempt reconstruction with canonical
semantic identity (`attempt_reconstruction.py`, `canonical_identity.py`),
a fully-implemented 5-way relation vocabulary with a real conflict/fail-
open truth table (`attempt_relationship_authority.py`, D-145/D-158/
D-161/D-162), and clause-level claim extraction (`semantic_claims.py`).
This rules out C (segmentation is not "too weak" — it is real and
proven) and D (no major ASR/semantic capability is missing). What is
genuinely missing is (1) a PHRASE-level typed object (today only an
internal, discarded heuristic), (2) ONE canonical normalized-token/
overlap function shared by the ~58 modules that each reimplement their
own today (14.12), (3) a de-conflation of `semantic_idea_id`/
`retry_family_id` (already a documented D-050 gap), and (4) re-
activating `SemanticRole`/conclusion evidence as live, structured
context on the active Clean Cut V1 path rather than dormant legacy
fields or prompt-only text. This is more work than pure consolidation
(A) but far less than rebuilding segmentation (C) or adding a new
language-model capability (D).

### 14.14 V1 Language Spine contract (design only, NOT implemented)

Smallest implementable V1, matching this task's own suggested shape and
the existing repo-conventional dataclass style (`contracts.py`,
`@dataclass(frozen=True)`):

```
LanguageWord(text, start, end, confidence, source_asset_id)
LanguagePhrase(source_asset_id, start, end, text, words,
               boundary_kind)  # partial_thought/restart/filler/
                                # clause_boundary/continuation_boundary
LanguageUtterance(source_asset_id, start, end, text, phrases,
                  complete_idea, provenance)
LanguageAttempt(attempt_id, source_asset_id, start, end, utterances,
                behavior_state, confidence, provenance)
PropositionCandidate(proposition_id, attempt_ids, content_tokens,
                     confidence, provenance)
RelationEvidence(relation, left_id, right_id, confidence, source,
                 provenance)  # reuses attempt_relationship_authority.py's
                              # existing FinalAttemptRelationship shape
```

`FAMILY` and `SELECTED REALIZATION` authority are explicitly OUT of V1's
scope (this task's own instruction) — V1 only produces evidence these
existing, unchanged authorities may consume later, in a separately-
authorized task.

### 14.15 Time index

All language objects reuse the SAME source-relative timeline already
proven in D-155/D-156 (`source_asset_id` + `start`/`end` seconds) — no
duplicate clock is proposed. This is already true of every real module
inspected in 14.11 (`Word.start/end`, `CandidateTake.start/end`,
`UnderstandingSpan`'s own span timing) and would remain true of any V1
`LanguageWord`/`LanguagePhrase`/etc.

### 14.16 Non-destructive result

The Language Spine, as designed, only INDEXES RAW — no physical deletion
occurs at this layer (identical to every existing perception/
understanding layer in Sections 2/13). Every language node maps back to
a recoverable RAW range via `source_asset_id` + `start`/`end`, exactly
as `CandidateTake`/`UnderstandingSpan` already do.

### 14.17 Parallelism role

Restates Section 13.2's existing finding: Language Spine construction
(Track A) has no canonical requirement to finish before visual/audio
perception (Tracks B/C) — they already run independently over the same
RAW today (D-155's `parallel_perception.py`, `CUTSELL_PARALLEL_
PERCEPTION_ENABLED`, default ON). A future V1 Language Spine should
preserve this, not introduce a new blocking dependency.

### 14.18 Cut.ai parity role (Milestone 1)

The Language Spine's contribution to Level 1 (Section 5) is entirely
things this audit found ALREADY REAL: clean retry identification
(attempt reconstruction + D-145 relations), dead-air/filler structure
(measured-pause boundary), complete-idea preservation (`complete_idea`),
conclusion integrity (`UNIQUE_CONCLUSION` marker family), duplicate
removal (Family Formation + claim dedup), meaning safety (`polarity_
safety.py`, `semantic_claims.py`'s negation protection). CTA integrity
and commercial beat structure are the two genuinely weaker areas (14.7/
14.8) — a future, separately-authorized task's most defensible target.

### 14.19 Human Gold role (Milestone 2, restates Section 13.15, unchanged)

A richer Language Spine may later serve minimal-sufficient-set reasoning,
narrative compression, preferred-realization ranking, stronger
conclusions, advanced composites, and editorial taste (Layers 10-16) —
explicitly NOT prioritized ahead of Cut.ai parity, per Section 5 and this
task's own instruction.

### 14.20 D-163 compatibility (binding, unmodified)

D-163's Watch+Listen BestTake performance/usability evidence
(`watch_listen_besttake_evidence.py`) remains fully valid and untouched.
Language Spine strengthens meaning/proposition/relation/coverage; D-163
strengthens performance/usability — complementary axes of the same
`SELECTED REALIZATION` decision (14.2's final layer), never overlapping,
never modified by this section.

### 14.21 Phased build plan (NOT authorized here)

A. Canonical `LanguageWord`/`LanguagePhrase` typed schema + one shared
   normalization function (replaces the ~58 ad hoc tokenizers, 14.12).
B. `LanguageUtterance`/`LanguageAttempt` construction (formalizes
   `complete_idea` + `attempt_reconstruction.py`'s existing logic into
   the typed hierarchy, no behavior change).
C. `PropositionCandidate`/`RelationEvidence` integration (formalizes
   `semantic_idea_equivalence.py` + `attempt_relationship_authority.py`
   output into the typed hierarchy; also the natural point to resolve
   the `semantic_idea_id`/`retry_family_id` conflation, D-050B/C).
D. Replace duplicate transcript readers (14.12's 58 modules) with Spine
   consumers, one module at a time, behavior-parity-tested each time
   (same discipline as D-050A's own additive-shadow migration).
E. ONE Video00 qualification (after A-D are offline-proven).
F. Unseen-RAW generalization (CleanCutBench expansion).
G. Human Gold refinement (14.19, explicitly deferred).

Smaller phases may be chosen if the actual code, once touched, suggests
better seams — this order is a starting proposal, not a commitment.

### 14.22 Architecture verdict

**B. LANGUAGE SPINE PARTIALLY EXISTS; TYPED HIERARCHY + NORMALIZATION
MUST BE BUILT** (14.13).

---

No change to Section 2's 20 layers, Section 3's status table, Sections
4-13, or any accepted D-096 through D-164 authority contract. Family
Formation, Proposition Identity, Attempt Relationships, BestTake,
Boundary, and Pacing are all restated, never modified, by this section.
See `docs/CUTSELL_DECISIONS.md` D-165 for the decision-log entry
recording this section's doctrine.

---

## 15. POST D-177 CANONICAL CONSOLIDATION (D-178A)

**Status: additive doctrine, documentation only. No engine behavior
change, no RAW, no provider call, no BestTake/Family/Boundary/Pacing/
Renderer authority change, no feature flag, no `cutsell_worker/*.py` file
touched to write this section.** No layer is renumbered, no Section 3
status changes, no accepted D-096 through D-177 authority contract is
weakened or overridden. This section incorporates the current PROVEN
engine state (through D-177) and canonizes the newly-approved forward
architecture the Product Owner named in D-178A, reading as ONE continuous
architecture with Sections 1-14, never a second system.

### 15.1 North Star (canonical, restates and extends Section 5's Milestone
progression)

```
RAW
  v
CLEAN RAW
  v
CUT.AI PARITY               (Milestone 1, Section 5, unchanged)
  v
HUMAN GOLD REFINEMENT        (Milestone 2, Section 5/13.15, unchanged, not started)
  v
COMMERCIAL / SALES-FUNNEL INTELLIGENCE   (NEW post-parity milestone, 15.13-15.14)
```

This restates Section 5's existing two-milestone progression and adds
ONE new named downstream milestone after Human Gold. No milestone is
reordered; Commercial/Sales-Funnel Intelligence remains explicitly
POST-PARITY and POST-HUMAN-GOLD, never authorized to run ahead of either.

### 15.2 Current canonical foundations (consolidated restatement, unchanged)

The following are PRESERVED, EXPLICITLY INTEGRATED, and not modified by
this section — each already has its own canonical home in Sections
2/9-14:

- **Language Spine** (`LanguageWord` -> `LanguagePhrase` ->
  `LanguageUtterance` -> `LanguageAttempt` -> `PropositionCandidate` ->
  `RelationEvidence`, Section 14.2's canonical hierarchy) — **CURRENT
  STATE (corrected D-178A.1): LANGUAGE SPINE FOUNDATION: OFFLINE_PROVEN.**
  D-165's own verdict ("B. LANGUAGE SPINE PARTIALLY EXISTS; TYPED
  HIERARCHY + NORMALIZATION MUST BE BUILT," Section 14.13/14.22) was the
  correct finding AT THE TIME D-165 was written — a forensic/design-only
  task with zero implementation. It is preserved below (15.2.1) as
  HISTORICAL PRE-IMPLEMENTATION STATE, not the current state. D-166
  (`language_spine.py`, `LanguageWord`+`LanguagePhrase`, OFFLINE PROVEN),
  D-168 (`language_utterance_attempt.py`, `LanguageUtterance`+
  `LanguageAttempt`, OFFLINE PROVEN), and D-169
  (`language_proposition_relation.py`, `PropositionCandidate`+
  `RelationEvidence`, OFFLINE PROVEN — proposition identity vs
  retry-family identity de-conflated at the type-system level for the
  first time) together IMPLEMENT the full six-rung hierarchy D-165 only
  designed. D-171 then began INCREMENTAL, bounded PRODUCTION consumer
  migration onto this foundation (`language_spine_consumer_migration.py`):
  exactly 2 of 3 candidate consumer clusters migrated (proposition/retry
  divergence evidence in `take_grouping_provider.py`; continuation
  evidence in `recording_meta_continuation.py`), fail-open by
  construction (Spine substitutes for legacy ONLY where a per-call
  comparison proves the two verdicts IDENTICAL; any disagreement or
  missing evidence falls back to the pre-existing legacy path
  unchanged), the third candidate correctly skipped as already
  non-duplicative. **This does NOT mean all transcript consumers have
  migrated** — Section 14.12's own audit found ~58 independent ad hoc
  tokenizers across `cutsell_worker/`; D-171 migrated 2. The remaining
  ~56 are unchanged, still reading `CandidateTake.text` directly, exactly
  as Section 14.12 originally found. No Family/BestTake/Proposition/
  final-Relation/Boundary/Pacing authority was changed by D-166/D-168/
  D-169/D-171 (see each entry's own decision-log record); Family
  Formation/BestTake/Boundary/Pacing/Renderer remain governed exactly as
  Sections 13/15.2 already state.

#### 15.2.1 D-165 historical status (preserved, not current)

For historical record only — **NOT the current engine state** (see the
corrected bullet above): at the time D-165 was authored (forensic/design
task, zero implementation), the accurate finding was **B. LANGUAGE SPINE
PARTIALLY EXISTS; TYPED HIERARCHY + NORMALIZATION MUST BE BUILT**
(Section 14.13/14.22, text unchanged, still visible verbatim at those
locations). D-166 through D-171 are the separately-authorized
implementation tasks that closed most of that gap; Section 14's own text
is left untouched below (D-098's own precedent, e.g. the D-164/D-165
numbering-reconciliation note) precisely so this document never erases
its own history — only Section 15's restatement of "current state" is
corrected here.
- **Parallel perception** (Speech/Language, Visual/Performance, Audio,
  Media/Timing — Section 13.2's four tracks A-D), unchanged.
- **Watch+Listen Multimodal Understanding** (upstream, Section 13.3;
  distinct from downstream Watch+Listen QA, Section 13.3.4/4), unchanged.
- **Behavior State** (Section 13.4/D-111 10.1 vocabulary), unchanged.
- **Proposition Identity precedes Retry Identity** (Section 13.5/D-111
  10.2, binding), unchanged.
- **Attempt Relationships** (Section 13.6/13.7, D-145's 5-way outcome +
  D-158's `UNCERTAIN`), unchanged.
- **Stable Family Formation** (Section 13.7-13.8, `take_grouping.py`
  family), unchanged.
- **BestTake** (Section 13.10/D-107 Section 9 priority order), unchanged.
- **Watch+Listen BestTake Guard** (D-174's
  `watch_listen_besttake_guard_authority.py` — MAY VETO, NEVER SELECTS;
  CLOSED, D-175 real-media-proven safe/discriminating), unchanged.
- **Freeze** (Selection Freeze, Layer 5/6 boundary), unchanged.
- **Boundary** (`boundary_engine_pass.py`, Section 13.11/D-107 CASE A/B/C
  split), unchanged in authority; its CURRENT STATE is updated at 15.9
  below to record D-177.
- **Dialogue/Pacing** (Section 13.12/D-129 Section 11), unchanged.
- **Renderer** (Layer 7 execution, `render_plan.py`), unchanged.

### 15.3 Audio architecture: V1 (current) vs V2 (future prosodic
understanding)

Separates, for the first time as an explicit named split, the existing
Track B (Section 13.2) into two generations:

**AUDIO V1 — CURRENT (EXISTING, restates Section 13.2 Track B/13.3.3
unchanged).** Signal-level silence/dead-air/pause/timing evidence only:
`audio_silence.py`, `silence_analysis.py`, `audio_boundary_completion.py`,
`attempt_reconstruction.py`'s measured-pause boundary (D-097.5/.6). Real,
waveform-derived, genuinely computed — not a stub, but explicitly NOT
semantic or prosodic understanding (Section 13.3.3's "Real audio honesty"
finding, restated, unchanged).

**AUDIO V2 — FUTURE PROSODIC UNDERSTANDING (MISSING/FUTURE, named here
for the first time as its own forward contract; restates Section 13.2
Track B's existing `DESIGNED_NOT_IMPLEMENTED` status, not a new gap).**
Future observable evidence, named as a target vocabulary only:
`hesitation`, `vocal_restart`, `emphasis`, `cadence`, `rhythm`,
`vocal_continuity`, `flat_or_expressive_delivery`, `delivery_energy`
(reframes the existing but always-default `MediaSignals.delivery_energy`
field per D-107 Section 9's DELIVERY_ENERGY_FIT concept, unchanged).

**Binding constraint (new invariant, extends Section 10.5/13.19's
invariant list):** Audio V2 evidence describes OBSERVABLE vocal/acoustic
signal properties only. **Do not canonize psychological or emotional
inference as fact** — CutSell may observe "flat delivery" or "vocal
restart" as measured signal shape; it must never assert or encode "the
creator felt nervous/confident/frustrated" as ground truth. This is a
permanent, general invariant (Section 6/10.4 anti-rule-proliferation
class: a universal safety invariant, not a benchmark-specific rule), not
implemented by naming it.

### 15.4 New pre-parity capability: Editorial Moment & Sequence
Understanding

**Status: MISSING/FUTURE, named here for the first time. Belongs to RAW
-> Cut.ai parity (Milestone 1), NOT Human Gold, and explicitly NOT
Commercial Moment Understanding (15.13).**

**Purpose:** understand the ROLE a moment plays in the recording/editing
PROCESS itself — distinct from Behavior State (13.4, what is physically
happening at an instant) and distinct from editorial FUNCTION (Layer 10,
what job a kept realization performs for the audience). **Dependency
direction (corrected D-178A.1 — binding):** Editorial Moment & Sequence
Understanding is a HIGHER-ORDER understanding layer. It CONSUMES already-
produced evidence from Parallel Perception (13.2), Watch+Listen (13.3),
Behavior State (13.4), and the Language Spine's own structured objects
(`LanguageAttempt`, `PropositionCandidate`, `RelationEvidence` — 14.2,
15.2 corrected status) — it does not sit UPSTREAM of them, does not widen
their evidence intake, and does not run before they exist. It may form
HIGHER-ORDER HYPOTHESES about recording process, clean audience
delivery, preassembled final sequences, and sequence structure by
combining that already-produced evidence — it does **not** independently
recreate Behavior State, Attempt Identity, Proposition Identity, or
Relation Identity (those remain owned exactly where Sections 13.4-13.7/
14 already place them, unchanged). This restates and sharpens 13.3.2's
authority principle (PERCEPTION PROPOSES EVIDENCE; UNDERSTANDING FORMS
HYPOTHESES; STRUCTURED EDITORIAL AUTHORITIES DECIDE) rather than
conflicting with it — D-178A's original phrasing ("widens the evidence
those existing authorities may consume," implying an upstream position)
is the specific error this correction fixes; see 15.10's corrected stack.

At minimum, future roles include: `PRE_TAKE_SETUP`, `RECORDING_PROCESS`,
`FALSE_START`, `ABANDONED_ATTEMPT`, `RETRY`, `CORRECTION`,
`CONTINUATION`, `CLEAN_AUDIENCE_DELIVERY`, `POST_TAKE_RESET`,
`BREAKING_CHARACTER`, `NEW_AUDIENCE_BEAT`, `PREASSEMBLED_FINAL_SEQUENCE`.

Relationship to existing vocabulary (clarifies, does not rename):
`PRE_TAKE_SETUP`/`FALSE_START`/`ABANDONED_ATTEMPT`/`RECORDING_PROCESS`/
`POST_TAKE_RESET`/`BREAKING_CHARACTER`/`NEW_AUDIENCE_BEAT` already exist
as Behavior States (Section 13.4/D-111 10.1); `RETRY`/`CORRECTION`/
`CONTINUATION` already exist as Attempt Relationships (Section 13.6).
Editorial Moment & Sequence Understanding does not invent new states or
relations for these — it reframes them as evidence about the moment's
ROLE IN SEQUENCE (was this moment part of assembling the take, or part of
delivering it to the audience), which is the missing piece Section
13.3.1's Structured RAW Understanding Map already anticipated
(`entry_state`/`delivery_state`/`exit_state`) but never fully specified.
Two genuinely NEW concepts are named here for the first time:
`CLEAN_AUDIENCE_DELIVERY` (a moment that IS the intended audience-facing
content, as distinct from process around it) and
`PREASSEMBLED_FINAL_SEQUENCE` (a RAW span that is ALREADY an edited/
assembled sequence rather than a single unedited take — see 15.5's
"source takes vs already-edited material" for why this matters). Not
implemented by naming it here.

### 15.5 Whole-Video Editorial Reasoning (new global reasoning layer)

**Status: MISSING/FUTURE, named here for the first time.**

Canonizes a GLOBAL reasoning layer sitting ABOVE local Watch+Listen
evidence (13.3), above Structured Local Understanding (Behavior State/
Language Spine's `LanguageAttempt`/`PropositionCandidate`/
`RelationEvidence` — 15.10's corrected stack), and above per-span
Editorial Moment Understanding (15.4) — reasoning across the FULL RAW
rather than per-attempt or per-family in isolation. **Dependency
direction (corrected D-178A.1 — binding):** like Editorial Moment
Understanding (15.4), Whole-Video Editorial Reasoning CONSUMES the
already-produced structured evidence beneath it; it does not precede,
replace, or independently re-derive Behavior State, Attempt Identity,
Proposition Identity, or Relation Identity. This does not create a new
decision authority over Family/Realization Formation, BestTake,
Boundary, or Freeze (13.3.2's authority principle still governs: it
proposes evidence and forms hypotheses, structured authorities still
decide) — it is a wider EVIDENCE-GATHERING and HYPOTHESIS-FORMING pass
that Family/Realization Formation (15.10) may consume, analogous to how
Section 13.7 already requires Family Formation to widen its evidence
intake beyond textual/provider comparison alone.

Purpose — reason across the whole RAW about: recording-process regions;
duplicate propositions; clean completed sequences; preassembled final
sequences (15.4's new state); source takes vs. already-edited material;
global redundancy; narrative continuity. This generalizes and gives a
name to a gap Section 13.3.1's Structured RAW Understanding Map already
implied (a per-span conceptual output) but never extended to a
WHOLE-VIDEO scope — today's real, narrower precursor is
`whole_video_context.sources[].events` (`TemporalEvent`/
`SourceVideoContext`, restated from Section 13.3.1, unchanged), which
carries per-source events but no global cross-family/cross-region
reasoning pass. No implementation is authorized here.

### 15.6 BestTake Future: BestTake Multimodal Fusion V2

**Status: MISSING/FUTURE, named here for the first time, extends but does
not modify D-107 Section 9's existing priority order (13.10, unchanged).**

Canonizes a future fusion sequence for BestTake's tier-(3) MULTIMODAL
PERFORMANCE QUALITY evidence (D-107 Section 9's existing 5-tier order,
restated unchanged: (1) MEANING/MESSAGE SUFFICIENCY, (2) TAKE USABILITY,
(3) MULTIMODAL PERFORMANCE QUALITY, (4) EDITABILITY/BOUNDARY QUALITY, (5)
NARRATIVE/ENERGY FIT):

```
Meaning
  v
Visual Delivery
  v
Vocal Delivery          (consumes Audio V2 evidence, 15.3, when it exists)
  v
Editability
  v
Narrative Context
```

**No master weighted score requirement** — restates D-098 Section 10.9's
existing "no arbitrary weighted master score" finding and Section
14.10's confirmation that every real confidence/ranking mechanism
inspected in this codebase is categorical/evidence-based, never a single
composite number. **Meaning remains P0** — restates D-107 Section 9's
existing tier-(1) priority, unchanged; BestTake Multimodal Fusion V2 only
elaborates tier (3), it never promotes Visual/Vocal Delivery ahead of
Meaning or Take Usability. No implementation is authorized here.

### 15.7 Continuation / Minimal Composite doctrine (formalizes existing
doctrine, no code change)

**Status: doctrine formalization of an ALREADY-EXISTING authority
(`CompositeResolver`/`RealizationResolver`, Section 3's existing L6 row),
not a new capability.**

**Placement is NOT a single universal stage (corrected D-178A.1 —
binding).** D-178A's original Cut.ai Parity Stack diagram (15.10)
implied Continuation/Minimal Composite always runs strictly AFTER
BestTake. That is corrected to two distinct conceptual cases:

**Case A — REALIZATION CONSTRUCTION (before Freeze).** When multiple
source-real complementary pieces are REQUIRED to construct one
meaning-sufficient realization in the first place (no single complete
take exists), the bounded composite must be constructed as part of
Stable Family / Realization Formation (15.10's corrected stack) — BEFORE
final realization competition and Freeze — so BestTake/Realization
authority has a real, complete, selectable realization to evaluate
against any competing complete take. This is the ALREADY-EXISTING
behavior of `CompositeResolver`/`RealizationResolver` (Section 3's L6
row, unchanged code) — this correction only names its correct pipeline
position explicitly for the first time.

**Case B — POST-SELECTION COMPLETION (after Freeze, narrowly bounded).**
A later, bounded continuation/repair may exist only where already
supported by an existing canonical authority (e.g., Boundary's own
edge-only physical repair, D-107 Section 9/D-177) and must NEVER mutate
meaning, membership, or family doctrine implicitly. This is not a second
composite mechanism — it restates the existing, narrow, physical-only
repair authorities already named elsewhere in this document (Boundary,
Section 13.11).

Canonical doctrine, restated and sharpened (applies to Case A; Case B is
explicitly narrower and physical-only):

**Complete same-job takes compete.** Two complete, sufficient
realizations of the same proposition/job are Good-vs-Good competitors
(restates Layer 14's existing target, and D-107 Section 9's existing
BestTake ordering) — a composite is never built merely because a second
complete take exists.

**Complementary incomplete pieces may form a minimum sufficient
composite ONLY when ALL of the following hold** (restates and
consolidates the existing `RealizationResolver`/`CompositeResolver`
usable-first-tiers, no-composite-with-failed-members, and
critical-veto-not-composite-forcing doctrine, Section 3's existing L6
row, unchanged code):

1. **source-real** — every piece is real RAW content, never invented
   speech (restates the binding "Never invent speech" editorial rule);
2. **same proposition/slot** — pieces address the SAME proposition (15.2's
   Proposition Identity doctrine), never merged across distinct
   propositions;
3. **nonduplicate** — no piece restates content another piece already
   supplies (restates Layer 12's target REDUNDANT classification);
4. **meaning-safe** — polarity/negation/numbers/factual terms survive
   intact (restates `polarity_safety.py`'s existing binding invariant);
5. **correct order** — pieces are sequenced per Ordering/Sequence
   Intelligence (15.8), never assembled out of narrative order;
6. **Boundary-isolatable** — each piece has a clean, Boundary-safe
   physical cut point (restates D-107 Section 9's CASE A/B ownership
   split — Boundary still trims; BestTake/Resolver still decides
   membership);
7. **minimum necessary pieces** — no more pieces than the minimum
   sufficient set (restates Layer 13's existing target objective).

**Do not implement** — this section formalizes doctrine already governing
the EXISTING `CompositeResolver`/`RealizationResolver` authority
(unchanged code); it authorizes no new composite logic, no threshold, and
no BestTake/Family change.

### 15.8 Ordering / Sequence Intelligence (new, distinct capability)

**Status: MISSING/FUTURE, named here for the first time as its own
distinct capability.**

Canonizes Ordering / Sequence Intelligence as DISTINCT from Selection,
BestTake, Boundary, and Pacing — restates and sharpens the existing
authority-separation doctrine (Section 4/13.3.2: each authority owns one
question). Purpose: determine the CORRECT ORDER of already-valid
editorial pieces (i.e., pieces Selection/BestTake has already decided
belong in the winning edit). This is explicitly NOT:

- Selection/BestTake (which decides WHAT survives, Layer 6);
- Boundary (which decides WHERE a clip starts/ends physically, Layer 7,
  D-107 Section 9);
- Pacing (which decides tightness/overlap BETWEEN two already-ordered,
  already-boundary-safe clips, D-129 Section 11, Section 13.12).

Ordering sits between Freeze/Boundary and Pacing in the Cut.ai Parity
Stack (15.10) — it answers "in what sequence do these already-frozen,
already-boundary-safe pieces play," which today has no dedicated
authority (the closest existing evidence is `realization_resolver.py`'s
own CTA-ordering logic, Section 14.7, which is narrow and
prompt-embedded, not a general sequence authority). No implementation is
authorized here.

### 15.9 Boundary current state (records D-177, does not modify Boundary
authority)

**D-177 — PARTIAL-EDGE BOUNDARY TRIM: OFFLINE PROVEN (verdict A).**
Records, without modifying, the current state of `boundary_engine_pass.py`
(Section 3's existing EXISTING L7 row, Section 13.11's CASE A/B/C split,
unchanged): a one-sided visual/performance event that STRADDLES the
measured DELIVERY boundary (D-115's `starts_before_delivery` XOR
`ends_after_delivery`) may now trim the EXTERNAL debris portion to the
measured DELIVERY hard floor (`delivery_span.start`/`.end`) WITHOUT
cutting into required speech — the trim clamps to the same hard floor the
pre-existing pure ENTRY/EXIT loops already used, so it can never remove
any part of a real word (docs/CUTSELL_DECISIONS.md D-177). **No new
threshold** — reuses the existing `AUDIO_EDGE_OVERLAP_TOLERANCE_SEC`/
`AUDIO_EDGE_MINIMUM_REMAINING_SEC` constants verbatim. **No BestTake
change** — Boundary still only trims debris; BestTake still chooses among
realizations, unchanged (D-177's own core doctrine, restating D-107
Section 9's CASE A ownership). Offline-proven: 34/34 new tests, zero
regression across the full Boundary/D-116/D-097-C/D-123/D-163/D-167/D-174
etc. battery and the full offline suite (D-177 decision entry, this
document's own precedence rule: live decision-log state over any
document's restatement of it). **Next runtime gate:** exactly ONE Video00
RAW (named, not launched by D-177 or by this document) — see 15.16.

### 15.10 Cut.ai Parity Stack (canonical conceptual pipeline, restates and
consolidates Section 13.1, integrates 15.4/15.5/15.7/15.8 — CORRECTED
dependency order, D-178A.1)

**This diagram supersedes D-178A's original 15.10 diagram, which placed
Editorial Moment & Sequence Understanding and Whole-Video Editorial
Reasoning UPSTREAM of Behavior State/Proposition/Family — implying they
replace or precede the structured objects they actually depend on. That
was an error, corrected here.** This is a clearer, more complete READING
of Section 13.1's existing top-level pipeline, now naming 15.4/15.5/15.7/
15.8's capabilities at their CORRECT dependency position — no layer is
renumbered, no existing file's ownership moves, no new authority is
created by drawing this diagram; this is dependency direction only, not
authorization to implement any future layer:

```
RAW
  v
PARALLEL PERCEPTION                    (Layer 1, Section 13.2: Language,
  v                                     Visual, Audio, Media/Timing)
WATCH+LISTEN MULTIMODAL UNDERSTANDING  (Layer 1->2 boundary, Section 13.3)
  v
STRUCTURED LOCAL UNDERSTANDING          (Behavior State, Section 13.4;
  v                                      LanguageWord/Phrase/Utterance/
                                         Attempt, Section 14.2, CURRENT
                                         STATE corrected 15.2/15.2.1;
                                         PropositionCandidate/
                                         RelationEvidence, Section 13.5-
                                         13.6/14.2)
  v
EDITORIAL MOMENT & SEQUENCE UNDERSTANDING   (NEW, Section 15.4 --
  v                                          CONSUMES the layer above,
                                             corrected D-178A.1)
WHOLE-VIDEO EDITORIAL REASONING             (NEW, Section 15.5 --
  v                                          CONSUMES both layers above,
                                             corrected D-178A.1)
STABLE FAMILY / REALIZATION FORMATION   (Layer 4, Section 13.7-13.8 +
  v                                      Case A Continuation/Minimal
                                         Composite where eligible,
                                         Section 15.7 -- renamed/
                                         clarified D-178A.1, see below)
BESTTAKE                               (Layer 6, Section 13.10, 15.6 future
  v                                     -- decides among VALID REALIZATIONS)
FREEZE                                 (Layer 5/6 boundary)
  v
ORDERING                               (NEW, Section 15.8)
  v
BOUNDARY                               (Layer 7, Section 13.11, D-177 15.9
  v                                     -- Case B bounded post-selection
                                        completion, Section 15.7, is a
                                        narrow Boundary-owned repair here,
                                        never a second composite mechanism)
PACING                                 (Layer 7 sub-stage, Section 13.12)
  v
RENDERER                               (Layer 7 execution)
```

**Corrected dependency statement (D-178A.1, binding):** Editorial Moment
& Sequence Understanding and Whole-Video Editorial Reasoning are
HIGHER-ORDER UNDERSTANDING layers that CONSUME evidence from Parallel
Perception, Watch+Listen, and Structured Local Understanding (Behavior
State + the Language Spine's `LanguageAttempt`/`PropositionCandidate`/
`RelationEvidence`) — they do not precede, replace, or independently
recreate any of those structured objects (15.4/15.5's own corrected
text). They may form higher-order HYPOTHESES (recording process, clean
audience delivery, preassembled final sequences, global redundancy,
sequence structure) that Stable Family/Realization Formation may then
consume — they do not decide membership themselves (13.3.2's authority
principle, unchanged). This restates the doctrine, corrects only the
diagram's implied ordering. They do not sit downstream as a QA/routing
role either (that remains Layer 8's Downstream Watch+Listen QA,
unchanged, Section 4/13.3.4).

**Stable Family / Realization Formation (named here, D-178A.1) —
placement and scope.** This is the pre-BestTake conceptual stage
(renames/clarifies what D-178A's original diagram called "FAMILY
FORMATION" alone) where BestTake's actual input candidates are
established: COMPLETE individual realizations (the ordinary case,
Section 13.7-13.8's existing `take_grouping.py` family, unchanged code),
and, ONLY when necessary and eligible per 15.7's 7-condition test (Case
A), a minimum sufficient COMPOSITE realization. **This must NOT imply
every family gets a composite** — for the ordinary case of one or more
complete competing takes, no composite is built; Case A composite
construction is the narrow exception, not the default path. Ordering and
Boundary are placed exactly where the existing target sequence authority
(15.8) and `boundary_engine_pass.py` (Section 13.11, D-177) already
conceptually sit relative to Freeze/Pacing/Renderer, unchanged from
D-178A's original placement.

### 15.11 Post-Parity Commercial Stack: Commercial Moment Understanding
vs Editorial Moment Understanding (new, explicit separation)

**Status: MISSING/FUTURE, named here for the first time. Belongs primarily
to the future Sales Funnel system (15.14), explicitly POST-PARITY.**

**Editorial Moment Understanding** (15.4, this section) answers: "what
role did this moment play in the RECORDING/EDITING PROCESS" — a
Milestone-1 (RAW -> Cut.ai) concern.

**Commercial Moment Understanding** (new, named here) answers: "what
COMMERCIAL/PERSUASIVE job does this kept content perform for an
audience/buyer" — a POST-PARITY, POST-HUMAN-GOLD concern, explicitly
separate and never conflated with Editorial Moment Understanding. Future
commercial roles/evidence may include: `HOOK`, `PROBLEM`, `SOLUTION`,
`FEATURE`, `BENEFIT`, `PROOF`, `DEMONSTRATION`, `OBJECTION`, `CTA`; and
evidence dimensions `specificity`, `novelty`, `commercial_relevance`,
`redundancy`, `product_presence`, `attention_strength`,
`commercial_usefulness`.

Relationship to existing dormant vocabulary (clarifies, does not
implement): `contracts.SemanticRole` (`HOOK`/`PROBLEM`/`FEATURES`/
`BENEFITS`/`PROOF`/`STORY`/`CTA`/`OTHER`, Section 14.9) is the closest
existing typed enum, but it is DORMANT on the active Clean Cut V1 path
today (Section 14.8/14.9's own finding, unchanged) and, per CLAUDE.md's
own binding rule ("Do not force rigid sales-funnel logic during Clean
Cut"), must remain out of scope for Clean Cut / Milestone 1 reasoning.
Commercial Moment Understanding is the FUTURE, properly-scoped home for
eventually re-activating and extending that vocabulary — strictly
post-parity, never pulled forward into Milestone 1's active path. No
implementation is authorized here.

### 15.12 Sales Funnel Intelligence (future post-parity layer)

**Status: MISSING/FUTURE, restates and extends Layer 10 (Editorial
Function)'s vocabulary toward a commercial-specific instance.**

A future, FLEXIBLE (not rigid) post-parity layer supporting MULTIPLE ad
structures, not one forced funnel shape: `Hook`, `Problem`, `Solution`,
`Benefit`, `Proof`, `CTA` as a starting vocabulary, explicitly extensible
— never hardcoded as the only valid structure. Consumes Commercial Moment
Understanding (15.11) evidence; does not replace or precede Editorial
Moment Understanding (15.4) or any Milestone-1 authority. No
implementation is authorized here.

### 15.13-15.14 (see 15.11-15.12 above; numbered per this section's own
internal cross-references)

### 15.15 Implementation Priority (canonical Product Owner sequence, P0-P11)

This is the canonical priority ordering as of D-178A. It does not itself
authorize any step — each step still requires its own bounded-capability
authorization per Section 6's anti-loop/execution contract, exactly like
every prior roadmap in this document (D-111 Section 10.10, D-148 Section
13.22).

- **P0** — one real-media D-177 Boundary qualification (15.9's named next
  gate; the one item in this list that already has a fully-specified,
  offline-proven implementation waiting on Product Owner authorization to
  run).
- **P1** — Editorial Moment & Sequence Understanding (15.4).
- **P2** — Whole-Video Editorial Reasoning (15.5).
- **P3** — Prosodic Audio Understanding (Audio V2, 15.3).
- **P4** — BestTake Multimodal Fusion V2 (15.6).
- **P5** — Continuation / Minimal Composite (15.7's doctrine, made
  concrete/extended if evidence justifies it).
- **P6** — Family Formation stability/generalization (Section 13.7-13.8's
  existing evidence-breadth doctrine, hardened).
- **P7** — Ordering / Sequence Intelligence (15.8).
- **P8** — Pacing V2 (extends D-129/Section 13.12; J_CUT/L_CUT/
  MICRO_AUDIO_OVERLAP per Section 13's existing D-142 status row).

Then Cut.ai parity qualification on multiple unseen RAWs (15.16's
Generalization Gate).

**POST-PARITY:**

- **P9** — Commercial Moment Understanding (15.11).
- **P10** — Sales Funnel Intelligence (15.12).
- **P11** — Human Gold refinement / higher editorial sophistication
  (Milestone 2, Layers 10-16, Section 5/13.15, unchanged, still not
  started).

### 15.16 Generalization Gate (binding, restates and sharpens Section
5/13.14's Milestone-1 exit criterion)

**Do not declare RAW -> Cut.ai from Video00 alone.** After Video00
stability (i.e., once Video00-measured LEVEL-1 discrepancies reach a
stable, low-Level-1 state), the required next step before declaring
Milestone 1 complete is to qualify **5-10 UNSEEN RAWs** with **no
video-specific rules** — restates and makes numerically concrete Section
5's existing "(J) Cut.ai-level behavior demonstrated on unseen RAWs" exit
criterion and Section 6's binding "never hardcode Video00 timestamps,
phrases or clip IDs" rule (CLAUDE.md's own Editorial rule, restated here
as an architectural gate, not merely an editorial guideline). No RAW is
launched by naming this gate.

### 15.17 Canonical Doctrines (restated, binding, extends Section
6/10.4-10.5/13.19's invariant lists — no new invariant removed or
weakened)

- **PERCEPTION PROPOSES EVIDENCE.** (Section 13.3.2, unchanged)
- **UNDERSTANDING FORMS HYPOTHESES.** (extends 13.3.2's phrasing;
  Editorial Moment/Whole-Video Reasoning, 15.4-15.5, are hypothesis-
  forming, never decision-making, per this same principle)
- **STRUCTURED EDITORIAL AUTHORITIES DECIDE.** (Section 13.3.2, unchanged
  — Proposition Identity, Family Formation, BestTake, Boundary, Freeze)
- **BOUNDARY/PACING EXECUTE.** (restates Section 4/13.11-13.12's existing
  authority statement — execution of a physical decision, not a semantic
  one)
- **QA REFERENCES NEVER BECOME RUNTIME INPUTS.** (restates CLAUDE.md's
  binding Cut.ai/Human Gold QA-only doctrine and Section 3's `video00_
  quality_ladder.py` "architecturally enforced never-imported-by-
  production" row, unchanged)
- **RAW REMAINS IMMUTABLE.** (restates D-107 Section 9's non-destructive
  editing doctrine, Section 13.18, unchanged)
- **MEANING PRESERVATION IS P0.** (restates D-107 Section 9's BestTake
  tier-(1) priority and `polarity_safety.py`'s binding invariant,
  unchanged)
- **NO COMPLETE FAMILY CONTEXT -> NO AUTHORITATIVE COMPARATIVE WINNER.**
  (restates D-145, strengthened by D-147's `COMPLETE_CONTEXT_CONFLICT`,
  Section 13.8/13.8.1, unchanged)
- **PROPOSITION IDENTITY PRECEDES RETRY IDENTITY.** (restates D-111
  Section 10.2/Section 13.5, unchanged, binding)

### 15.18 Confirmation: D-177 preserved, unmodified

No part of this section changes, weakens, or supersedes D-177's
implementation, its offline proof, its verdict (A: PARTIAL-EDGE BOUNDARY
TRIM OFFLINE PROVEN), or its named next real-media gate. Section 15.9
above restates D-177's finding for architectural context only — the
authoritative record remains `docs/CUTSELL_DECISIONS.md`'s own D-177
entry, per this document's own precedence rule (Section 0's header:
"Where sources conflict, the accepted contract wins").

### 15.19 Confirmation: documentation only

No `cutsell_worker/*.py` file, no `tests/*.py` file, no `.github/
workflows/*.yml` file, no feature flag, and no authority contract was
touched, added, or changed to write this section. No RAW was dispatched.
No provider/network call was made.

### 15.20 Exact next runtime gate (not launched by this document)

Per P0 (15.15) and D-177's own named next step (15.9): exactly ONE
Video00 RAW on the current head, evaluating D-177's partial-edge Boundary
trim on the gynecologist-retry region, against D-177's own success
criteria (same selected take, same content coverage, entry/exit
`false_keep` window shrinks toward/to zero, no new clipped phoneme or
truncation anywhere, no new Boundary regression elsewhere, zero BestTake/
Family/DeliveryScorer/Watch+Listen verdict change, F1 vs Cut.ai improves
or remains stable). **Not launched by this document.** Requires separate
Product Owner authorization, per Section 6's anti-loop/execution
contract and D-177's own "Wait for Product Owner authorization" close.

---

No change to Section 2's 20 layers, Section 3's status table, or any
accepted D-096 through D-177 authority contract. Family Formation,
Proposition Identity, Attempt Relationships, BestTake, Boundary, and
Pacing are all restated, never modified, by this section. See
`docs/CUTSELL_DECISIONS.md` D-178A for the decision-log entry recording
this section's doctrine.
