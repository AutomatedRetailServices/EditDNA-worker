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
