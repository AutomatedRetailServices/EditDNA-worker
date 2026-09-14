# D-121 — BestTake CASE B forensic + design (multimodal performance during DELIVERY)

Status: **FORENSIC + DESIGN ONLY**. No engine behavior changed. No RAW. No
provider calls. No fallback implemented. This document is read alongside
`docs/CUTSELL_DECISIONS.md` D-121 and does not by itself authorize any
implementation.

## 1. Scope and method

This is a code-traced audit (every claim below cites the file/function that
proves it) plus a real-evidence trace across four persisted Video00 RAWs
(D-113 `34143240479`, D-116 first qualification `34150026795`, D-118
`34160335330`, D-120 `34162868778`). Nothing here is inferred from
intent/comments alone where the actual code could be read; nothing is
fabricated where evidence was not retrievable — those gaps are stated
explicitly.

## 2. Current BestTake dataflow (traced)

```
takes (CandidateTake, post attempt-reconstruction)
  -> semantic_equivalence_groups (IdeaClusterer / semantic-equivalence arbiter)
       = the ONLY place a retry-family COMPETITOR SET is formed
  -> per family, pipeline.py::build_flow_b_draft loop:
       judged = safe_rank_takes(members)              # take_judge.rank_takes -> score_take
       ranked, cleanliness_rows = apply_delivery_cleanliness_evidence(judged.ranked, members, source_events)
                                                        # D-097: interior dead-air / multimodal-reset penalty
       local_selected_clip_id = ranked[0].clip_id      # DeliveryScorer's own pick
       family_semantic_decisions = family_scoped_semantic_decisions(...)   # Hybrid/Gemini labels for this family
       deterministic_unusable = {...}                  # ranker fragment-penalty OR hybrid local-failure corroboration
       selected_clip_id, semantic_preferred_clip_id, reason = _semantic_best_take(
           members, family_semantic_decisions, local_selected_clip_id, ranked,
           semantic_delete_recommended=..., deterministic_unusable=...,
       )
  -> draft.selected / draft.discarded populated per family
  -> apply_deterministic_best_take_authority(draft, swap_enabled=False)   # POST-HOC, reads the SAME take_judge_groups
       -- can move clips again if the ranked top-two gap >= 0.30 (CLEAR_WINNER_MINIMUM_GAP)
  -> apply_claim_coverage_best_take(...)                # D-038: critical-claim safety net
  -> apply_final_story_coherence_validation(...)        # StoryValidator, pre-Freeze
  -> Selection Freeze
  -> apply_post_freeze_boundary_pass(...)               # D-116: audio + visual CASE A edge trims only
```

Both the per-family loop and `apply_deterministic_best_take_authority` are
confirmed ACTIVE in the current production path: `universal_clean_cut.py`'s
`clean_cut_core_v1_enabled` branch (the one CLAUDE.md names as the sole
active path) calls `apply_deterministic_best_take_authority(result.draft,
swap_enabled=False)` unconditionally, immediately after the initial draft
(`build_flow_b_draft`) is built.

## 3. Competitor-set owner (Q3)

The retry-family COMPETITOR SET — the group of candidates that ever compete
against each other at all — is formed **once**, upstream of everything else,
by the semantic-equivalence/IdeaClusterer grouping stage
(`semantic_equivalence_groups` in `pipeline.py`, built from the bounded
semantic arbiter + `split_incohesive_retry_groups` cohesion gate). Neither
`take_judge.rank_takes`, `_semantic_best_take`, nor
`apply_deterministic_best_take_authority` can ever add a member to a
family or merge two families — they only choose among members a family
already has. **This means a BestTake regression can originate from grouping
alone, before any scoring or semantic-label logic runs at all** — see the
D-113 pimples trace below, where this is exactly what happened.

## 4. Meaning-sufficiency gate status (Q4)

**Exists and runs before any performance consideration**, but it is a
collection of narrow, purpose-built checks rather than one named "gate":
D-081 `semantic_delete_recommended` exclusion → attempt-completeness
exclusion → D-103 required-condition-realization exclusion → D-063/65/66
`resolve_critical_coverage_dominance` → contradiction/asymmetry safety
(`any_pair_contradicts`, unique-fact-coverage comparison) — all inside
`pipeline.py::_semantic_best_take`, all reused verbatim (no new heuristic
per-call). This IS the doctrine's "same topic ≠ same proposition" and
"complementary ≠ retry" protection at the SELECTION-vs-DELIVERY boundary:
delivery/performance never even reaches step 6 ("delivery score / richness
tie-break") until every one of these has failed to find a decisive winner
AND the survivors are a genuine (non-contradictory, coverage-equal) tie.
**This part of the doctrine is already correctly implemented and must not
be touched by CASE B.**

## 5. Hybrid/Gemini current authority — the exact bypass (Q5/Q6)

`pipeline.py::_semantic_best_take`, lines 468–482:

```python
winners = []
for member in members:
    label, confidence = semantic_decisions.get(member.clip_id, ("", 0.0))
    if label == "winner" and confidence >= winner_confidence:   # winner_confidence=0.85
        winners.append((member.clip_id, confidence))
if len(winners) == 1:
    preferred_id, _ = winners[0]
    veto_reason = _single_winner_safety_veto(preferred_id, members, semantic_delete_recommended)
    if veto_reason is None:
        return preferred_id, preferred_id, "single_semantic_winner"   # <-- NEVER reads `ranked`
```

**When Hybrid/Gemini emits exactly one "winner" label at ≥0.85 confidence,
the function returns immediately.** `ranked` (DeliveryScorer's own
family-relative score, already including the D-097 cleanliness penalty)
is a parameter of `_semantic_best_take` but is **never read** on this path.
The only gate is `_single_winner_safety_veto`, which checks five things —
delete-recommended flag, `complete_idea is False`, factual contradiction,
missing CRITICAL-claim coverage, missing required-condition-realization —
**all five are meaning/safety checks; none is a performance/multimodal
check.** This is doctrinally correct for what it checks (meaning before
performance), but it means **zero performance evidence, position-aware or
aggregate, is ever consulted when Hybrid is decisive.** This is the exact
bypass the task asked to identify — confirmed by code, and confirmed live
by the D-118 pimples regression (Section 13).

A **second, independent** override path exists:
`deterministic_best_take_authority.apply_deterministic_best_take_authority`
runs AFTER the family loop, re-reads the same `take_judge_groups.ranked`,
and — regardless of what `_semantic_best_take` already decided, including
`single_semantic_winner` — moves the DeliveryScorer top-ranked member into
`select` whenever the top-two score gap is `>= 0.30` (`CLEAR_WINNER_
MINIMUM_GAP`) and that top member is not itself evidenced as a failed
fragment (`FRAGMENT_PENALTY_MARKERS`). So performance score **can** overturn
a Hybrid nomination today, but only as a coarse, all-or-nothing 0.30-point
blowout check — not a graduated influence, and not aware of D-115 position
or DELIVERY-zone evidence at all (it reads the same aggregate `ranked`
scores `take_judge.score_take` already produced).

## 6. Current DeliveryScorer components (Q7) — `take_judge.score_take`

| Term | Weight | Source | Position-aware? |
|---|---|---|---|
| `completeness` (text-based) | 0.16 | `CandidateTake.complete_idea` | no |
| `duration_fit` | 0.06 | duration bucket | no |
| `audio_quality` | 0.12 | `MediaSignals.audio_quality` | no |
| `face_visibility` | 0.08 | `MediaSignals.face_visibility` | no |
| `eye_contact` | 0.09 | `MediaSignals.eye_contact` | no |
| `framing_quality` | 0.06 | `MediaSignals.framing_quality` | no |
| `product_visibility` | 0.05 | `MediaSignals.product_visibility` | no |
| `motion_stability` | 0.07 | `MediaSignals.motion_stability` | no |
| `continuity` | 0.07 | `MediaSignals.continuity` | no |
| `expression_naturalness` | 0.10 | `MediaSignals.expression_naturalness` | no |
| `gesture_naturalness` | 0.07 | `MediaSignals.gesture_naturalness` | no |
| `delivery_energy` | 0.07 | `MediaSignals.delivery_energy` | no |
| `visual_fumble` | −0.12 | `MediaSignals.visual_fumble` | no |
| `distraction_risk` | −0.08 | `MediaSignals.distraction_risk` | no |
| `_handling_failure_penalty` | up to −0.23 compound | combination of the above | no |

**Every term is a whole-take aggregate scalar** (`MediaSignals`, computed
once per take), never a positioned/DELIVERY-zone value. `score_take` never
imports or reads anything from `positioned_performance_evidence.py`.

**Layered on top, per retry family (not per-take), before the winner is
read:** `take_judge.apply_delivery_cleanliness_evidence` /
`delivery_cleanliness_evidence` (D-097) — the closest thing that exists
today to CASE B. It re-derives its OWN "interior" window
(`take.start + 0.35s .. take.end - 0.35s`, `_CLEANLINESS_EDGE_MARGIN_SEC`),
independent of D-115's word-derived DELIVERY span, and penalizes:
`interior_dead_air_penalty` (0.12/interval, capped 0.24, for an
`audio_silence_interval` ≥1.20s inside that window) and
`multimodal_reset_penalty` (flat 0.10, requires BOTH a strong reset
event ≥0.88 confidence AND an independent break event ≥0.76 confidence
inside that same window). **This already reads the identical raw
`TemporalEvent` stream D-115/D-116 read** (`body_reset_candidate`,
`hand_motion_reset_candidate`, `camera_disengagement_candidate`,
`facial_expression_shift_candidate`, `audio_silence_interval`) but through
its own margin-based window, not D-115's canonical `compute_delivery_span`/
`classify_event_zone`.

## 7. REAL vs DEFAULT component table (Q8, Q with D-099/D-114 re-audit)

| `MediaSignals` field | Status | Real producer | Position-aware/aggregate | Weighted in `score_take`? |
|---|---|---|---|---|
| `face_visibility` | REAL (when local_performance runs) | `local_performance.apply_local_performance_to_takes` (cv2/MediaPipe Holistic face detection, duration-window mean) | aggregate (whole-take mean) | yes, 0.08 |
| `eye_contact` | REAL (when local_performance runs) | same, `eye_contact_proxy` mean | aggregate | yes, 0.09 |
| `motion_stability` | REAL (when local_performance runs) | same, motion-variance derived | aggregate | yes, 0.07 |
| `visual_fumble` | REAL (when local_performance runs) | same, `body_reset_candidate`+`hand_motion_reset_candidate`+`facial_expression_shift_candidate` event-count / duration | aggregate (counts events without regard to position in the take) | yes, −0.12 (and inside `_handling_failure_penalty`) |
| `expression_naturalness` | REAL (when local_performance runs) | same, `facial_expression_shift_candidate` event count | aggregate | yes, 0.10 |
| `gesture_naturalness` | REAL (when local_performance runs) | same, `body_reset_candidate`+`hand_motion_reset_candidate` count | aggregate | yes, 0.07 |
| `distraction_risk` | REAL (when local_performance runs) | same, `camera_disengagement_candidate` count | aggregate | yes, −0.08 |
| `audio_quality` | REAL | `take_segmentation.py::_audio_quality` (audio-ratio derived) | aggregate | yes, 0.12 |
| `framing_quality`, `product_visibility`, `continuity`, `delivery_energy` | **PARTIAL** — a real code path exists (`visual_analysis.py`, `visual_openai.py`, an OpenAI-vision provider) that CAN produce measured values, but the dataclass default (0.5/0.0/0.5/0.5) applies whenever that provider is not invoked/available for a given run | `visual_analysis.py`/`visual_openai.py` (provider-gated) | aggregate | yes, 0.06+0.05+0.07+0.07 = **0.25 of total weight is PARTIAL/possibly-default** |
| `silence_ratio` | REAL | `take_segmentation.py` | aggregate | **not read by `score_take` at all** |

**This task did NOT confirm whether `visual_analysis`/`visual_openai` was
active or default-only for the four traced Video00 RAWs** (D-113/D-116-
first/D-118/D-120) — that would require inspecting each run's own
`diagnostics.visual_*` provider-status block, which was not pulled in this
pass. This is the honest gap to flag for a future D-099/D-114 re-audit
rather than an assumption: **0.25 of `score_take`'s total positive weight
may be running on defaults on every real Video00 RAW to date, silently
indistinguishable from a genuinely neutral/average take.** No current
weighted term was found "pretending" to be measured when it structurally
cannot be (each field's producer path is real code, not a stub) — the risk
is specifically whether that code path is exercised in production, which
this task leaves open.

## 8. Current position-aware CASE B evidence (Q9)

`positioned_performance_evidence.py` (D-115) computes, for every take, a
`DeliverySpan` (from `CandidateTake.words`) and a list of `PositionedEvent`
rows (ENTRY/DELIVERY/EXIT-classified, straddle-aware) from the SAME four
local-performance kinds plus `audio_silence_interval`. **It is called
exactly once in the active pipeline**, in `flow_b.py` (~line 375), and its
output is stored ONLY under `attempt_reconstruction_diagnostics
["positioned_performance_evidence"]` — diagnostics, never fed back into
`takes`, `MediaSignals`, `score_take`, `_semantic_best_take`, or
`apply_delivery_cleanliness_evidence`. **The only real consumer of this
evidence anywhere in the codebase is D-116's `boundary_engine_pass.py`
(`tighten_selected_visual_edges`), and only for its own ENTRY/EXIT edge-trim
decision — never for scoring.** This confirms the CASE B gap precisely as
the task frames it: the canonical DELIVERY-zone classification already
exists and is already computed per take; nothing downstream of Boundary
reads it.

## 9. Double-counting risk — CONFIRMED, already active today (Q10)

This is not hypothetical. **The identical raw events are already consumed
twice, independently, by two different mechanisms active in every run:**

1. `local_performance.apply_local_performance_to_takes` folds
   `body_reset_candidate`/`hand_motion_reset_candidate`/
   `facial_expression_shift_candidate`/`camera_disengagement_candidate`
   into the `MediaSignals` aggregates (`visual_fumble`, `distraction_risk`,
   `expression_naturalness`, `gesture_naturalness`, `motion_stability`)
   using a duration-weighted count over the WHOLE take window — these
   already influence `score_take`'s base score before any family
   comparison happens.
2. `take_judge.delivery_cleanliness_evidence` re-derives events from the
   SAME raw `TemporalEvent` list (a second, independent read, this time
   windowed to `take.start+0.35s .. take.end-0.35s`) and applies an
   ADDITIONAL `multimodal_reset_penalty`/`interior_dead_air_penalty` on
   top of the score that (1) already reflects.

**A future CASE B consumer reading D-115's DELIVERY-zone events would be a
THIRD independent consumption of the same physical defect.** Any CASE B
design MUST either (a) replace/subsume (1) and (2) rather than stack a
third scorer on top, or (b) explicitly zero out or discount the
contribution of (1)/(2) for the specific event instances CASE B also
scores, with the double-counting boundary stated in code, not left
implicit. This document does not resolve which; it is the single most
important open design question CASE B must answer before implementation.

## 10. PERFORMANCE_CONTINUITY recommendation (Q11)

Recommended shape (no thresholds, no weights — counts/durations only,
each traceable to a timestamp):

- `delivery_event_count` — count of `PositionedEvent`s classified
  `zone == DELIVERY` for this take (from D-115, unmodified).
- `delivery_reset_count` — of those, count with `kind in
  {body_reset_candidate, hand_motion_reset_candidate}`.
- `delivery_disengagement_count` — count with `kind ==
  camera_disengagement_candidate`.
- `delivery_expression_break_count` — count with `kind ==
  facial_expression_shift_candidate`.
- `delivery_straddle_count` — count with `starts_before_delivery or
  ends_after_delivery` true (an event D-116 could never safely trim
  because it also touches ENTRY/EXIT — exactly this run's `visual_cross_
  boundary_count` in the D-120 evidence).
- Each row individually inspectable back to `(kind, start, end,
  confidence)` — no collapsing into a single opaque scalar before the
  observability layer (Section 15) can show the raw rows.

**Explicitly NOT recommended:** a single `performance_continuity_score`
float computed by this task — that would be inventing a weight, which is
out of scope here.

## 11. Delivery energy fit — current status (Q12)

`delivery_energy` is a real `MediaSignals` field with a real (provider-
gated) producer (`visual_analysis.py`/`visual_openai.py`, Section 7), but
**no position-aware energy signal exists anywhere** — D-115/`local_
performance.py` measure resets/disengagement/expression-shift, never
tone/pacing/enthusiasm. **Stating explicitly, per this task's instruction:
today's energy-fit evidence, when present at all, is default-or-provider-
aggregate only — never position-aware, never DELIVERY-zone-scoped.** No
future position-aware energy score is proposed here.

## 12. Editability/Boundary-quality interaction (Q13)

D-116 CASE A already owns and safely trims any real ENTRY/EXIT-zone event
that touches a clip's own current edge (Section 5/6 of D-116/D-120's own
decision entries). Per this task's explicit instruction, **CASE B must
only ever score a DELIVERY-zone event (or a straddling ENTRY/EXIT event
Boundary could not safely trim without crossing the delivery floor) — an
EXIT-only defect fully outside DELIVERY that Boundary CAN safely trim must
never also be scored as a performance defect**, or the same physical
imperfection would be simultaneously punished twice through two different
authorities (Boundary trimming it away AND BestTake penalizing a take that
Boundary already cleaned up). The natural implementation boundary: CASE B
consumes exactly the `PositionedEvent`s with `zone == DELIVERY` (D-115's
own classification), nothing D-116 ever touches or could safely touch.

## 13. Pimples cross-run trace (Q14) — real evidence, four runs

| Run | Pimples-region Level-1 | Family formed? | Mechanism (traced) | Category |
|---|---|---|---|---|
| D-113 `34143240479` | 0s baseline, but a `false_keep`/`ungrouped_retry_of_kept_idea` finding exists at the pimples-adjacent region, attributed to `IdeaClusterer/RetryFamilyFormation` | **NO** — traceability table shows both pimples clips (`clip_...da6431bd` "También me salían espinillas." and `clip_...4334221d9209` "Otro síntoma...") with blank `family`/`idea` columns; ladder: `"clip_d73ff2b74334221d9209 was never grouped with clip_043b363a9823da6431bd although their content overlaps"` | **Grouping never formed a contest at all** — no DeliveryScorer ranking, no semantic label, ever ran on this pair | **A — competitor-set (grouping) variance** |
| D-116 first qualification `34150026795` | ≈19.689s, attributed to `BestTakeResolver`/`take_choice_against_both_references` (per D-116's own decision entry; not independently re-traced with fresh log evidence in this task) | Yes (per the D-116 entry's own characterization) | Not re-derived at the `semantic_best_take_reason` level in this task — **flagged as not independently confirmed here**, relying on the already-recorded D-116 entry characterization | **B or C (unconfirmed which)** |
| D-118 `34160335330` | 17.439s physical, family `tg_72535925d777047e70`, `BestTakeResolver`/`take_choice_against_both_references` | **Yes** — `take_judge_groups` entry: `"group_id": "tg_72535925d777047e70", "selected_clip_id": "clip_6cc1223155bdc96db3b8", "semantic_best_take_reason": "single_semantic_winner"`. Hybrid/Gemini decisions: `clip_1cebf820...` = `("alternate", 0.8, local_failure_corroborated=true, local_failure_reasons=["dense_physical_reset:7","visual_fumble:0.85"])`; `clip_6cc1223...` = `("winner", 0.95, local_failure_corroborated=true, local_failure_reasons=["dense_physical_reset:7"])`. **Both candidates already carry real, already-computed local-failure evidence; the eventual winner carries FEWER failure reasons (1) than the loser Cut.ai/Gold both actually chose (2), but `_semantic_best_take`'s `single_semantic_winner` fast path never compares them — it only checked `label=="winner" and confidence>=0.85`.** | **B — semantic-arbiter variance, via the confirmed `single_semantic_winner` bypass (Section 5)** — the strongest, most concrete evidence in this trace |
| D-120 `34162868778` | 0s — no pimples-family LEVEL_1 region at all; the run's one `BestTakeResolver` region (0.75s) is an unrelated gynecologist-visit family | Not independently re-checked whether a pimples family formed and was decided correctly, or never contested at all (D-119 summary covers Boundary/Watch+Listen only, not `take_judge_groups`) | **Not determined in this task** — flagged rather than guessed | **Unknown (A or correct-B/C)** |

## 14. Primary cause of pimples variance (Q14/Q15 answer)

**E — multiple factors, both independently confirmed by code+evidence in
this trace, neither ever attributable to D-116/Boundary:**

1. **Grouping/competitor-set variance** (D-113): whether the two competing
   pimples deliveries are even recognized as the same retry family at all
   is itself unstable — when they are not grouped, no scoring layer of any
   kind ever runs on the pair.
2. **Semantic-arbiter bypass of already-computed performance evidence**
   (D-118, concretely traced): when Gemini/Hybrid emits one decisive
   `"winner"` label, `_semantic_best_take`'s fast path accepts it without
   ever reading `ranked` (DeliveryScorer) or comparing the two candidates'
   own `local_failure_reasons` — evidence that, in this exact real case,
   already existed and already favored the OTHER candidate.

**Boundary is ruled out as a cause in all four runs**, independently
confirmed three times over (D-116's own qualification, D-118, D-120):
Boundary runs strictly post-Freeze on already-frozen membership and, per
D-120's complete accounting, made zero visual trims in the one run where
full evidence was available — it cannot move which realization wins a
family contest that resolves upstream of Freeze.

**Could performance evidence have changed the D-118 winner?** Plausibly
yes — the losing candidate carried a strictly larger local-failure evidence
set (`dense_physical_reset:7` + `visual_fumble:0.85`, both already computed
and attached to the SAME diagnostics record) than the winner
(`dense_physical_reset:7` alone), and both references rejected the winner.
This is not proof a correctly-designed CASE B would have flipped the
result (the veto/gate design matters), but it is concrete proof the
information needed to at least RAISE the question was already sitting,
unused, in the same decision record.

## 15. Minimum CASE B representation (Q17)

Per-candidate, computed once from D-115's existing output, no new events:

```
case_b_evidence: {
  delivery_event_count, delivery_reset_count, delivery_disengagement_count,
  delivery_expression_break_count, delivery_straddle_count,
  events: [ {kind, start, end, confidence, straddle: bool}, ... ]   # auditable, not collapsed
}
```

Sourced by filtering `PositionAwarePerformanceEvidence.positioned_events`
to `zone == "DELIVERY"` — nothing new is measured; this is a re-projection
of D-115's existing, already-tested output.

## 16. Minimum future implementation shape (Q18)

1. Build `case_b_evidence` (Section 15) from the SAME
   `PositionAwarePerformanceEvidence` records `flow_b.py` already computes
   for diagnostics — no new event detector, no new provider call.
2. Resolve the double-counting question (Section 9) FIRST, explicitly, in
   code comments and a test: either (a) `case_b_evidence`'s DELIVERY-only
   events are excluded from `local_performance.apply_local_performance_to_
   takes`'s whole-take aggregate and from `delivery_cleanliness_evidence`'s
   interior window (so each physical event is scored by exactly one
   mechanism), or (b) CASE B is deliberately advisory/diagnostic-only at
   first (mirrors Watch+Listen's `advisory_v1` doctrine) until (a) is
   designed — Product Owner decision, not made here.
3. Attach `case_b_evidence` to `RankedTake`/`take_judge_groups` diagnostics
   for every family member — visible, never silently folded into a single
   score.
4. CASE B evidence influences ranking **only after** the existing meaning-
   sufficiency ladder (Section 4) has already run and found no decisive
   winner, and **only as an additional, clearly-labeled term** — never
   inserted ahead of or instead of the D-081/D-082/D-063 ladder steps.
   Whether it also gates the `single_semantic_winner` fast path (Section 5)
   — e.g., a new safety-veto condition alongside `_single_winner_safety_
   veto`'s existing five, refusing to trust a "winner" label whose own
   `case_b_evidence` is materially worse than a sibling's — is the single
   highest-leverage design question this forensic surfaced (Section 14),
   but implementing it is out of scope here.
5. `deterministic_best_take_authority`'s existing 0.30-gap override
   (Section 5) is a second point CASE B evidence could feed, or could be
   left untouched — not resolved here.
6. No new threshold, weight, or confidence cutoff is proposed by this
   document (per explicit scope).

## 17. Required observability (Q19)

Per competitor, at minimum:
`clip_id`, `semantic/proposition family id` (`group_id`), meaning-
sufficiency status (which ladder step, if any, excluded/preferred it),
DeliveryScorer base score (`take_judge.score_take` + `apply_delivery_
cleanliness_evidence` adjustment, already exposed today), `case_b_
evidence` (Section 15, new), the aggregate `MediaSignals` values consumed
by (1) in Section 9 (so a reviewer can see the double-counting boundary
directly), final performance score (whatever CASE B computes, if
anything), Hybrid/Gemini label+confidence, final BestTake winner, and the
exact reason string the winner changed or did not (extending the existing
`semantic_best_take_reason` vocabulary, e.g. a new
`case_b_veto`/`case_b_tie_break` reason alongside `single_semantic_winner`/
`delivery_tie_break_among_survivors`/etc.).

## 18. Future fallback trigger conditions (identified, not implemented)

Per D-111's own bounded-arbiter doctrine, a future multimodal fallback
arbiter would be appropriate ONLY when ALL of: semantic equivalence is
already high (a genuine retry-family contest, not a grouping failure —
Section 13 shows grouping failures are a SEPARATE problem a fallback
arbiter cannot fix), structured performance scores (CASE B once built)
conflict with each other across candidates, the top candidates are close
by BOTH meaning coverage and CASE B evidence (neither side is a blowout),
and the Hybrid/Gemini label disagrees with the measured CASE B evidence
(the exact D-118 shape). This remains FUTURE and unauthorized.

## 19. Risks / regressions to watch when CASE B is eventually built

- Double-counting (Section 9) silently over-penalizing a take whose real
  defect is legitimately small.
- A CASE B veto on the `single_semantic_winner` path weakening D-101's own
  hard-won safety fix if implemented carelessly (the veto function's
  existing five meaning checks must never be reordered after a new
  performance check).
- `apply_delivery_cleanliness_evidence`'s own interior window (0.35s
  margin) and D-115's DELIVERY span will disagree at the edges for some
  takes — CASE B must not silently inherit a stale margin-based window
  when D-115's word-derived span is available and more precise.
- Watch+Listen's `reset_debris_at_edges_source_evidence` capability
  (confirmed active, D-120 Section 13) already independently measures
  post-render residue from the same underlying events — a THIRD consumer
  if not accounted for in the eventual cross-authority ownership table
  (D-096 Appendix B).
