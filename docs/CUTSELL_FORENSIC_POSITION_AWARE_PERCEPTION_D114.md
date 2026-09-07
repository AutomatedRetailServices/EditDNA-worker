# D-114 — Position-Aware Performance Perception Forensic (Entry / Delivery / Exit)

**Status:** Design + dataflow forensic only. No `cutsell_worker/*.py`, test,
threshold, provider, or infra file was changed to produce this document.
**Scope:** post D-113. Answers whether, and how, CutSell can make performance
evidence position-aware relative to the take timeline, per D-111's CASE A
(post-delivery defect → BoundaryEngine, trim) / CASE B (delivery-overlapping
defect → BestTake, penalize usability) / CASE C (both, recorded independently)
doctrine (`docs/CUTSELL_CANONICAL_ENGINE_ARCHITECTURE_D098.md` §10.3).

**Headline finding:** the premise "no position-aware evidence exists" is
FALSE. Real, RAW-absolute-timestamped visual/motion events are produced today
by `local_performance.py` and survive, unaggregated, inside
`WholeVideoContext.sources[].events` for the life of a run. Four separate
consumers (`take_judge.delivery_cleanliness_evidence`,
`boundary_engine_pass.tighten_selected_audio_edges`,
`attempt_reconstruction._measured_pause_at_transition`,
`perceptual_watch_listen._reset_debris_at_edges`) already read that raw,
positioned stream directly, each for its own narrow purpose. What does NOT
exist is a **DeliveryScorer-visible** entry/delivery/exit split: the scalar
`MediaSignals` object that `take_judge.score_take` (BestTakeResolver's ranking
formula) actually consumes is produced by collapsing every frame/event in a
take into duration-weighted means (or, for two fields, a `max()`), which
destroys position before the score is computed. The gap is narrower and more
precisely located than "add position awareness from scratch" — it is "connect
BestTake/Boundary to evidence the system already has, instead of only to its
aggregated shadow."

---

## 1. Current position-aware signal map

| Signal (event kind) | Producer | RAW timestamp? | Clip-relative derivable? | Aggregate-only? | Current consumers | Current authority | Real vs default |
|---|---|---|---|---|---|---|---|
| `camera_disengagement_candidate` | `local_performance.py::detect_candidate_events` (MediaPipe Holistic, 12fps default) | YES — `PerformanceFrame.timestamp = index/source_fps`, source-absolute | YES (`event.start - take.start`) | NO (point-in-time interval) | `merge_local_events_into_context`, `apply_local_performance_to_takes` (aggregated away), `confirm_local_performance_events`, `take_judge._interior_events`, `perceptual_watch_listen._reset_debris_at_edges` | none of its own; feeds `MediaSignals.distraction_risk`/`expression_naturalness` via aggregation, and feeds `wrong_take`/`retry_setup` promotion | REAL when `local_performance.status.available` (unconditional call in `flow_b.py:198`, not gated by `visual_provider`) |
| `facial_expression_shift_candidate` | same | YES | YES | NO | same as above | feeds `MediaSignals.expression_naturalness` | REAL (same condition) |
| `body_reset_candidate` | same | YES | YES | NO | same as above + `attempt_reconstruction._measured_pause_at_transition`'s sibling window logic (indirectly, via `_events_near_transition` in `performance_confirmation.py`) | feeds `MediaSignals.motion_stability`/`gesture_naturalness` | REAL (same condition) |
| `hand_motion_reset_candidate` | same | YES | YES | NO | same | feeds `MediaSignals.gesture_naturalness`/`visual_fumble` | REAL (same condition) |
| `audio_silence_interval` | `audio_silence.py` (ffmpeg `silencedetect`) | YES, real interval + confidence | YES | NO | `attempt_reconstruction._measured_pause_at_transition`, `boundary_engine_pass.tighten_selected_audio_edges`/`split_selected_interior_performance_gaps`, `take_judge._interior_events` | BoundaryEngine (post-Freeze entry/exit/interior); AttemptReconstructor (transition boundary only, D-097.5) | REAL — the only signal class multiple authorities already consult directly at real timestamps |
| `wrong_take` / `retry_setup` (confirmed) | `performance_confirmation.py::confirm_local_performance_events` (promotes clusters of the four candidate kinds above + lexical retry similarity) | YES, but at **take/edge granularity**, not exact defect instant (`current.start`/`current.end` for `wrong_take`; edge-anchored span for `retry_setup`) | YES | Partially — position is the whole take, not the defect's own onset | `hybrid_retry_winner_authority._retry_setup_confidence`, `take_grouping` (pre-group credit), `performance_confirmation`-adjacent guards | RetryFamilyResolver / hybrid retry-authority chain | REAL (derived from the above) |
| word envelope (`CandidateTake.words[i].start/end`) | ASR alignment (upstream of this module set) | YES, word-level | YES | NO | `boundary_engine_pass.tighten_selected_audio_edges` (hard floor), `attempt_reconstruction._merge_attempt` (concatenated across fused members) | BoundaryEngine (hard floor only) | REAL |
| `MediaSignals.{face_visibility, eye_contact, motion_stability, visual_fumble, expression_naturalness, gesture_naturalness, distraction_risk}` (7 of 12 fields) | `local_performance.py::apply_local_performance_to_takes` (**aggregation point**) | NO — collapsed to one scalar per take | N/A | YES — this is the loss point | `take_judge.score_take` (DeliveryScorer formula), `hybrid_session_cleanup._local_failure` gate | DeliveryScorer / BestTakeResolver | REAL underlying data, but position-blind by the time it reaches the scorer |
| `MediaSignals.{audio_quality, framing_quality, product_visibility, continuity, delivery_energy}` (5 of 12 fields) | `visual_provider` (hardcoded `None` in `brain_runtime.py` — D-098 dataflow map Gap 2, unchanged since) | N/A | N/A | N/A | `take_judge.score_take` | DeliveryScorer | DEFAULT-only, permanently (unrelated to this forensic's fix surface — a separate, already-documented gap) |

## 2. Existing temporal signals audit

Four independent real, position-timestamped evidence classes already exist
in production before any new work:

1. **`audio_silence_interval`** — the most mature: ffmpeg-measured, confidence-scored, consumed today by three separate authorities at real timestamps (BoundaryEngine's edge/interior trims, AttemptReconstructor's measured-pause transition boundary).
2. **Four `local_performance` candidate kinds** — MediaPipe-measured per-frame-delta onsets (camera disengagement, facial expression shift, body reset, hand-motion reset), real RAW-absolute timestamps, consumed today by `performance_confirmation.py` (promotion), `take_judge._interior_events` (penalty), `perceptual_watch_listen._reset_debris_at_edges` (post-render routing).
3. **Confirmed `wrong_take`/`retry_setup`** — a derived, coarser-grained (take/edge, not instant) promotion of (1) into an editorial-usable event, consumed by the retry-authority chain (D-109–D-113's own subject matter).
4. **Word-level ASR envelope** — the most precise positional evidence in the system (`words[i].start/end`), already used as BoundaryEngine's hard floor for audio-edge trims and available on every take/attempt (fused attempts concatenate `words` verbatim in `_merge_attempt`).

No signal today expresses the relationship BETWEEN classes (2)/(1) and (4) —
i.e., nothing currently asks "did this visual/audio event's onset fall
before, during, or after the measured word envelope?" That comparison is
computable today from data that already exists; it is simply not computed
anywhere.

## 3. Media signal aggregation-loss mapping

Two independent, additive loss points, both duration-weighted-mean
collapses of the SAME underlying event stream:

- **`local_performance.py::apply_local_performance_to_takes`** — the primary
  loss point. Every frame/event inside a take's `[start, end]` window is
  reduced to one scalar per `MediaSignals` field via a duration-weighted
  blend against the take's prior (base) value, for 5 of the 7 fields it
  touches; `visual_fumble` and `distraction_risk` instead take `max()` across
  members — these two partially escape dilution (a brief severe event is not
  washed out by a long clean take) but even they lose WHERE within the span
  the event happened.
- **`attempt_reconstruction._merge_signals`** — a SECOND, compounding
  collapse when raw ASR-level takes are fused into a paragraph-level
  "attempt": it duration-weights each member's ALREADY-aggregated
  `MediaSignals` scalar again (mirroring the same `weighted()`/`max()` split).

Net effect: by the time `take_judge.score_take` runs, 7 of 12 `MediaSignals`
fields are two collapses removed from the original event stream for a fused
attempt, one collapse removed for a lone take. Critically, **the underlying
raw events are never deleted** — they persist independently in
`WholeVideoContext.sources[].events` for the whole run, so the loss is a
property of the scalar `MediaSignals` representation specifically, not of the
system's total evidence. Every consumer that already reasons positionally
(§2, items 1–3) does so by going around `MediaSignals`/`score_take` straight
to `WholeVideoContext` — exactly the pattern a fix should extend, not
reinvent.

## 4. Target Entry / Delivery / Exit model (measured, not fixed-duration)

Per this task's explicit constraint, no fixed duration (no "ENTRY = first 0.5
s") is proposed. All three zones are defined off evidence already computed
elsewhere in the pipeline for every take/attempt:

- **DELIVERY** = `[first_word_start, last_word_end]` — the take's (or fused
  attempt's) own word envelope, already present on every `CandidateTake` via
  `words` and already concatenated correctly across fused members by
  `_merge_attempt`. No new measurement.
- **ENTRY** = `[take.start, first_word_start)` — extended, when visual
  evidence exists, to the end of any reset/disengagement event that
  overlaps `take.start` and ends before `first_word_start` (the creator
  visibly settling in before speaking). Absent such evidence, ENTRY is
  exactly the audio gap already measured by
  `boundary_engine_pass.tighten_selected_audio_edges`.
- **EXIT** = `(last_word_end, take.end]` — symmetric: extended to cover any
  reset/disengagement event beginning at/after `last_word_end`.

This is a boundary DEFINITION, not new instrumentation: every timestamp it
references (`words[i].start/end`, reset/disengagement event `start`/`end`)
already exists today.

## 5. Delivery-span definition — what's already available

The word envelope is already the most precise "delivery span" measurement in
the system and needs no new provider: `first_word_start`/`last_word_end` are
literally the values `boundary_engine_pass.tighten_selected_audio_edges`
already uses as the hard floor it will never trim past. For a fused
multi-member attempt, `_merge_attempt` already concatenates every member's
`words` tuple, so the same computation (`min(w.start for w in words)`,
`max(w.end for w in words)`) is valid unchanged. A finer-grained
"sub-span-within-the-words is-this-part-of-the-claim" model does not exist
(the closest proxies — `complete_idea` and the semantic label — apply to the
WHOLE take, not a sub-range) and is out of scope here: entry/exit reasoning
only needs the take/attempt's own edges, where the word envelope already
suffices.

## 6. Position-aware signal targets — recommendation (not implemented)

Recommend, for a future bounded implementation, computing — at the exact
point `apply_local_performance_to_takes` currently aggregates, ADDITIVELY,
never replacing the existing scalar fields — three derived values per
take/attempt, reusing the EXISTING reset/break event vocabulary
(`_RESET_KINDS`/`_BREAK_KINDS` already defined in `performance_confirmation.py`
and `perceptual_watch_listen.py`) and the EXISTING word envelope (§5):

1. `earliest_defect_event_offset_from_delivery_end_sec` — earliest
   reset/break event start, expressed relative to `last_word_end` (negative =
   began before the words finished → CASE B territory).
2. `latest_settle_event_offset_from_delivery_start_sec` — symmetric, for
   ENTRY.
3. `defect_overlaps_delivery_span: bool` — whether the earliest disqualifying
   event's `[start, end]` intersects `[first_word_start, last_word_end]` —
   the concrete, measurable form of D-111's CASE A/B/C test.

These are recommendations only. No threshold for "disqualifying" is proposed
here (constraint §15) — that decision belongs to whichever authority
(BestTake or Boundary) is authorized to consume the new fields.

## 7. Event-vs-aggregate doctrine

Recommend an explicit, additive doctrine rather than a parallel system: the
current `MediaSignals` scalar fields remain the default-safe, backward-
compatible representation (unchanged formula, unchanged tests). A new,
optional, per-take/attempt structure — carrying the already-surviving raw
positioned events (§2) plus the measured delivery span (§5) — is exposed
alongside it for any consumer that needs position, exactly mirroring the
pattern `take_judge.delivery_cleanliness_evidence` already established for
audio events (it reads raw `WholeVideoContext` events directly, not the
collapsed `MediaSignals`). Extending that SAME precedent to the four visual
event kinds is the minimal doctrine-consistent step — not a second pipeline.

## 8. BestTake ownership (CASE B — delivery-overlapping defect)

BestTakeResolver/DeliveryScorer (`take_judge.py`) should own any
reset/break/silence event whose `[start, end]` intersects the measured
delivery span (§4/§5) — i.e., happens WHILE the creator is delivering the
idea. This is a usability/quality PENALTY (score down), never a trim, since
trimming inside the delivery span would cut words. `delivery_cleanliness_
evidence`'s existing fixed-0.35s-margin interior penalty is the direct
precedent and template; the recommended future step (not implemented here)
is retargeting its interior boundary from the fixed margin to the real word
envelope where doing so does not regress currently-passing fixtures.

## 9. Boundary ownership (CASE A — pre/post-delivery edge defect)

BoundaryEngine (`boundary_engine_pass.py`) should own any reset/break/
silence event entirely OUTSIDE the delivery span (before `first_word_start`
or after `last_word_end`) — recording-process slack. This is a TRIM, never a
rejection, consistent with `PHYSICAL_OWNERSHIP_CONTRACT`'s existing ENTRY row,
which already lists "multimodal reset evidence (edge-only trim)" as evidence
for entry. **Concrete gap found:** the contract table already anticipated
this in its documentation, but `tighten_selected_audio_edges` — the only
function currently implementing that contract row — consumes ONLY
`audio_silence_interval` events; it does not yet read the four visual event
kinds at all. The multimodal half of BoundaryEngine's own documented
contract is therefore currently undelivered code, not merely unoptimized
code.

## 10. Pimples C hypothesis — timing answer

The directive asks whether rapid movement begins before/during/after
semantic completion, or whether evidence is missing. The honest, bounded
answer, without launching a RAW: **the mechanism to answer this precisely
already exists and runs unconditionally** (`analyze_local_performance` is
called without a feature flag at `flow_b.py:198` for every source), so IF the
pimples-C source produced a detectable rapid-movement signal, its onset
timestamp is real and already sitting in that run's
`WholeVideoContext.sources[].events`. Answering it for the SPECIFIC pimples
region requires comparing that event's `start` against the take's own
`words[-1].end` (§4/§5) in that run's diagnostics artifact — a concrete,
one-comparison check, but one this task is not authorized to perform (no RAW
diagnostics for that exact run are cached locally, and this task is bounded
to no new paid compute / no RAW). This forensic's contribution is narrowing
the open question from "does timing evidence exist?" (no — it does) to "what
is the value of one already-computed comparison for one already-completed
run?" (an artifact-inspection task, not an architecture question).

## 11. Watch+Listen relationship

`perceptual_watch_listen.py` is architecturally downstream and diagnostic-
only (D-098 §4: never edits Selection/Boundary/render). Its
`_reset_debris_at_edges` capability already performs a position-aware
mapping of the same four visual event kinds onto the RENDERED timeline,
using a fixed `EDGE_DEBRIS_WINDOW_SEC = 0.35` — i.e., it already answers a
narrower version of the entry/exit question (was there reset debris at the
rendered edge?), but purely for post-render routing (`ROUTE_BOUNDARY`), never
for pre-render BestTake scoring. The recommendation in §6–§9 is upstream of
Watch+Listen and does not change its contract; if BestTake/Boundary become
position-aware pre-render, Watch+Listen's post-render edge-debris finding
should simply fire less often (the defect having already been penalized or
trimmed upstream) — it remains the same safety net, unweakened.

## 12. Default-signal re-audit

No change from the existing D-098 dataflow map (`docs/CUTSELL_
PERCEPTION_UNDERSTANDING_DATAFLOW_MAP.md` Gap 2): 5 of 12 `MediaSignals`
fields (`audio_quality`, `framing_quality`, `product_visibility`,
`continuity`, `delivery_energy`) remain permanently default because
`visual_provider=None` is still hardcoded in `brain_runtime.py`. This is
orthogonal to this forensic's subject (the 7 `local_performance`-sourced
fields, which ARE real by default) and remains open, separately-tracked
work.

## 13. Minimum future implementation shape (not authorized here)

If and when authorized: (a) add the three derived fields from §6 at the
existing `apply_local_performance_to_takes` call site, additive to current
`MediaSignals` fields, backward compatible; (b) extend `_merge_signals` to
carry them across fused attempts without dilution (e.g. `max()`/earliest, the
same pattern already used for `visual_fumble`/`distraction_risk`); (c) wire
`tighten_selected_audio_edges` (or a sibling function under the same
BoundaryEngine module) to also consume the four visual event kinds for
CASE A edge trims, closing the gap in §9; (d) let `take_judge.score_take` or
`delivery_cleanliness_evidence` consume `defect_overlaps_delivery_span` for
CASE B usability penalties, closing the gap in §8. No threshold values are
proposed here — that is implementation-time work requiring its own targeted
tests and offline qualification under a future, separately-authorized
directive.

## 14. Required observability (for a future implementation)

Any future implementation must record, per take/attempt: the computed
delivery span (`first_word_start`, `last_word_end`), the raw event(s) that
determined each derived field (kind, start, end, confidence), and which
authority (BestTake vs Boundary vs neither) consumed the result and what
action it took — mirroring the existing diagnostic shape already used by
`boundary_engine_pass`'s `audio_edge_rows` and
`take_judge.delivery_cleanliness_evidence`'s existing penalty rows, so a
future forensic can trace CASE A/B/C attribution the same way D-097's
reconciliation traces dead-air findings.

## 15. No-threshold-invention constraint

Honored. No new numeric threshold is introduced or recommended as final in
this document; §6's three fields are defined as measurements, not gates.

## 16. No-fallback-implementation constraint

Honored. No bounded multimodal fallback arbiter (D-111 MEDIUM-confidence
mode) was designed, implemented, or invoked. This forensic's subject
(entry/delivery/exit position-awareness) is a HIGH-confidence, deterministic
measurement question — comparing existing timestamps — not a case requiring
arbitration.

## 17. No-engine-change constraint

Honored. This task read `cutsell_worker/local_performance.py`,
`take_judge.py`, `attempt_reconstruction.py`, `boundary_engine_pass.py`,
`perceptual_watch_listen.py`, `whole_video_analysis.py`,
`performance_confirmation.py`, `flow_b.py` (targeted), `contracts.py`
(targeted), plus the D-098 canon and D-113 decision entry, and wrote only
this document. `git diff --stat` at the end of this task shows zero changes
under `cutsell_worker/`, `tests/`, or any provider/threshold/render/UI file.
