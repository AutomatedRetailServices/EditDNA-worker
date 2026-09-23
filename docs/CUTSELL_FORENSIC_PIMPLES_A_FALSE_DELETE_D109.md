# CutSell Forensic: Pimples Candidate A False-Delete (D-109)

**Status: OFFLINE INVESTIGATION ONLY. No `cutsell_worker/*.py` file was
modified to produce this document. No RAW was launched, no paid provider
was called.**

**Branch:** `feature/runpod-pod-on-demand`. **HEAD:** `c689c89`, clean
tree, unchanged throughout this investigation.

**Source RAW:** `34113549497` (D-108's real-media qualification run),
job log decoded to
`scratchpad/d108_qual_raw_log.txt`. All clip texts and diagnostic fields
quoted below are taken directly from that log.

---

## 1. Target candidate

**A** = `clip_c7c1e5ce8c3a68550ac2`, source range `[191.77, 197.52]`
(dur 5.75s): *"También me salían espinillas, era como un rush, una
alergia."* Physical result (D-108's own quality-ladder run):
`missing_delivery`, LEVEL_1, 5.75s, refinement `false_delete_outside_family`,
authority `CompositeResolver/PreResolverCleanup`.

## 2. Full evidence trail for A

| Field | Value |
|---|---|
| Transcript | "También me salían espinillas, era como un rush, una alergia." |
| Source range | 191.77–197.52 (5.75s) |
| `complete_idea` | Not surfaced in this diagnostic dump |
| Semantic label (governing chunk) | `"failed"`, confidence 0.85 (an earlier chunk shows `"alternate"`, confidence 0.8 -- label instability across chunks, the same class of issue already recorded as an open escalation at D-097.11/D-104) |
| `dense_semantic_failure_cluster` | `false` in the governing (chunk_index 4) record |
| `local_failure_corroborated` | `true` |
| `local_failure_reasons` | `["dense_physical_reset:7", "visual_fumble:0.85"]` (no `event:retry_setup:` entry in this summarized field) |
| `later_retry_replacement_id` | `null` |
| `later_retry_semantic_overlap` | `0.7` |
| `lexical_identity_passed` | `false` |
| `semantic_delete_recommended` | `true` |
| `delete_basis` | `"semantic_failed_plus_local_performance"` |
| `applied_delete` | **`false`** (confirmed in the FINAL printed diagnostic dump) |
| `replacement_candidate_clip_id_before_guard` | `clip_16fdfabcdbfc97bdf490` (= C) |
| `replacement_rejection_reason` | `"SEQUENCE_IDENTITY_BELOW_THRESHOLD"` |
| `sequence_identity` | `0.4152542372881356` |
| `sequence_identity_threshold` | `0.52` |
| A vs B, A vs C in `semantic_idea_equivalence` / `grouping_safety_budget` | **Absent entirely** -- A never appears as a candidate pair in either the cross-group (60 candidates, 14 checked) or within-group (7 weak pairs, 0 checked) reconciliation blocks |
| Final selection membership | **Not present in the delivered MP4** (physical ladder row 82: `missing_delivery`, LEVEL_1) |
| Later recovery opportunity | None observed in this run's diagnostics; per non-destructive doctrine the source footage itself is untouched (immutable RAW), so a future run/authority could still recover it, but nothing in the current pipeline re-offers it |

## 3. A exact decision path

```
candidate A (audience-facing beat, complete sentence)
  -> hybrid_session_cleanup.apply_hybrid_session_cleanup (base stage):
       semantic label = "failed" (0.85)
       local performance evidence: dense_physical_reset:7, visual_fumble:0.85
         -> corroborated_failed_delete = True
       delete_basis = "semantic_failed_plus_local_performance"
       semantic_delete_recommended = True
       mechanical_delete = False (only "micro_failed_plus_local_performance" is mechanical)
       applied_delete = False   <-- CORRECT per D-081: A remains "kept"
     replacement search (complete_retry_identity_guard, consulted for the
     SEPARATE "semantic_failed_plus_later_overlapping_complete_retake" basis):
       candidate replacement = C
       sequence_identity = 0.415 < threshold 0.52
       replacement_rejection_reason = SEQUENCE_IDENTITY_BELOW_THRESHOLD
       -> retry_replaced_failed_delete = False (this specific basis never applies)
  -> [A is still "kept" at this point, decisions[].applied_delete = false]
  -> composite_resolver.py's 19-hook chain runs (hybrid_session_cleanup's
     result is threaded through each hook's own install_*() wrapper)
       -> hook "hybrid_retry_winner_authority" (enforce_proven_retry_winners):
            reads semantic_decisions (label/confidence only) + raw
            whole_video_context events, NOT complete_retry_identity_guard's
            sequence_identity verdict
            failed = A (label "failed", confidence 0.85 >= 0.80 floor)
            IF retry_setup_confidence(A, context) >= 0.84 (UNCONFIRMED from
              available diagnostics -- A's own local_failure_reasons list
              carries no "event:retry_setup:" entry, but this function reads
              raw whole_video_context events directly, a superset of what
              that summarized field reports)
            candidate winners after A, same source, gap <= 20s:
              C: label "winner" 0.92 >= 0.90 floor, gap 1.36s
              _same_retry_attempt(A, C) -- OWN, DIFFERENT shared-content-
                token test (reused/verified offline against the real text):
                shared = {alergia, era, espinillas, salían, también} (5),
                failed_coverage 0.833, winner_coverage 0.333
                -> "same retry attempt" = TRUE (>=4 shared, max coverage >=0.45)
            -> IF the retry_setup gate also passes: A is added to
               removed_ids and returned in "survivors" WITHOUT it, i.e.
               excluded from `kept` directly, NEVER touching
               decisions[].applied_delete
  -> grouping (session_boundaries.safe_group_takes_by_sessions) never
     receives A at all -- confirmed absent from every candidate-pair list
  -> render: A is missing from the delivered MP4
```

## 4. First delete-authorizing branch

**Not `apply_hybrid_session_cleanup` itself** (it correctly does not apply
the delete for this `delete_basis`, per D-081's own preserved rule). The
first branch in the active pipeline that CAN independently remove a
`kept` candidate from the take list on this exact evidence shape is
`cutsell_worker/hybrid_retry_winner_authority.py::enforce_proven_retry_winners`
(installed as hook #9 of 19 in `composite_resolver.py`'s
`_CHAIN_SPEC`, called via `hybrid_session_cleanup.apply_hybrid_session_cleanup`'s
monkeypatched wrapper before grouping ever runs). Its `_same_retry_attempt`
check independently returns `True` for the real (A, C) pair when tested
offline against the actual transcript text (Section 3). Whether this
specific hook is the one that fired in the live RAW is **UNCONFIRMED**
(the one missing link is `retry_setup_confidence(A, context) >= 0.84`,
which requires the raw `whole_video_context` event stream, not available
in the captured diagnostics) -- but it is the single strongest, most
directly evidenced candidate, and no other hook in the chain was found
with a matching "drop a failed attempt in favor of a later winner" role
that operates independently of `complete_retry_identity_guard`'s
sequence-identity verdict.

## 5. Performance-bad vs replaceable distinction in current code

The distinction the doctrine calls for **is honored at exactly one
place** and **not honored at a second place that can independently
override it**:

- **Honored:** `hybrid_session_cleanup.py` (D-081, `_SEMANTIC_JUDGMENT_DELETE_BASES`)
  treats `semantic_failed_plus_local_performance` as `PERFORMANCE_BAD`
  evidence only -- `applied_delete = mechanical_delete` (False for this
  basis), the candidate stays `kept`, and the module's own docstring
  states the survival decision belongs to "grouping / take_judge_groups /
  claim extraction / BestTake dominance / Unified Resolver / StoryValidator
  ... which alone decide."
- **Honored, separately:** `complete_retry_identity_guard.py`'s
  `sequence_identity` check independently asks "is C a valid
  REPLACEMENT for A" and correctly answers NO (0.415 < 0.52) --
  this is exactly a `SEMANTICALLY_REPLACEABLE` test, kept distinct from
  the performance judgment.
- **NOT honored:** `hybrid_retry_winner_authority.py::enforce_proven_retry_winners`
  re-asks a conceptually identical question ("is this failed attempt
  superseded by a later winner covering the same communication attempt?")
  using its OWN, looser, independently-computed shared-content-token test
  (`_same_retry_attempt`), with **no reference at all** to
  `complete_retry_identity_guard`'s stricter, already-computed rejection
  for the exact same (A, C) pair. If its own gates are satisfied, it
  removes A from `kept` directly (bypassing the `delete_basis`/
  `applied_delete` fields entirely), independently of whether a valid
  semantic replacement was ever established.

## 6. Valid replacement exists? YES / NO / UNCERTAIN

**NO_VALID_REPLACEMENT**, per the pipeline's own strictest, already-computed
evidence: `complete_retry_identity_guard`'s `sequence_identity = 0.415`
is below its own `0.52` threshold for treating C as a valid replacement
for A. No other candidate (B) was ever proposed as a replacement for A in
the recovered diagnostics. Per the directive's own instruction not to
infer substitution merely from shared topic, and reusing only the
strongest existing evidence: the engine's own strict-identity guard
already answered this question, and the answer is NO.

## 7. Why `SEQUENCE_IDENTITY_BELOW_THRESHOLD` did not prevent deletion

Because that rejection is scoped to exactly one delete pathway inside
`hybrid_session_cleanup.py` (the `retry_replaced_failed_delete` /
`semantic_failed_plus_later_overlapping_complete_retake` basis). It is
recorded as observability (`replacement_rejection_reason`) but is never
read by any other function (the same pattern already documented for
`replacement_candidate_clip_id_before_guard` at D-072: "additive
observability only ... never read by any decision below this point or
anywhere else in the pipeline"). `hybrid_retry_winner_authority.py`
computes its own independent verdict on the identical question and has
no mechanism to consult, or be overridden by, the first guard's answer.

## 8. First correct decision

`hybrid_session_cleanup.apply_hybrid_session_cleanup` + the
`complete_retry_identity_guard` sequence-identity check: correctly
classifies A's evidence as performance-only, correctly declines to
irreversibly delete it, and correctly rejects C as a valid replacement.

## 9. First wrong decision

`hybrid_retry_winner_authority.enforce_proven_retry_winners`: independently
re-derives "same retry attempt" via a looser, uncoordinated test and (if
its own `retry_setup_confidence` gate also passes -- unconfirmed) removes
A from the take list without ever consulting the stricter guard that had
already rejected the same substitution.

## 10. Authority collision

**YES -- the exact D-096-class collision:** two authorities
(`complete_retry_identity_guard`'s sequence-identity threshold and
`hybrid_retry_winner_authority`'s shared-content-token test) independently
answer the SAME question ("is this a valid retry replacement / same
communication attempt?") with different evidence and different
thresholds, with no cross-authority communication. The stricter, correct
answer from the first is silently capable of being overridden by the
looser answer from the second. **LATER OVERRIDE** (not "no override") --
structurally possible and strongly, though not conclusively, evidenced as
having actually fired in RAW `34113549497`.

## 11. Root cause

`hybrid_retry_winner_authority.py`'s `_same_retry_attempt` re-implements
a "same retry attempt" judgment independently of, and with a looser
threshold than, `complete_retry_identity_guard`'s own `sequence_identity`
check -- allowing a `PERFORMANCE_BAD` candidate with `NO_VALID_REPLACEMENT`
(per the stricter, already-computed guard) to still be removed from the
edit on the authority of the looser, uncoordinated check. This directly
violates the canonical doctrine's ordering: meaning/replacement validity
must gate removal; performance quality alone must not.

## 12. Minimum safe fix location (not implemented)

The most surgical fix consistent with "meaning first, performance second"
and with not touching BestTake/grouping/visual work: before
`enforce_proven_retry_winners` removes a `failed`-labelled candidate in
favor of a later `winner`, it should check whether
`complete_retry_identity_guard`'s own sequence-identity evaluation for
that exact (failed, winner) pair (if already computed this run, as it is
here) recorded a `replacement_rejection_reason` -- and if so, decline to
remove. This reuses existing evidence, adds no new heuristic, and is
scoped to exactly the collision identified in Section 10. No code was
changed to implement this in this task.

## 13. Expected pimples effect (if fixed, not measured)

A would remain in `kept` at this stage, reach grouping/BestTake as its
own realization (a singleton or a distinct-content pairing with B, per
D-108's own evidence that A is NOT equivalent to B), and become eligible
for delivery instead of silently disappearing. This does not resolve the
separate, still-open B-vs-C BestTake winner question.

## 14. Risks / regressions

`hybrid_retry_winner_authority.py`'s own docstring names its exact
intended positive case (Human Gold's fumble-plus-retake shape) -- any fix
must not reintroduce that regression. Any change here is regression-
sensitive to whichever fixture suite already locks this hook's behavior
and must be tested against it before any implementation, which this task
does not do.

## 15. Confirmations

- **NO CODE CHANGED.** No `cutsell_worker/*.py` file was modified. This
  document and the accompanying D-109 decision entry are the only files
  this investigation produced.
- **NO RAW / PROVIDER / INFRA.** No Video00 RAW was launched, no paid
  provider was called, no infrastructure was touched.
