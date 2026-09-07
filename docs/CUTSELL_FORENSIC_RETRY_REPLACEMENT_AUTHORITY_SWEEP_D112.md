# CUTSELL — Retry Replacement Authority Coordination Sweep (D-112)

**Status: OFFLINE FORENSIC ONLY. No `cutsell_worker/*.py` file was modified
to produce this document. No RAW/provider/S3/Modal/RunPod/UI work was
performed.**

**Branch/HEAD at time of writing:** `feature/runpod-pod-on-demand` @
`71cbd06` (clean tree confirmed before starting).

**Purpose.** D-110's real-media qualification (RAW `34123511687`) proved
that the authority collision D-109/D-110 fixed at `hybrid_retry_winner_
authority` (chain hook #8) also exists, uncoordinated, at an EARLIER hook
(`hybrid_retry_completion_integrity`, chain hook #2). This sweep traces
every pre-hook-#8 authority in the real `composite_resolver.py` take-level
chain to determine whether this is one additional isolated collision or a
repeated architectural defect.

---

## 1. Actual pre-hook-#8 chain order

Read directly from `cutsell_worker/composite_resolver.py`'s `_CHAIN_SPEC`
and module docstring ("Canonical order", verbatim historical
`__init__.py` install order). Execution order matches installation order
(each `install_*()` wraps the *previous* wrapper as its own `original`,
so post-processing runs in the SAME order hooks are installed/listed):

0. `hybrid_session_cleanup.apply_hybrid_session_cleanup` (base — LLM
   classify + corroborated delete; the ONLY authority besides hook #8
   that consults `complete_retry_identity_guard`)
1. `semantic_fragment_guard`
2. `hybrid_retry_completion_integrity` — **PROVEN COLLISION (D-109/D-110's
   original finding, reconfirmed here)**
3. `hybrid_story_guard`
4. `hybrid_alternate_integrity`
5. `hybrid_cross_group_retry_integrity` — **PROVEN COLLISION (NEW finding,
   this sweep)**
6. `incomplete_bridge_retry_authority`
7. `hybrid_failed_continuation_integrity`
8. `hybrid_retry_winner_authority` — **D-110's fix lives here**
(9-19: `hybrid_gold_reconciliation` through
`hybrid_semantic_conflict_arbitration` — out of this sweep's scope, all
downstream of the fixed hook)

---

## 2. Complete hook-by-hook authority table

Grep-confirmed: only `hybrid_session_cleanup.py` and `hybrid_retry_
winner_authority.py` import anything from `complete_retry_identity_
guard.py` anywhere in `cutsell_worker/`. All 6 other pre-hook-#8 hooks
below are blind to it by construction.

### Hook 1 — `semantic_fragment_guard`

- MAY REMOVE MEMBERSHIP? YES (micro/open/repetition-pathology fragments
  only)
- MAY SUBSTITUTE ONE TAKE FOR ANOTHER? NO (deletes standalone debris; the
  one exception, `_alternate_micro_in_failure_cluster`, only checks a
  nearby `winner`'s duration ratio, never its text content)
- MAY CLAIM A WINNER COVERS A LOSER? Narrowly — only for a `label==
  "alternate"`, `duration<=1.6s`, `<=3 token` micro fragment beside a
  failed neighbor and a much longer (`>=5x` duration) winner
- REPLACEMENT / SAME-ATTEMPT EVIDENCE USED: structural (duration, token
  count, sentence-end punctuation, repetition detector), not content
  overlap with a specific proposed replacement
- CONSULTS `complete_retry_identity_guard` REJECTION? NO
- CONSULTS THE EXACT DIRECTIONAL X→Y REJECTION? NO
- CAN CONTRADICT A PRIOR REJECTION? Only theoretically, and only for a
  micro (<=1.6s/<=3 token) fragment — real audience-facing propositions
  this small are rare and separately protected by
  `PROTECTED_POLARITY_FRAGMENT` when they carry polarity
- REAL D-110 RAW EVIDENCE OF ACTIVATION? Not against A; not checked
  against every other clip this sweep (out of the sweep's bounded scope)
- **RISK CLASS: POTENTIAL_COLLISION** (very narrow surface — size gate
  makes it structurally unlikely to intersect a genuine X→Y replacement
  question, but the gate is a duration/token heuristic, not a
  proposition-identity check)

### Hook 2 — `hybrid_retry_completion_integrity`

- MAY REMOVE MEMBERSHIP? YES
- MAY SUBSTITUTE ONE TAKE FOR ANOTHER? YES (`_safe_failed_retry`,
  `_safe_full_alternate_retry`)
- MAY CLAIM A WINNER COVERS A LOSER? YES — its own diagnostic reason IS
  literally `"semantic_failed_cross_group_retry_covered"` /
  `"semantic_reset_backed_full_alternate_retry"`
- REPLACEMENT / SAME-ATTEMPT EVIDENCE USED: shared content-token count +
  coverage ratio, `_same_opening` 2-token match, `complete_idea`
  asymmetry — an independently-coded "is this the same retry attempt"
  test, structurally similar to (but coded separately from)
  `hybrid_retry_winner_authority`'s own `_same_retry_attempt`
- CONSULTS `complete_retry_identity_guard` REJECTION? **NO**
- CONSULTS THE EXACT DIRECTIONAL X→Y REJECTION? **NO**
- CAN CONTRADICT A PRIOR REJECTION? **YES**
- CAN CONTRADICT IT ONLY INDIRECTLY? NO — directly: it removes X and
  records Y as `winner_clip_id` in the same call
- REAL D-110 RAW EVIDENCE OF ACTIVATION? **YES — PROVEN.** RAW
  `34123511687`: removed candidate A
  (`clip_593f2f22cb02fca4e346`) in favor of C
  (`clip_390e5f221849f30bb34a`), diagnostic `{"clip_id":
  "clip_593f2f22cb02fca4e346", "reason":
  "semantic_failed_cross_group_retry_covered", "winner_clip_id":
  "clip_390e5f221849f30bb34a"}`, contradicting the SAME run's own
  `complete_retry_identity_guard` rejection of that exact pair
  (`SEQUENCE_IDENTITY_BELOW_THRESHOLD`, 0.4153 < 0.52)
- **RISK CLASS: PROVEN_COLLISION**

### Hook 3 — `hybrid_story_guard`

- MAY REMOVE MEMBERSHIP? NO (it is a restorer — operates on already-
  `result.deleted` candidates, deciding what to bring BACK)
- MAY SUBSTITUTE ONE TAKE FOR ANOTHER? Indirectly, in the suppression
  direction only: `_covered_by_kept_delivery` DECLINES to restore a
  deleted incomplete `failed` candidate (confidence>=0.90) when an
  EARLIER-starting kept delivery already covers >=50% of its tokens and
  preserves its critical (numeric/negation) tokens
- MAY CLAIM A WINNER COVERS A LOSER? Only in that narrow non-restoration
  sense — it never actively deletes a currently-kept candidate
- REPLACEMENT / SAME-ATTEMPT EVIDENCE USED: token-coverage ratio (>=0.50)
  + critical-token subset check, own independent computation
- CONSULTS `complete_retry_identity_guard` REJECTION? NO
- CONSULTS THE EXACT DIRECTIONAL X→Y REJECTION? NO
- CAN CONTRADICT A PRIOR REJECTION? Only by DECLINING to undo one — it
  can never be the FIRST authority to contradict a rejection (nothing to
  contradict; it only inherits the `result.deleted` state prior hooks
  already produced) but it can perpetuate an earlier hook's false delete
  by refusing to restore it. On RAW `34123511687`, A's cleanup-stage
  confidence (0.8) was below this suppression path's own 0.90 floor, so
  the suppression did not explicitly fire for A — but A was still passed
  into `restore_unique_story_coverage` as `eligible_deleted` and was NOT
  restored (a missed-opportunity finding, not itself a rejection
  contradiction)
- REAL D-110 RAW EVIDENCE OF ACTIVATION? Fired this run, but on a
  DIFFERENT clip (`clip_ea807447f65a2db70465`,
  `"incomplete_failed_retry_covered_by_kept_delivery"`) — not proven to
  have touched a `complete_retry_identity_guard`-rejected pair this run
- **RISK CLASS: POTENTIAL_COLLISION** (restoration-suppression direction
  only; cannot be the FIRST hook to contradict a rejection, but can be
  the LAST hook that fails to un-do one)

### Hook 4 — `hybrid_alternate_integrity`

- MAY REMOVE MEMBERSHIP? YES
- MAY SUBSTITUTE ONE TAKE FOR ANOTHER? YES
- MAY CLAIM A WINNER COVERS A LOSER? YES — removes an `alternate`-labelled
  candidate beside a `winner`, `winner_clip_id` recorded directly in its
  diagnostic
- REPLACEMENT / SAME-ATTEMPT EVIDENCE USED: shared content-token count +
  coverage ratio (two separate threshold sets depending on whether the
  alternate is before or after the winner), syntactic-openness test — an
  independently-coded, THIRD version of "is this covered by that winner"
- CONSULTS `complete_retry_identity_guard` REJECTION? NO
- CONSULTS THE EXACT DIRECTIONAL X→Y REJECTION? NO
- CAN CONTRADICT A PRIOR REJECTION? YES, structurally — but ONLY for
  candidates Hybrid itself already labelled `alternate` (never `failed`),
  which narrows the practical overlap with `complete_retry_identity_
  guard`'s consultation path today (that path is entered from a `failed`-
  labelled candidate at the cleanup stage)
- CAN CONTRADICT IT ONLY INDIRECTLY? N/A (label-gated, not indirect)
- REAL D-110 RAW EVIDENCE OF ACTIVATION? No diagnostic entry for
  `hybrid_alternate_integrity` was found in this run's captured
  diagnostics window
- **RISK CLASS: POTENTIAL_COLLISION** (same defect shape as hooks 2 and
  5, gated to `alternate`-labelled candidates)

### Hook 5 — `hybrid_cross_group_retry_integrity`

- MAY REMOVE MEMBERSHIP? YES
- MAY SUBSTITUTE ONE TAKE FOR ANOTHER? YES
- MAY CLAIM A WINNER COVERS A LOSER? YES — its own diagnostic reason is
  literally `"cross_group_semantic_retry_covered_by_authoritative_
  delivery"`, `strongest_peer_clip_id`/`peer_clip_ids` recorded directly
- REPLACEMENT / SAME-ATTEMPT EVIDENCE USED: single-peer or same-side
  contiguous-chain content-token coverage (duration-tiered thresholds) +
  critical-token (negation/number) subset preservation — a FOURTH
  independently-coded "is this covered" test, operating on `failed` OR
  `alternate` labels at confidence >=0.75
- CONSULTS `complete_retry_identity_guard` REJECTION? **NO**
- CONSULTS THE EXACT DIRECTIONAL X→Y REJECTION? **NO**
- CAN CONTRADICT A PRIOR REJECTION? **YES**
- CAN CONTRADICT IT ONLY INDIRECTLY? NO — directly
- REAL D-110 RAW EVIDENCE OF ACTIVATION? **YES — PROVEN, a SECOND real
  contradiction on the SAME RAW** `34123511687` (unrelated to pimples):
  removed `clip_a3260a4974b01a17a628` ("Al terminar mi contrato, le pedí
  a mi ginecóloga.") in favor of `clip_2dfc08fc82f6830b17e5` ("Al
  terminar mi contrato, cambié de ginecóloga y le pedí que me hiciera un
  test d..."), diagnostic `{"clip_id": "clip_a3260a4974b01a17a628",
  "reason": "cross_group_semantic_retry_covered_by_authoritative_
  delivery", "strongest_peer_clip_id": "clip_2dfc08fc82f6830b17e5",
  "coverage": 1.0, "shared_union": 4, "critical_preserved": true}` — this
  run's own `hybrid_session_cleanup` decisions record the EXACT SAME
  (X=`clip_a3260a4974b01a17a628`, Y=`clip_2dfc08fc82f6830b17e5`) pair
  rejected via `complete_retry_identity_guard`:
  `"replacement_candidate_clip_id_before_guard":
  "clip_2dfc08fc82f6830b17e5", "replacement_rejection_reason":
  "SEQUENCE_IDENTITY_BELOW_THRESHOLD", "sequence_identity":
  0.41081081081081083, "sequence_identity_threshold": 0.52"`. X stays
  deleted for the rest of the run (never restored by `hybrid_failed_
  soft_restore`, which restored a DIFFERENT clip,
  `clip_cac44ba0950aecab319a`, from the same composite step). No
  dedicated physical-ladder row names X specifically (its short, near-
  duplicate span appears absorbed into the same family's existing
  `BestTakeResolver`-attributed Level-1 rows — consistent with the
  already-documented finding that the ladder's attribution label is a
  generic heuristic, not proof of which module actually acted)
- **RISK CLASS: PROVEN_COLLISION**

### Hook 6 — `incomplete_bridge_retry_authority`

- MAY REMOVE MEMBERSHIP? YES (mutates `kept`/`deleted` directly via
  `dataclasses.replace`)
- MAY SUBSTITUTE ONE TAKE FOR ANOTHER? YES — narrow 3-tuple pattern
  (complete `early` kept, incomplete `bridge` deleted, complete `late`
  deleted, all physically adjacent in source order, gaps <=8s each)
- MAY CLAIM A WINNER COVERS A LOSER? YES, but inverted: it **restores**
  `late` and marks `early` superseded — the opposite direction from every
  other hook here (a LATER delivery supersedes an EARLIER one, not the
  reverse)
- REPLACEMENT / SAME-ATTEMPT EVIDENCE USED: **recomputes its own
  SequenceMatcher-based `_sequence_identity`** (same underlying metric
  `complete_retry_identity_guard` uses) at a DIFFERENT, LOWER threshold
  (`seq >= 0.42` here vs. the guard's `0.52`), plus a separate semantic-
  overlap ratio (`>=0.60`) and a number-preservation check
- CONSULTS `complete_retry_identity_guard` REJECTION? **NO**
- CONSULTS THE EXACT DIRECTIONAL X→Y REJECTION? **NO**
- CAN CONTRADICT A PRIOR REJECTION? **YES, and by the WORST mechanism of
  any hook in this sweep** — it does not merely ignore the guard's
  verdict, it independently RE-DERIVES the identical metric with a
  laxer threshold, which is exactly the anti-pattern D-110's own
  implementation spec explicitly forbade ("Do NOT recompute sequence
  identity with another threshold"). This hook predates D-110 and was
  never audited against that rule until now.
- CAN CONTRADICT IT ONLY INDIRECTLY? NO — directly, via its own duplicate
  sequence-identity computation
- REAL D-110 RAW EVIDENCE OF ACTIVATION? Not proven against A (A's real
  physical position has no genuine incomplete bridge fragment between it
  and C — B sits chronologically AFTER C, not between A and C, per the
  D-109 forensic's corrected timestamp finding) or against any other
  clip in this run's captured diagnostics window
- **RISK CLASS: POTENTIAL_COLLISION** (narrow 3-tuple structural gate,
  but the WORST-shaped defect of the seven — an independent
  re-derivation of the guard's own metric at a looser bar, not merely a
  different evidence type)

### Hook 7 — `hybrid_failed_continuation_integrity`

- MAY REMOVE MEMBERSHIP? YES (two functions: `collapse_failed_split_
  retry_continuations` removes a `failed` fragment + its immediate
  continuation; `suppress_selected_prefixes_with_failed_suffixes` removes
  an already-`winner`/`keep`-labelled candidate when a failed suffix
  chain follows it)
- MAY SUBSTITUTE ONE TAKE FOR ANOTHER? YES, both directions
- MAY CLAIM A WINNER COVERS A LOSER? YES — `"failed_split_retry_covered_
  by_authoritative_winner"` and `"selected_prefix_yields_when_immediate_
  suffix_fails_same_retry"` are both explicit winner-covers-loser claims
- REPLACEMENT / SAME-ATTEMPT EVIDENCE USED: combined-fragment content-
  token coverage + critical-token preservation — a FIFTH independently-
  coded coverage test
- CONSULTS `complete_retry_identity_guard` REJECTION? NO
- CONSULTS THE EXACT DIRECTIONAL X→Y REJECTION? NO
- CAN CONTRADICT A PRIOR REJECTION? YES, structurally, for either
  direction
- CAN CONTRADICT IT ONLY INDIRECTLY? NO — directly
- REAL D-110 RAW EVIDENCE OF ACTIVATION? No diagnostic entry for
  `hybrid_failed_continuation_integrity` was found in this run's
  captured diagnostics window
- **RISK CLASS: POTENTIAL_COLLISION**

### Hook 8 — `hybrid_retry_winner_authority` (for contrast; D-110 already fixed)

- MAY REMOVE MEMBERSHIP? YES
- CONSULTS `complete_retry_identity_guard` REJECTION? **YES, as of
  D-110** — via `_prior_replacement_rejections(session_diagnostics)`
- CONSULTS THE EXACT DIRECTIONAL X→Y REJECTION? YES (directional, keyed
  by the exact failed-clip-id → proposed-winner-clip-id pair)
- CAN CONTRADICT A PRIOR REJECTION? NO, as of D-110, for the pairs it
  itself would otherwise act on
- REAL D-110 RAW EVIDENCE OF ACTIVATION? Confirmed NOT triggered on RAW
  `34123511687` (A was already gone from `kept` by hook 2 before hook 8
  ran; its diagnostics key is entirely absent from this run)
- **RISK CLASS: SAFE** (for the collision this sweep is about — its own
  `_same_retry_attempt` test is unchanged and still independently-coded,
  but it now defers to the guard's verdict when one exists for the exact
  pair)

---

## 3. Real pimples A→C trace (all pre-hook-#8 hooks, execution order)

A = `clip_593f2f22cb02fca4e346` ("También me salían espinillas, era
como un rush, una alergia."). C = `clip_390e5f221849f30bb34a` (the
later, longer pimples realization). Strict evidence recorded this run:
`replacement_candidate_clip_id_before_guard: clip_390e5f221849f30bb34a`,
`replacement_rejection_reason: SEQUENCE_IDENTITY_BELOW_THRESHOLD`,
`sequence_identity: 0.4153`, `threshold: 0.52`.

| Hook | A enters hook? | C available? | C considered replacement/coverage? | A removed? | Reason | Consults strict A→C rejection? | If not removed here: could this hook have removed A under slightly different diagnostics? |
|---|---|---|---|---|---|---|---|
| 1. `semantic_fragment_guard` | YES (still in `kept`) | YES | NO — A is 12 tokens/5.75s, fails every micro-fragment size gate | NO | N/A | N/A | NO (A is far too long for this hook's structural gates to ever apply) |
| 2. `hybrid_retry_completion_integrity` | YES | YES | **YES** — `_safe_failed_retry` matched (A: `failed`, 0.8 conf; C: `winner`, shared tokens >=4, coverage/opening test passed) | **YES** | `semantic_failed_cross_group_retry_covered`, `winner_clip_id: clip_390e5f221849f30bb34a` | **NO** | — (this IS the removing hook) |
| 3. `hybrid_story_guard` | NO — A already in `result.deleted` by hook 2 | YES (as a peer, not evaluated for restoration purposes) | NO — A's confidence (0.8) sits below the `_covered_by_kept_delivery` suppression's 0.90 floor, so A was passed to `restore_unique_story_coverage` as eligible but that function itself declined to restore it (reason not surfaced in captured diagnostics) | Stays removed (not re-removed; already gone) | N/A (restoration attempted and failed, not a fresh removal) | NO | NOT APPLICABLE (restorer, not remover) |
| 4. `hybrid_alternate_integrity` | NO — A not in `kept` (still deleted) | — | NO — A's label is `failed`, not `alternate`; this hook only ever considers `alternate`-labelled candidates | Stays removed | N/A | N/A | NO (label mismatch, would not apply even if A were still present) |
| 5. `hybrid_cross_group_retry_integrity` | NO — A not in `kept` | — | NO — hook operates only on `kept`, A is absent | Stays removed | N/A | N/A | **YES, in principle** — A's label (`failed`, could exceed the 0.75 floor at some confidence readings this run) and C's `winner` label/confidence would satisfy this hook's own single-peer-coverage test (A's 12 content tokens vs C's much longer text plausibly clears the >=4-shared/>=0.53-coverage bar for a >6s candidate) — **UNPROVEN whether it actually would have, since A never reached this hook this run**, but structurally plausible |
| 6. `incomplete_bridge_retry_authority` | NO | — | NO — requires a 3-tuple (complete `early` KEPT, incomplete `bridge` DELETED, complete `late` DELETED) with `early` still in `kept`; A is not in `kept` so it cannot serve as `early` | Stays removed | N/A | N/A | NO this run (A already absent from `kept`); NOT APPLICABLE to the A→C shape regardless, since no real incomplete bridge fragment sits physically between A and C (B is chronologically after C, not between A and C, per the D-109 forensic's corrected real timestamps) |
| 7. `hybrid_failed_continuation_integrity` | NO | — | NO — both sub-functions require the candidate to be in `kept`; A is not | Stays removed | N/A | N/A | NO this run; structurally plausible only if A had an adjacent failed continuation fragment within 3s, which is not evidenced |
| 8. `hybrid_retry_winner_authority` (D-110) | NO — A not in `kept` | — | NO | Stays removed (already gone) | N/A | N/A (never reached) | — |

**FIRST hook that actually contradicted the rejection:** Hook 2,
`hybrid_retry_completion_integrity::_safe_failed_retry` (already known
from D-110's own qualification; reconfirmed here as the first in
execution order among ALL seven pre-hook-#8 hooks, not just the one
compared against hook 8).

**ALL later hooks that could independently do the same:** Hook 5,
`hybrid_cross_group_retry_integrity`, is PROVEN to independently
contradict a DIFFERENT recorded rejection this same run (the ob-gyn
pair). Hooks 1, 3, 4, 6, 7 are POTENTIAL_COLLISION by construction
(same missing consultation) but were not proven active against ANY
guard-rejected pair on this specific run.

**Can the same candidate→winner pair be re-decided more than once?**
**YES, architecturally.** Once hook 2 removes A in favor of C, A is gone
from `kept` for the rest of the chain — no later hook gets a SECOND
chance to re-litigate that SPECIFIC pair in this run's trace, because A
never re-enters `kept`. But the ob-gyn pair shows the general risk is
real: a candidate (`clip_a3260a4974b01a17a628`) that survived stage-1
cleanup (`applied_delete: false`, `delete_basis: "kept_fail_open"` in one
of its two per-session decision rows) was still removed later, by hook 5,
independently — i.e. the SAME clip was evaluated for replacement-safety
TWICE this run (once by stage-1 cleanup consulting the guard and
declining to delete, once by hook 5 which does not consult the guard and
did delete) with two different outcomes. This is the general shape of
the "re-litigation" risk the canonical question asks about, proven, not
hypothetical.

---

## 4. D-110 analysis

**D-110 remains conceptually correct.** Its rule (do not delete X in
favor of Y when this run's own `complete_retry_identity_guard` already
rejected that exact X→Y pair) is exactly right for the ONE hook it
covers, and its directional, non-global design (a rejection for A→C
never blocks C from competing with anything else) is proven sound by
this sweep — nothing here suggests loosening or generalizing that
specific directionality was wrong.

**What D-110 does NOT do, and was never asked to do:** cover any OTHER
hook. This sweep proves that gap is real (hook 2, hook 5 proven; hooks 1,
3, 4, 6, 7 potential) rather than hypothetical.

**Verdict: B — should become part of a shared chain-level replacement-
authority contract**, not stay local to `hybrid_retry_winner_authority`.
Patching hook 2 alone (the originally proposed next step) would leave
hook 5's PROVEN collision and hooks 1/3/4/6/7's POTENTIAL collisions
completely unaddressed, guaranteeing this exact forensic conversation
repeats per-hook, one bounded fix at a time, forever — precisely the
"rule proliferation" pattern D-111 §10.4 exists to prevent. (D-110 itself
is not refactored by this observation — this task makes no code change.)

---

## 5. Root architectural cause

**CutSell currently has MULTIPLE local authorities independently
re-deriving "same retry" / "winner covers loser" / "safe replacement",
using at least FIVE different evidence computations, and only TWO
authorities in the entire ~20-hook chain (`hybrid_session_cleanup`
stage 1, and `hybrid_retry_winner_authority` as of D-110) ever consult
the one authority (`complete_retry_identity_guard`) that actually
computes a strict, principled sequence-identity verdict.**

Map of independent re-derivations found in this sweep alone (all
distinct code, distinct thresholds, none importing
`complete_retry_identity_guard`):

1. `hybrid_retry_completion_integrity::_safe_failed_retry` — shared-token
   count + coverage + `_same_opening` 2-token match.
2. `hybrid_story_guard::_covered_by_kept_delivery` — token-coverage ratio
   + critical-token subset (restoration-suppression direction only).
3. `hybrid_alternate_integrity` — shared-token count + coverage,
   before/after-winner tiered thresholds.
4. `hybrid_cross_group_retry_integrity::_covered_by_authoritative_peers`
   — single-peer or contiguous-chain coverage + critical-token subset.
5. `incomplete_bridge_retry_authority::_sequence_identity` — an
   independent SequenceMatcher computation of the SAME metric
   `complete_retry_identity_guard` uses, at a different (lower) threshold
   — the most concerning shape, since it does not even use different
   evidence, just a different bar on the identical evidence.
6. `hybrid_failed_continuation_integrity` (two functions) — combined-
   fragment content coverage + critical-token preservation.
7. `hybrid_retry_winner_authority::_same_retry_attempt` — the ORIGINAL
   D-109/D-110 finding, shared-token coverage, now gated by the guard's
   verdict when one exists.

This is not five variations of one bug — it is one architectural gap
(no chain-wide replacement-verdict contract) expressed independently
seven times, because each hook was authored separately, over time, to
solve its own narrow pattern, with no shared consultation point ever
established. D-096/D-098's own authority-collision framing named this
class of risk in the abstract; D-109/D-110/this sweep are the concrete
proof.

---

## 6. Minimum general authority-coordination fix shape (conceptual only — NOT implemented)

One canonical directional replacement verdict, computed once per run by
the authority that already computes the strongest evidence
(`complete_retry_identity_guard`), consumed — never recomputed — by every
downstream authority capable of destructive membership action:

```
candidate X
  → proposed replacement Y
  → verdict: ACCEPTED | REJECTED | UNKNOWN
  → evidence (e.g. sequence_identity, threshold)
  → confidence
  → owning authority (complete_retry_identity_guard)
```

Any hook that is about to remove X in favor of Y (or record Y as
"covering" X) first checks whether a verdict already exists for that
EXACT directional (X, Y) pair this run:
- REJECTED → decline the removal, record why (D-110's own
  `prior_replacement_rejection_respected` pattern, generalized).
- ACCEPTED or UNKNOWN → proceed under the hook's own existing evidence,
  unchanged.

This is authority coordination, not a new heuristic: no threshold
changes, no new evidence type, no new authority that DECIDES anything —
only a shared, already-computed verdict every destructive hook consults
before acting, exactly the same shape D-110 already proved safe for one
hook.

**14. Which module should own the shared contract?**
`complete_retry_identity_guard.py` — it already computes the strongest,
most principled evidence (real `SequenceMatcher`-based sequence
identity against an evaluated threshold) and already has a documented,
tested verdict vocabulary (`NO_CANDIDATE`, `SEMANTIC_OVERLAP_BELOW_
THRESHOLD`, `NUMBER_PRESERVATION_FAILED`, `SEQUENCE_IDENTITY_BELOW_
THRESHOLD`, `LEXICAL_REPLACEMENT_VERIFIED`, `INCOMPLETE_RETRY_LOOSER_
MATCH`, `NOT_APPLICABLE`). Reusing it as the shared source needs no new
authority, per D-111 §10.4/§10.6 doctrine (Hybrid/Gemini and any
secondary evidence remain nominations; the guard's own stricter
deterministic verdict already outranks them, per D-110's proven
doctrine).

**15. Which hooks should consume it?** At minimum, every hook proven or
potential in this sweep that can independently claim "Y covers X" or
delete X in favor of Y: hooks 2, 4, 5, 6, 7 (all PROVEN or POTENTIAL
COLLISION for the removal/substitution direction), plus hook 3 in its
restoration-suppression direction (so it does not perpetuate a false
delete a shared verdict would otherwise flag as REJECTED). Hook 1
(`semantic_fragment_guard`) is lowest priority given its narrow
micro-fragment gate, but is not categorically exempt.

**16. Whether existing `complete_retry_identity_guard` can be the
source:** YES, per 14 above — it already runs early enough (consulted
by stage-1 cleanup) and its own module docstring already documents that
its diagnostic is "additive observability only" and not yet read by
anything but D-110's hook. The `_consume_replacement_guard_diagnostic()`
accessor pattern D-110 already built (`_prior_replacement_rejections` in
`hybrid_retry_winner_authority.py`) is a directly reusable template for
every other consumer — this sweep does not need a new extraction
mechanism, only its repetition at more call sites.

---

## 7. Risks / regression surface (conceptual, since nothing is implemented)

- The SAME risk D-110 already proved manageable for one hook (a
  legitimate retry pair with no prior rejection must keep working
  unchanged) applies at each additional consumer — every hook needs its
  own "no prior rejection → existing behavior unchanged" positive control,
  exactly as D-110's own test suite did.
- Hook 6 (`incomplete_bridge_retry_authority`) is the highest-risk single
  consumer to migrate, because it currently recomputes sequence identity
  at a LOWER threshold (0.42) than the guard's 0.52 — simply gating it on
  the guard's verdict (without touching its own threshold, per D-110's
  "do not recompute" rule) could change its behavior for genuine 0.42–
  0.52 cases that currently pass this hook's own bar; that interaction
  needs its own dedicated test matrix, not an assumption either way.
- Multiple consumers reading the same verdict store raises an ordering
  question this sweep does not resolve: if hook 2 removes X in favor of
  Y (no prior rejection existed at that point), and the guard's OWN
  verdict for (X, Y) is only computed/recorded at stage 1 BEFORE the
  chain runs (confirmed: stage 1 always runs first, chain hooks run
  after), then the verdict IS available to every hook in the chain by
  construction — no read-order hazard exists for the stage-1-computed
  verdict specifically, but a future consumer must not confuse "no
  verdict recorded" (`UNKNOWN`/`NOT_APPLICABLE`, not itself a green
  light) with "verdict says ACCEPTED".
- Doctrine risk (D-111 §10.4): coordinating five re-derivations into one
  contract is itself a moderate-size change touching five files;
  "genuine authority coordination" must not slide into "one more
  reconciliation layer" — the fix must delete/bypass the five hooks' own
  redundant evidence computation for the SPECIFIC pairs the guard has
  already ruled on, not add a sixth parallel check on top of them.

---

## 8. Expected effect on pimples A

**Unmeasured, and likely still incomplete even if hook 2 alone is
fixed.** If ONLY hook 2 is given D-110-style guard-consultation (the
originally proposed next step), A would very likely survive hook 2 this
time — but hook 5 (`hybrid_cross_group_retry_integrity`) runs THREE
positions later in the SAME chain and, per its own coverage math (12
content tokens vs. C's much longer text), is structurally capable of
independently re-deriving "C covers A" and removing A anyway, exactly as
it proved doing to the unrelated ob-gyn pair this same run. **A full fix
requires at least hooks 2 AND 5 to consult the shared verdict before A's
fate is provably stable** — this sweep does not authorize implementing
either.

## 9. Whether this would affect B/C complementary rescue

**No effect expected, by design.** The proposed contract is scoped to
declining removals when a REJECTED verdict already exists for the EXACT
directional pair being acted on. B (`clip_f62795f39a827a8e197e`) and C
(`clip_390e5f221849f30bb34a`) have no recorded `complete_retry_identity_
guard` rejection between them this run — `hybrid_semantic_complementary_
rescue`'s restoration of B (unique-content coverage 0.6429, `unique_
fraction` 0.3571) is untouched by any authority discussed here and does
not consult the guard either way. The directional design (per D-110 and
reaffirmed by this sweep) never poisons a whole family or an unrelated
pair — B↔C legitimately competing/complementing is a completely
different (B, C) pair from the rejected (A, C) pair, so a shared
verdict store changes nothing about it.

---

## 10. Confirmations

- **NO CODE CHANGED.** `git status`/`git diff --stat` confirm only this
  new forensic document and (if appended) one decision-log entry were
  written; no `cutsell_worker/*.py`, `tests/*.py`, or `benchmarks/*.py`
  file was modified.
- **NO RAW / PROVIDER / INFRA.** No RAW was launched, no provider was
  called, no S3/Modal/RunPod/UI work was performed. All evidence reused
  the already-persisted job log from RAW `34123511687` (fetched again,
  read-only, via the GitHub Actions job-logs API — not a new paid run).

---

## 11. One exact next implementation scope (recommended, NOT authorized here)

A single bounded task: build the shared verdict-consumption helper once
(reusing D-110's own `_prior_replacement_rejections`-style pattern
against `complete_retry_identity_guard`'s existing diagnostic), and wire
it into the TWO PROVEN-COLLISION hooks first (`hybrid_retry_completion_
integrity` and `hybrid_cross_group_retry_integrity`) as the minimum
change that would make the real pimples A→C case and the proven ob-gyn
case both behave correctly, with the same positive/negative/directional
control shape D-110's own test suite already established as sufficient.
The five POTENTIAL_COLLISION hooks (1, 3, 4, 6, 7) would be named as
explicitly out-of-scope-for-now, tracked, not silently left undocumented.
This is a recommendation only — it is not authorized by this task.
