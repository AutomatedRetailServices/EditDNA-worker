# D-042 — Editorial-slot resolution and the minimum sufficient editorial set

**Status: CANONICAL ADDENDUM**

> This decision is canonical for `cutsell/mobile-v1-clean` and is intended to be folded
> into `docs/CUTSELL_DECISIONS.md` as D-042. It lives as a separate addendum only because
> the current connected GitHub write surface supports full-file replacement but not a
> safe append/patch of the large decision log. Do not treat this separation as a product
> or architecture fork. The master log remains authoritative except where this later
> decision explicitly refines D-009/D-019/D-020/D-021/D-023 behavior.

## Human-Gold finding

A direct editorial comparison of RAW Video00, the Human Gold edit, a Cut.ai edit, and
CutSell diagnostics exposed a failure class that is more specific than generic retry
selection or claim coverage.

Human Gold keeps one complete conclusion from the RAW around `~294.9–313.9`, then moves
to the final CTA around `~358.1–361.4`. Cut.ai and the then-current CutSell behavior both
kept an additional later conclusion/reformulation around the RAW `~319–346` region even
though the story already had a sufficient conclusion. In Cut.ai this manifested as an
extra continuous block of roughly `22.6s` immediately before the CTA.

The second conclusion contains wording/details that are not literal duplicates of the
first. Treating every unique supporting fact as a reason to preserve both therefore
produces an over-preserved, non-human edit. The human editorial decision is made at a
higher level: both deliveries perform the same audience-facing job — CONCLUSION — and
one complete realization is sufficient.

This is not SWAP. D-019 remains unchanged: Clean Cut Core V1 must produce an authoritative
first edit. Manual alternate-take UX is not the resolver for this failure.

## Canonical objective

Clean Cut optimizes for the **MINIMUM SUFFICIENT EDITORIAL SET**, not the union of all
facts spoken across every usable take.

For a retry family / candidate family:

1. Determine the audience-facing **EDITORIAL FUNCTION** before treating literal wording
   differences as separate ideas (for example HOOK, SETUP, DIAGNOSIS, SYMPTOM, EXAMPLE,
   REFLECTION, CONCLUSION, CTA).
2. If two complete deliveries perform the same function and communicate the same core
   intended message, they are competing realizations even when wording, length, examples,
   or supporting facts differ.
3. Distinguish **REQUIRED PROPOSITIONS** from **SUPPORTING/RESTATED/ELABORATIVE DETAIL**.
   Unique supporting detail does not automatically create a separate idea that must
   survive beside an already sufficient realization.
4. If one complete realization is sufficient, the family must resolve to one winner.
   **GOOD + GOOD does not imply KEEP BOTH.**
5. Use a composite only when no single realization is sufficient and complementary clean
   pieces are genuinely required to create one coherent complete realization.
6. Do not collapse genuinely complementary micro-deliveries into one-winner competition;
   true composites remain protected.

The practical ranking order for complete same-slot retries is:

1. required idea/proposition coverage;
2. contradiction and factual safety;
3. completeness;
4. redundancy with an already-sufficient realization;
5. delivery quality;
6. narrative fit;
7. rhythm / brevity.

Completeness is an editorial prerequisite, not merely a stylistic preference.

## Active-path implementation

The active Clean Cut Core V1 path remains the D-020/D-021 idea-first path. The deprecated
whole-video Unified Selection reasoner is NOT reactivated.

### 1. Complete alternate cannot be destroyed by an incomplete semantic winner

`hybrid_retry_completion_integrity.py` no longer irreversibly deletes a complete
alternate solely because it overlaps a high-confidence semantic winner that is itself
incomplete. The complete realization remains available for downstream family resolution.
A complete alternate may still yield to an independently complete winner under the
existing conservative coverage rule.

This reverses an obsolete benchmark regression that had encoded the Video00 conclusion
backwards (delete the Human-Gold-like complete `~295–314` realization in favor of the
later incomplete retry). The regression now preserves the complete realization for
competition.

### 2. RESTORE is not co-keep authority

`hybrid_semantic_complementary_rescue.py` may restore valid material that an earlier
classifier would otherwise lose, but a semantic rescue is no longer promoted into
Best-Take immunity.

`composite_resolver._composite_split_ids()` now returns only IDs proven by
`hybrid_composite_best_take` to be a true multi-piece composite. Semantic-rescue IDs are
consumed/cleared but are not forced into singleton groups. Therefore:

- semantic rescue => RESTORE, then normal competition;
- true composite => preserve complementary pieces together.

This is the mechanical form of: **unique content may justify restoration, but does not
by itself justify KEEP BOTH.**

### 3. Editorial-slot semantics are installed in the ACTIVE bounded arbiter

An initial implementation only added the policy to `unified_selection_google`, which is
a rollback/deprecated path under D-020. That would have been a no-op for normal Clean Cut
Core V1 and was corrected before this decision was accepted.

`editorial_slot_resolution_install.py` now injects the editorial-function definition into
`semantic_idea_equivalence_google.build_semantic_equivalence_request`, the bounded arbiter
actually called by `pipeline.py` before Best Take.

The arbiter is instructed that two deliveries may be the SAME intended idea even if
wording, length, examples, or supporting facts differ. Extra detail that merely restates,
supports, specifies, or elaborates the same audience-facing point does not create a new
idea by itself. A second complete conclusion/restatement of the same takeaway is SAME;
a genuinely distinct required story proposition or complementary next story beat is
DIFFERENT. The arbiter forms the competition family; it does not rank the winner.

### 4. Fixed pair budget now protects family coverage instead of dense-region redundancy

A regression with more than the existing `14` eligible semantic-equivalence pairs proved
that the Human-Gold-like conclusion pair could still be absent from the arbiter request:
the score-ranked budget could spend many slots comparing variants in one dense local
region and starve a later retry family entirely.

The budget is NOT increased and no extra provider call is introduced.

`editorial_slot_resolution_install._coverage_first_pair_order()` wraps the existing
priority-ranked candidate stream. It first prefers pairs that expose previously unseen
group endpoints (2 unseen groups, then 1, then 0), using the existing score order as the
tie-break. Once group coverage is exhausted, remaining slots return to strongest-score
ordering.

This changes only which already-eligible pairs receive the fixed bounded request budget.
The semantic arbiter still decides SAME/DIFFERENT; no deterministic semantic merge was
added. Protected true-composite IDs remain excluded from candidate generation exactly as
before.

### 5. Incomplete semantic winner cannot override a complete local Best Take

`semantic_best_take_integrity.py` now rejects one narrow unsafe override: if local Best
Take selected a complete realization and Hybrid/semantic evidence proposes an incomplete
peer as the unique semantic winner, the semantic override is rejected and the complete
local winner remains selected.

If both candidates are complete, the existing semantic override path remains available.
Existing failed/BTS safety, numeric-fact protection, and tied-winner information-coverage
rules remain intact.

This enforces the D-042 ordering `completeness > delivery preference` without hardcoding
Video00 wording or timestamps.

## Composite non-regression

Human Gold's pimples/rash region demonstrates the opposite case: clean subparts from
multiple attempts are genuinely complementary and should form one realization. The fix
therefore explicitly preserves true composite immunity while removing immunity from a
mere semantic rescue.

Regression coverage proves:

- semantic-rescue IDs alone do NOT force Best-Take singleton groups;
- true composite IDs DO remain singleton/protected for Best Take;
- when rescue and true-composite evidence overlap, only the true-composite IDs receive
  co-keep immunity.

## Regression coverage added / updated

The branch includes dedicated tests for:

- preserving a complete Video00-like conclusion against an incomplete semantic winner;
- still allowing a complete alternate to yield to an independently complete winner;
- active editorial-slot policy injection and idempotence;
- keeping genuinely complementary next story beats distinct;
- semantic rescue versus true composite immunity;
- a saturated semantic pair budget in which late same-slot conclusions must still reach
  the arbiter and merge into one retry family;
- blocking an incomplete semantic winner from overriding a complete local Best Take;
- preserving normal semantic override when both competing realizations are complete.

## Validation at acceptance

Accepted branch head: `eea83456a1aae92c44deaa3eac7df9ad88e4ac72`.

At that head:

- `pytest -q tests/test_cutsell_*.py`: PASS (all current CutSell tests green);
- Python compile of `cutsell_worker` + `cutsell_app`: PASS;
- staging API container build: PASS;
- staging API health endpoint: PASS;
- iOS project generation / Simulator build: PASS.

`main` is untouched and PR #25 remains unmerged.

## What this decision does NOT claim yet

This decision proves the architectural/mechanical invariants and regression behavior. It
does **not** claim empirical Human-Gold parity on a newly rendered Video00 MP4 yet.

Per D-018, the next controlled Video00 RAW / RunPod benchmark is a paid action and requires
explicit user approval. That benchmark must verify at minimum:

- the complete `~295–314` conclusion survives and wins the conclusion slot;
- the later redundant conclusion/reformulation does not co-survive merely because it has
  different supporting detail;
- CTA follows the sufficient conclusion cleanly;
- the genuine pimples/rash composite behavior remains intact;
- no new content-loss or boundary regressions appear elsewhere in the rendered output.

## Follow-up observability

A useful next diagnostic is **REDUNDANT COMPLETE REALIZATION RATE**: the rate/count of
narrative/editorial functions that remain represented by more than one complete
realization in the final selected timeline. This should be derived from decisions already
made by the bounded semantic-equivalence/family-resolution path, not by adding another
paid whole-video semantic pass. It is a diagnostic follow-up, not a substitute for the
D-042 selection behavior and not required to authorize the next controlled RAW.
