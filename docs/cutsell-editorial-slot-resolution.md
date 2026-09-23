# CutSell Editorial Slot Resolution

## Purpose

Clean Cut must optimize for a minimum sufficient editorial set, not maximum semantic coverage. Multiple good retries may express different wording or supporting details while still performing the same audience-facing narrative job. When that happens, they should compete for one editorial slot instead of being co-kept by default.

## Core rule

> Never keep two complete realizations of the same editorial slot merely because both are good or contain unique wording.

## Resolver contract

For candidates inside a retry family:

1. Infer whether candidates serve the same audience-facing editorial function.
2. Determine whether each candidate is a complete rhetorical realization of that function.
3. When two or more candidates are complete realizations of the same function, mark them as competing realizations.
4. Distinguish required propositions from supporting, restated, or elaborative detail.
5. If one complete realization sufficiently covers the required intent, select one winner using:
   - required-idea coverage;
   - contradiction/factual safety;
   - completeness;
   - redundancy against already-selected material;
   - delivery quality;
   - narrative fit;
   - rhythm/brevity.
6. Build a composite only when no single realization is sufficient and the combined parts form one coherent superior realization.
7. Preserve existing critical-claim safety, but supporting wording must not automatically force co-keep.

## Diagnostic

Add a `redundant_complete_realization_rate` signal (or equivalent) for selected output. It should count duplicated complete narrative functions after selection. Human Gold target is effectively zero except where repetition is intentionally justified by the edit.

## Canonical Video00 failure

Human Gold keeps the first complete closing realization and then moves to CTA. Cut.ai and current CutSell diagnostics preserve another good closing/restatement because its wording/supporting facts differ. This is a complete-realization competition failure, not SWAP and not a missing-claim problem.

## Required QA order

`diagnose -> fix -> tests -> CI -> RAW -> artifact -> inspect -> repeat`

Do not declare Gold until automatic QA and Human Watch+Listen pass.
