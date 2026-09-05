# CLAUDE.md — CutSell.ai Operating Contract

You are continuing an existing production-oriented app. Do NOT redesign CutSell and do NOT start from scratch.

## Mandatory startup
Before editing, inspect live Git state:

```bash
git status
git branch --show-current
git rev-parse HEAD
git log -1 --oneline
```

Expected handoff context unless newer intentional commits exist:
- repo: `AutomatedRetailServices/EditDNA-worker`
- branch: `cutsell/mobile-v1-clean`
- PR #25: OPEN / DRAFT / UNMERGED
- main base SHA: `2fb13e5aa228e8e525b942a9b49182032b797e61`
- handoff head before Claude docs: `9fad0788e120b2af07576f2d145fc4c179b24adb`

If HEAD is newer, inspect intervening commits and reconcile state. Never reset blindly.

## Read order
1. `docs/claude-handoff/CUTSELL_COMPLETE_HANDOFF.md`
2. `docs/claude-handoff/SECURITY_CONSTITUTION.md`
3. existing `AGENTS.md`
4. existing `docs/CUTSELL_DECISIONS.md`
5. existing `docs/CUTSELL_BRAIN_DOCTRINE.md`
6. existing `docs/CUTSELL_MOBILE_V1_ASAP_SCOPE.md`
7. existing `docs/CUTSELL_COMMERCIAL_ENGINEERING_OPERATING_MODEL.md` (D-062) -- canonical roles/gates/QA modes; an engine change is never self-certified release-ready by the role that implemented it.
8. existing `docs/CUTSELL_EDITORIAL_RESOLUTION_AND_HUMAN_ESCALATION_CONTRACT.md` (D-062.2) -- Automatic Editor Doctrine, CRITICAL_COVERAGE_DOMINANCE, the 16-layer Automatic Resolution Hierarchy, and the HUMAN_CHOICE_ELIGIBLE escalation contract; human choice is a last resort, never a convenience valve for resolver uncertainty.
9. `docs/CUTSELL_COMMERCIAL_ENGINEERING_OPERATING_MODEL.md` Section 12 (D-091) -- the Continuous Autonomous Engineering & Escalation Contract: within an authorized technical scope Claude continues through diagnose -> fix -> tests -> QA -> retest without waiting for a relayed "continue"; it stops only on a listed Product Owner escalation condition.

## Source precedence
When sources conflict:
1. live code/live Git state;
2. current Claude handoff;
3. current scope/decision contract;
4. canonical repo doctrine;
5. historical checkpoint;
6. old conversations.

## Current mission
**Flow B → Clean Cut Core V1 (idea-first).** See `docs/CUTSELL_DECISIONS.md` D-019/D-020.

Clean Cut Core V1 reasons idea-first: complete intended ideas → all delivery attempts
per idea → quality/completeness competition → one winning delivery or a necessary
composite → KEEP/DISCARD. Gemini is a bounded semantic arbiter only (idea-equivalence
during clustering, residual-ambiguity resolution during final coherence validation),
never the primary editor; the whole-video Unified Selection reasoner is deactivated in
the active path (rollback: `CUTSELL_CLEAN_CUT_CORE_V1=0`).

**SWAP IS OUT OF SCOPE FOR CLEAN CUT V1 UNTIL THE USER EXPLICITLY REINTRODUCES IT**
(D-019). The active semantic membership model is SELECT/KEEP vs DISCARD only — no
alternate-take inventory in the winning timeline. Do not design, optimize, validate,
or preserve the active Clean Cut architecture around SWAP; do not spend engineering
time maintaining or improving SWAP behavior. Legacy SWAP machinery stays in the
codebase, deactivated for this path only (never delete it destructively if that would
destabilize unrelated systems) — see `deterministic_best_take_authority.py`'s
`swap_enabled` parameter and `draft_edits.py`'s unrelated manual editor-layer
`swap_take`, which is a different product layer and out of this scope decision.

Preserve:
- one final semantic Selection authority;
- one winning realization per retry family (GOOD TAKE != UNIQUE IDEA);
- complete delivery dominates an incomplete/abandoned retry of the same idea;
- contradictory retries (differing number/negation) are never composited or left to
  coexist silently — they block Selection Freeze for human review (D-020);
- idea coverage — an intended idea must not silently vanish from the winning edit;
- Selection Freeze;
- Boundary-only physical timing after freeze; Boundary never repairs a semantic
  membership mistake;
- KEEP/DISCARD semantics (not SELECT/SWAP/DISCARD — see D-019);
- Human Gold QA (oracle only, never fed into runtime production logic);
- CleanCutBench as the general editorial test suite gating paid Video00 iteration;
- unseen generalization.

See `docs/CUTSELL_DECISIONS.md` D-021 for the canonical component map (AttemptReconstructor, IdeaClusterer, RetryFamilyResolver, DeliveryScorer, BestTakeResolver, SemanticArbiter, CompositeResolver, StoryValidator, SelectionFreeze, BoundaryEngine, Renderer) that every active behavior must map to.

## Editorial rules
- WHEN UNCERTAIN, KEEP.
- Preserve unique audience-facing information.
- Preserve story/personality.
- Remove real failed/retry/BTS material.
- Never invent speech.
- Never hardcode Video00 timestamps, phrases or clip IDs.
- Do not force rigid sales-funnel logic during Clean Cut.
- Human performance errors matter even when transcript is complete.

## Engineering rules
- Diagnose from latest run/artifact before editing.
- Prefer structural root-cause fixes.
- Add targeted tests for behavior changes.
- Run tests before CI.
- CI success is not editorial success.
- Do not run overlapping paid RAW benchmarks.
- Always verify RunPod teardown.
- Preserve observability; never accept silent provider fallback.

## Security rules
`docs/claude-handoff/SECURITY_CONSTITUTION.md` is binding.
Security runs in parallel with editorial QA; do not postpone it to launch.

## Repository protection
Without explicit user approval:
- do not merge PR #25;
- do not write to `main`;
- do not close PR #25;
- do not deploy production;
- do not perform TestFlight/App Store release;
- do not make destructive repository/archive changes;
- do not expose/move secrets;
- do not create materially new recurring paid infrastructure.

## Current state pointer
The live current state is the newest `D-xxx` entry in `docs/CUTSELL_DECISIONS.md`
(read its LAST VERIFIED RESULT / NEXT AUTOMATIC ACTION lines). The handoff
document's `CURRENT LIVE BLOCKER` / `EXACT NEXT ACTION` sections are historical
context from the Unified Selection era, superseded by the decision log; do not
treat them as the standing next task.

## Continuity contract (D-091)
Within an already-authorized technical scope, CONTINUE AUTONOMOUSLY. Do not wait
for "continue", "proceed", "what next?" or a relayed prompt when the next action
is an ordinary technical consequence of the current authorized objective. A
technical checkpoint (a D-xxx entry, a green test run, a QA verdict) is not a
stopping point by itself.

Canonical loop:
`diagnose → reproduce → fix → targeted tests → QA (independent pass) → if QA
finds an in-scope defect return to Engineering → retest → offline qualification
/ CI where applicable → analyze evidence → next proven root cause within scope`.

Engineering and QA remain distinct roles with separate reporting; Claude
orchestrates the hand-off between them itself. Engineering never rewrites a QA
verdict; it fixes and re-submits.

Stop and ask the Product Owner ONLY when at least one is true:
- A. PRODUCT DECISION REQUIRED (product behavior, UX doctrine, workflow,
  editorial policy, pricing, scope, acceptance criteria);
- B. SAFETY / AUTHORITY CHANGE REQUIRED (would weaken Freeze, critical-content
  protection, security, privacy, authorization, tenant isolation, claim safety,
  or another canonical safety contract);
- C. PAID COMPUTE OUTSIDE AUTHORIZATION (any Modal/RunPod/provider run beyond
  the currently approved count/budget/scope; default: no paid compute);
- D. PROTECTED REPOSITORY ACTION (the list under "Repository protection");
- E. P0/P1 ACCEPTED-RISK DECISION;
- F. HUMAN EDITORIAL ACCEPTANCE (a real rendered artifact needs watch/listen);
- G. TRUE SCOPE BOUNDARY (the proven next root cause belongs to a materially
  different objective).
Otherwise: continue.

Task-local "Then STOP" / "No code" / "No RAW" / "Report only" bound THAT
directive's actions only; they never disable this contract. NO RAW still allows
offline diagnose/fix/test/QA. NO CODE allows investigation/documentation.
REPORT ONLY ends before code modification; an implementation already authorized
by the Product Owner may then proceed under that authorization.

Root-cause continuity: a defect exposed within the same authorized objective is
investigated and fixed automatically; an unrelated one is recorded separately
without expanding scope. Sequential D-xxx entries may be created autonomously
while inside authorized scope and no stop condition is crossed.

## QA loop
`diagnose → fix → targeted tests → independent QA → auto-loop back on in-scope
defects → offline qualification / CI where applicable → (paid run only when
authorized) one RAW → JSON+MP4 → architecture check → Selection analysis →
Human Gold → Watch+Listen → unseen/regression`

On `feature/runpod-pod-on-demand` no unpaid CI workflow runs on push (the
clean-worker CI runs on pull requests to `main`), so the offline qualification
set (compileall, targeted suites, CleanCutBench both modes, full
`tests/test_cutsell_*.py`, whole `tests/`) is the CI-equivalent gate.

## Status vocabulary
While continuing, report compactly with this block:
- CURRENT OBJECTIVE
- CURRENT STAGE
- LAST VERIFIED RESULT
- CURRENT ROOT CAUSE
- NEXT AUTOMATIC ACTION
- HUMAN ACTION REQUIRED: YES/NO (with the escalation condition letter if YES)

Never ask "what do you want me to do next?" when the next technical action is
already implied by the current objective.

Report exact state instead of generic "done":
- CODE FIXED
- TESTS PASS
- CI GREEN
- RAW COMPLETE
- ARCHITECTURE PASS/FAIL
- GOLD CANDIDATE
- HUMAN WATCH+LISTEN PASS
- REGRESSION PASS
- SECURITY REVIEWED / SECURITY SCAN PASS when applicable

## Documentation duty
After material checkpoints, update durable current-state/decision docs. Never make the user reconstruct CutSell from chat again.
