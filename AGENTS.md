# CutSell.ai — Agent Operating Instructions

These instructions apply to Codex and any developer/reviewer/QA agent working in this repository.

## Read first

Before changing CutSell code, read in this order:

1. `docs/CUTSELL_CURRENT_STATE.md`
2. `docs/CUTSELL_DECISIONS.md`
3. `docs/CUTSELL_BRAIN_DOCTRINE.md`
4. `docs/CUTSELL_MOBILE_V1_ASAP_SCOPE.md`
5. `docs/CUTSELL_STAGING_DEPLOYMENT_CONTRACT.md`
6. `docs/CUTSELL_STAGING_READINESS.md`
7. `docs/CUTSELL_COMMERCIAL_ENGINEERING_OPERATING_MODEL.md` -- canonical roles, gates, and QA modes (D-062); governs who may implement vs. who may certify release-ready.

Treat these as the product constitution for the clean CutSell release path.

## Active release context

- Product generation: `CutSell.ai 7`
- Repository: `AutomatedRetailServices/EditDNA-worker`
- Active branch: `cutsell/mobile-v1-clean`
- Active PR: `#25` Draft
- Current Brain checkpoint: Benchmark #38 completed technically; editing-quality review remains the meaningful gate.

## Never do without explicit user approval

- merge PR #25;
- modify or merge `main` as part of CutSell release work;
- modify PR #24 unless explicitly requested;
- deploy production;
- perform Apple/TestFlight release actions;
- create or replace paid RunPod workers for a new benchmark/run;
- create new paid infrastructure;
- make destructive legacy/repository/archive changes;
- silently increase recurring provider/infrastructure spend.

## Engineering rules

- Make permanent changes in GitHub; do not rely on undocumented VM/container hot patches.
- Prefer the smallest general fix that addresses a root cause.
- Do not add one-video hardcoded exceptions to make a benchmark pass.
- Keep modules small and responsibility-specific.
- Preserve typed/versioned contracts and deterministic source identity.
- Provider failure and fallback behavior must remain observable.
- Add or update tests whenever behavior changes.
- Run targeted tests before broad regression.
- Do not declare a Brain improvement from structural workflow success alone.

## Brain invariants

### Understand before destructive editing

Use whole-source Watch + Listen context before making destructive local decisions when relevant.

### Clean Cut authority

Clean Cut may remove production errors such as false starts, failed takes, dead air, fumbles, retry debris, explicit restart behavior and boundary artifacts.

Clean Cut must not delete otherwise valid speech simply because it lacks commercial value or does not fit a sales role.

When deletion evidence is uncertain, prefer preserving valid content.

### Retry grouping / Best Take

Group retries by the same specific communication attempt, not merely broad topic or funnel label.

Best Take ranks valid alternatives using multimodal/contextual evidence. It is not a deletion authority.

### No fabricated speech

Never create speech the creator did not deliver by cross-source or cross-take Frankenstein sentence stitching.

### Flexible sales/story logic

Do not force `HOOK -> PROBLEM -> BENEFIT -> PROOF -> CTA`.

Discover the coherent sales/story path actually present in the footage. Functions may be missing, repeated or reordered when justified by the source and global coherence.

### Visual vs semantic understanding

Do not confuse visual presentation type with semantic/commercial role.

Examples:
- visual type: talking head/headshot, product shot, demo, B-roll/lifestyle, close-up;
- semantic role: hook, problem, story, discovery, feature, benefit, proof, result, CTA.

A segment may have multiple semantic roles.

### Story continuity

Storytelling requires relationships and context: setup, progression, cause/effect, discovery, payoff and chronology. Do not optimize isolated clips at the expense of the complete story.

### Preserve personality

Do not remove authentic humor, mannerisms, reactions or creator personality merely because a more sterile edit looks cleaner.

## Validation protocol

For a Brain/editorial change:

1. State the observed failure in the current benchmark/evidence.
2. Identify the responsible layer/module.
3. Form a general root-cause hypothesis.
4. Implement the smallest general fix.
5. Add/update targeted tests.
6. Run targeted tests.
7. Run existing non-paid regression/CI where appropriate.
8. Compare expected behavior against `CUTSELL_BRAIN_DOCTRINE.md`.
9. Do not launch a new paid benchmark without approval.
10. If a paid benchmark is approved, review rendered outputs; workflow `success` is not sufficient to claim quality success.

## Documentation synchronization

When product behavior changes intentionally:

- update `docs/CUTSELL_DECISIONS.md` in the same cycle;
- update `docs/CUTSELL_BRAIN_DOCTRINE.md` if editing doctrine changed;
- update `docs/CUTSELL_MOBILE_V1_ASAP_SCOPE.md` if V1 product scope changed;
- update `docs/CUTSELL_STAGING_DEPLOYMENT_CONTRACT.md` if topology/deployment gates changed;
- update `docs/CUTSELL_CURRENT_STATE.md` whenever the active benchmark/checkpoint/focus changes.

Do not allow chat-only decisions to become invisible product requirements.

## Agent roles

Initially use a small role set rather than uncontrolled multi-agent sprawl:

### Developer
Investigates root causes, implements scoped fixes, updates tests and reports exact changes.

### QA / Eval
Attempts to falsify the change using targeted tests, regression cases and benchmark evidence. Does not rewrite product doctrine.

### Reviewer
Checks the diff against the canonical docs, guards against regressions/hardcoding/scope creep and verifies tests before changes are accepted.

Specialize further into Brain/Backend/iOS roles only when the current small-team loop is demonstrably useful.

## Definition of done for an engineering task

A task is not done merely because code was written. It is done when:

- the requested/root-cause behavior is implemented;
- tests cover the changed behavior;
- relevant tests pass;
- no canonical rule is violated;
- documentation is synchronized if behavior/scope changed;
- paid/irreversible gates were not crossed without approval;
- the result and remaining risks are reported clearly.