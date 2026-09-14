# CutSell Commercial Engineering Operating Model

Canonical governance document. This defines WHO owns WHAT, WHO may block release,
and WHAT independent verification looks like as CutSell moves from benchmark-driven
engine development toward a commercial product with paying users. It is a durable
governance layer on top of the existing engineering record in
`docs/CUTSELL_DECISIONS.md` -- it references historical D-xxx decisions by number
rather than restating them, and it does not change, reinterpret, or re-gate any
already-shipped decision.

This document does not authorize production deployment, merging PR #25, or any
paid infrastructure run on its own. Repository-protection rules in `CLAUDE.md`
remain binding and are unchanged by this document.

## Scope and non-goals

This is a governance/documentation checkpoint, not an implementation task:

- It does not modify engine behavior, editorial logic, Human Gold, Selection
  Freeze, or any file under active D-xxx development.
- It does not launch Modal, RunPod, or any paid infrastructure.
- It does not build a release-readiness UI -- Section 10 is a contract/schema
  only.
- It does not implement the reusable review commands in Section 8 as runnable
  tools -- it defines their contract so a future directive can wire them.
- It does not invalidate, restart, or reinterpret any decision already recorded
  in `docs/CUTSELL_DECISIONS.md`, including D-061.

## 1. Canonical roles

Eleven roles. A role is a set of responsibilities and authorities, not
necessarily a distinct person -- one engineer (human or agent session) may
carry several roles, but may never grant itself the authority of a role whose
whole purpose is to check its own work (see Section 3).

### 1. CutSell Product Owner / MVP Authority

Owns: product scope, MVP definition, acceptance criteria, prioritization,
diminishing-return decisions, and the decision to move from benchmark
development to broader product validation.

Cannot override: P0/P1 QA blockers, security blockers, data-integrity blockers.
Scope authority is not a veto over safety findings.

### 2. CutSell Platform / Application Engineer

Owns: API, backend, database, jobs, uploads, storage, auth integration,
project lifecycle, provider abstraction, billing integration, mobile/web
contracts, idempotency, error handling.

### 3. CutSell AI / Video Engine Engineer

Owns: ASR, canonical evidence, attempt reconstruction, semantic grouping,
retry resolution, Semantic Ledger, Unified Resolver, CanonicalEditPlan,
Freeze, Boundary, Render, Watch/Listen QC, engine benchmarks -- the D-021
canonical component map and everything built on it through D-061.

May implement. May NOT independently certify its own implementation for
release -- see Section 3.

### 4. CutSell QA / Release Engineer

Independent verification authority. Operates in modes, each a distinct review
posture over the same evidence:

- `QA_COMPONENT`
- `QA_ENGINE`
- `QA_INTEGRATION`
- `QA_E2E`
- `QA_MOBILE`
- `QA_REGRESSION`
- `QA_EXPLORATORY`
- `QA_PERFORMANCE`
- `QA_RELEASE`

Engineering evidence (test counts, offline qualification, an engineer's own
report) is INPUT to QA, not automatic proof of PASS. QA re-derives its own
verdict from the evidence and, where warranted, from independent inspection of
the implementation itself -- exactly the posture D-061's own QA_ENGINE pass
already modeled (re-deriving live diagnostics shapes from `pipeline.py` rather
than trusting the implementer's own account of them).

### 5. CutSell Security & Privacy Engineer

Owns: authorization, tenant/user isolation, object ownership, API security,
rate limiting, upload safety, signed URLs, storage privacy, secret handling,
dependency security, abuse prevention, retention, deletion, PII/log exposure,
provider-data exposure.

Security may independently block release regardless of any other role's
verdict.

### 6. CutSell SRE / DevOps Engineer

Owns: deployments, CI/CD, environments, Modal/RunPod/provider health, queues,
timeouts, retries, idempotency, duplicate-run prevention, observability,
metrics, alerts, rollback, backup/recovery, incident response, provider
outages, capacity/concurrency, runbooks.

### 7. CutSell Product / UX Engineer

Owns: onboarding, project creation, upload UX, processing UX, progress,
errors/retries, preview, timeline/editor, captions, manual correction,
export, mobile interaction, accessibility, responsive behavior.

### 8. CutSell FinOps / AI Cost Engineer

Owns: ASR cost, LLM cost, GPU cost, render cost, storage, bandwidth,
database/provider costs, failed-job cost, retry waste, COGS/video,
COGS/minute, gross-margin modeling, cost ceilings, cost anomalies.

Semantic safety must never be silently weakened to save money -- a cost
concern is raised to the Product Owner and Engine Engineer, never
unilaterally resolved by loosening a safety guard (e.g. `AMBIGUOUS_COVERAGE_
FLOOR`, `_DEFINITIVE_MISMATCH_COVERAGE_CAP`, or any arbiter cost ceiling)
without an explicit, recorded decision.

### 9. CutSell Data / Product Analytics

Owns canonical metrics: upload success, processing completion, processing
latency, Freeze-pass rate, render success, export success, manual correction
time, Human Edit Reduction %, output acceptance, rerun rate, failure rate,
cost/video, activation, conversion, retention.

### 10. CutSell Trust / Compliance Owner

Owns operational compliance readiness: privacy requirements, terms
requirements, UGC/content handling, copyright considerations,
retention/deletion, subprocessors, AI-provider handling, TikTok/platform-
policy considerations, billing/refund requirements, consent/disclosure
requirements.

Not a substitute for qualified legal counsel.

### 11. CutSell Human Editorial Acceptance

Independent human/product-quality authority for rendered output. Evaluates:
failed takes removed, bad retries removed, best retake selected, useful
micro-fragments preserved, duplicate ideas removed, critical meaning
preserved, story flow, visual cuts, audio cuts, pacing, publish-readiness,
manual correction required.

Canonical classifications:

- `PUBLISH_WITHOUT_CHANGES`
- `MINOR_CORRECTIONS`
- `MAJOR_CORRECTIONS`
- `REJECT_OUTPUT`

Canonical KPI:

```
HUMAN_EDIT_REDUCTION_PERCENT =
    (manual_baseline_edit_time - post_CutSell_correction_time)
    / manual_baseline_edit_time
    * 100
```

Human Editorial Acceptance may reject an output even when automated tests
pass -- this is the same relationship CLAUDE.md already establishes between
"CI success" and "editorial success" ("CI success is not editorial success"),
now given a named, independent role.

## 2. Separation of duties

Canonical doctrine:

- Engineering builds.
- QA verifies.
- Security challenges security.
- SRE challenges operability.
- FinOps verifies economics.
- Human Editorial Acceptance judges the actual video.
- Product Owner controls scope.
- Release Gate decides go / no-go.

No single role may self-certify the complete product. In particular: the AI /
Video Engine Engineer that implements a fix (e.g. D-061) is never the same
authority that certifies it ready for a paid canary or for release -- that is
QA's job, run as a genuinely independent pass over the implementer's own
evidence, not a restatement of it. When one session or person carries both
roles (as is common in a small team), the QA pass must still be performed as
its own distinct, skeptical review -- exactly the disclosed, self-conducted
QA_ENGINE pattern already used for D-061 when no dedicated QA tool existed.

## 3. Canonical gates

Seven gates. Each gate is a checkpoint a change must clear; gates compose into
the Release Gate.

| Gate | Checks |
|---|---|
| `ENGINE_GATE` | Feature/engine contract, regression, architecture invariants (D-021 component map, no new post-Resolver semantic mutator, Freeze policy unchanged unless explicitly authorized). |
| `QA_GATE` | Independent tests, negative cases, integration, E2E where applicable. |
| `SECURITY_GATE` | Authorization, tenant isolation, secrets, API security, upload/storage security. |
| `RELIABILITY_GATE` | Timeouts, provider outage, network failure, duplicate request, retry, worker death, DB failure, upload interruption, app restart, recovery. |
| `EDITORIAL_ACCEPTANCE_GATE` | Actual rendered-video quality, manual correction, Human Edit Reduction. |
| `ECONOMICS_GATE` | COGS, bounded spend, retry amplification, margin. |
| `RELEASE_GATE` | Aggregates every gate above that applies to the change under review. |

`RELEASE_GATE` returns one of:

- `GO`
- `GO_WITH_ACCEPTED_RISK`
- `NO_GO`

`GO_WITH_ACCEPTED_RISK` requires a named accepting owner and an explicit,
recorded risk description -- it is never a silent default.

## 4. Defect severity

- **P0 Blocker** -- security breach, cross-user exposure, data loss, billing
  corruption, unsafe delivered output, major production outage.
- **P1 Critical** -- core workflow unusable, critical content silently lost,
  repeated processing failure, unusable render, major auth failure.
- **P2 Major** -- important defect with a viable workaround.
- **P3 Minor** -- non-blocking UX/polish.

P0/P1 block release. P2 requires explicit accepted-risk ownership (a named
role signs off that the risk is understood and accepted, recorded in the
release readiness contract's `accepted_risks`). P3 is informational.

This scale is used retroactively for existing evidence, not only new
findings: D-061's own QA_ENGINE pass surfaced one P3 (`resolve_ambiguous_
coverage` ignores arbiter confidence, D-038, pre-existing, out of scope) --
that classification stands under this model unchanged.

## 5. Commercial QA matrix

Coverage contract for future QA work -- registering the areas, not building
them now.

- **AUTH** -- signup/login, session handling, token expiry/refresh, password
  reset, multi-device sessions, account deletion.
- **PROJECTS** -- creation, listing, rename, delete, ownership transfer,
  concurrent edits, project-level idempotency.
- **UPLOAD** -- presigned URL issuance/expiry, multi-file upload, large-file
  upload, resumable/interrupted upload, unsupported format handling, malicious
  file handling.
- **PROCESSING** -- job queueing, progress reporting, cancellation, retry,
  timeout, partial-failure recovery, concurrent jobs per user/project.
- **CUTSELL ENGINE** -- the full D-021 component chain (AttemptReconstructor
  through Renderer); CleanCutBench, Human Gold, contradiction safety,
  claim-coverage safety, Freeze/Boundary correctness -- reference, do not
  duplicate, the assets in Section 11.
- **PREVIEW/EDITOR** -- timeline rendering, manual trim/reorder, caption
  editing, undo/redo, autosave, draft recovery.
- **EXPORT** -- render correctness, format/codec compliance, download
  reliability, re-export after edit, export failure recovery.
- **BILLING** -- plan enforcement, usage metering, overage handling, payment
  failure, refund/cancellation, invoice correctness.
- **SECURITY** -- see Section 1.5's ownership list; QA exercises it, Security
  owns the standard.
- **PRIVACY** -- data retention/deletion enforcement, PII handling in logs and
  provider payloads, subprocessor data flow.
- **MOBILE** -- iOS build/boot, offline/background behavior, push
  notifications if present, device-storage limits, app-store review
  constraints.
- **PERFORMANCE** -- latency under load, GPU/queue concurrency limits, cold-
  start behavior, degradation under provider slowdown.

## 6. Development checkpoints

- **After a structural engine change:** `ENGINE_GATE` -> `RUN QA_ENGINE`.
- **Before a paid GPU canary:** `RUN QA_REGRESSION`, config verification, cost
  guard.
- **After a GPU canary:** `RUN QA_ENGINE` verifies whether the canary actually
  proved the intended change (not merely that it completed).
- **Before a feature merge:** `RUN QA_INTEGRATION`.
- **Before private beta:** `RUN QA_E2E`, `SECURITY_GATE`, `RELIABILITY_GATE`,
  `EDITORIAL_ACCEPTANCE_GATE`.
- **Before paid beta:** all of the above, plus `ECONOMICS_GATE`, billing QA,
  privacy/compliance review.
- **Before production:** full `RELEASE_GATE`.

These checkpoints are gates a change must clear, not points at which the
orchestrating agent waits for a relayed instruction -- see Section 12.

## 7. Reusable canonical commands

Definitions only -- none of these run as part of this governance update.
Each is specified as: purpose, required inputs/evidence, checks, outputs,
PASS/FAIL criteria, severity/blocking behavior.

### RUN ENGINE_GATE

- **Purpose:** confirm a structural engine change respects the D-021
  component map and introduces no undocumented new authority.
- **Inputs:** diff/commit range, `docs/CUTSELL_DECISIONS.md` entry for the
  change, full targeted + regression test output, compileall output.
- **Checks:** no new post-Resolver semantic mutator unless explicitly
  authorized; no Freeze-policy change unless explicitly authorized; no
  Grouping/BestTake/Resolver/Ledger change outside declared scope; CI green.
- **Outputs:** PASS/FAIL + the Standard Review Report (Section 9).
- **Blocking:** FAIL blocks merge and blocks `RUN QA_ENGINE`.

### RUN QA_COMPONENT

- **Purpose:** verify one component in isolation (e.g. one validator
  function) against its own contract.
- **Inputs:** the component's source, its existing unit tests, its docstring
  contract.
- **Checks:** positive cases, negative cases, boundary cases, documented
  fail-open/fail-closed behavior actually holds.
- **Outputs:** PASS/FAIL/PASS_WITH_KNOWN_ISSUES + Standard Review Report.

### RUN QA_ENGINE

- **Purpose:** independent verification of an engine-level change (a D-xxx
  directive's implementation) before any paid canary -- the mode used for
  D-061.
- **Inputs:** the full diff, all new/changed tests, offline qualification
  output, the implementer's own report, live diagnostics shape where
  derivable from source (not assumed from the report alone).
- **Checks:** implementation correctness against the stated root cause;
  negative-case/safety-guard coverage; regression evidence; no post-Resolver
  semantic mutation; no Freeze weakening; test quality (deterministic,
  generic, no Video00-specific fixtures unless the directive requires them);
  side effects on other consumers of touched code paths.
- **Outputs:** PASS/PASS_WITH_KNOWN_ISSUES/FAIL, P0-P3 counts, Standard
  Review Report.
- **Blocking:** any P0/P1 blocks a canary/release; FAIL blocks both.

### RUN QA_INTEGRATION

- **Purpose:** verify a feature across component boundaries before merge.
- **Inputs:** full feature diff, all affected components' tests, a manual or
  scripted cross-component scenario walk-through.
- **Checks:** contracts between components hold under real data shapes, not
  only unit fixtures; diagnostics propagate correctly end to end.
- **Outputs:** PASS/FAIL + Standard Review Report.

### RUN QA_E2E

- **Purpose:** verify a full user-facing flow (upload -> process -> preview
  -> export) before private beta.
- **Inputs:** a real or staging environment, a real source video, the full
  API/mobile contract.
- **Checks:** the flow completes, recovers from at least one injected
  failure, and produces a deliverable matching Editorial Acceptance
  expectations.
- **Outputs:** PASS/FAIL + Standard Review Report.

### RUN QA_MOBILE

- **Purpose:** verify the mobile app build/boot/interaction surface.
- **Inputs:** current app build, target device/simulator matrix.
- **Checks:** build succeeds, core flows work, offline/background/interrupt
  behavior is correct, accessibility basics hold.
- **Outputs:** PASS/FAIL + Standard Review Report.

### RUN QA_REGRESSION

- **Purpose:** confirm nothing already-working broke.
- **Inputs:** full test suite, CleanCutBench LEGACY+AUTHORITATIVE, prior
  baseline counts.
- **Checks:** test counts match baseline + expected new tests, with 0
  unexplained new failures; same known/pre-existing failures only.
- **Outputs:** PASS/FAIL + exact counts.
- **Blocking:** any unexplained new failure blocks a canary/release.

### RUN QA_EXPLORATORY

- **Purpose:** unscripted probing for gaps the formal matrix doesn't yet
  cover.
- **Inputs:** current build, tester judgment.
- **Checks:** none fixed in advance -- findings are logged as new P0-P3
  candidates and triaged.
- **Outputs:** a findings list, not a single verdict.

### RUN QA_PERFORMANCE

- **Purpose:** verify latency/throughput/capacity under realistic and peak
  load.
- **Inputs:** load profile, target SLOs, current infra limits.
- **Checks:** latency within SLO, no unbounded queue growth, graceful
  degradation under provider slowdown.
- **Outputs:** PASS/FAIL + measured numbers.

### RUN QA_RELEASE

- **Purpose:** the final aggregate check immediately before a release
  decision.
- **Inputs:** every gate's latest verdict, the release readiness contract
  (Section 10).
- **Checks:** all required gates for the target release tier (private beta /
  paid beta / production, per Section 6) are present and non-FAIL.
- **Outputs:** feeds `RUN RELEASE_GATE`.

### RUN SECURITY_REVIEW

- **Purpose:** independent security verification, may run standalone or as
  part of `SECURITY_GATE`.
- **Inputs:** diff, dependency manifest, auth/storage/API surface touched.
- **Checks:** the Section 5 CutSell Security & Privacy Engineer ownership
  list.
- **Outputs:** PASS/FAIL, may independently block release regardless of any
  other verdict.

### RUN SRE_READINESS

- **Purpose:** verify operability before a release tier increase.
- **Inputs:** deployment config, runbooks, observability/alerting setup.
- **Checks:** the Section 6 CutSell SRE/DevOps ownership list; rollback and
  incident-response paths are tested, not just documented.
- **Outputs:** PASS/FAIL + Standard Review Report.

### RUN COST_REVIEW

- **Purpose:** verify economics before a spend-increasing change or a beta
  tier that exposes real billing.
- **Inputs:** COGS model, cost ceilings (e.g. `hybrid_provider_settings.py`
  ceilings), retry-amplification analysis.
- **Checks:** the Section 8 CutSell FinOps ownership list; confirms no safety
  guard was loosened to reduce cost.
- **Outputs:** PASS/FAIL + Standard Review Report.

### RUN EDITORIAL_ACCEPTANCE

- **Purpose:** human judgment of actual rendered output quality.
- **Inputs:** the rendered deliverable(s), Human Gold where applicable.
- **Checks:** the Section 1.11 evaluation list.
- **Outputs:** one of `PUBLISH_WITHOUT_CHANGES` / `MINOR_CORRECTIONS` /
  `MAJOR_CORRECTIONS` / `REJECT_OUTPUT`, plus `HUMAN_EDIT_REDUCTION_PERCENT`
  where a manual baseline exists.
- **Blocking:** `REJECT_OUTPUT` blocks release of that output regardless of
  every other gate's verdict.

### RUN BETA_READINESS

- **Purpose:** aggregate check before private or paid beta (Section 6).
- **Inputs:** every gate relevant to the target beta tier.
- **Checks:** all required gates for that tier are present and non-FAIL.
- **Outputs:** GO / GO_WITH_ACCEPTED_RISK / NO_GO for that tier specifically.

### RUN RELEASE_GATE

- **Purpose:** the final go/no-go decision for production.
- **Inputs:** the full release readiness contract (Section 10).
- **Checks:** every required gate is GO or explicitly GO_WITH_ACCEPTED_RISK
  with a named accepting owner; P0/P1 counts are both 0.
- **Outputs:** `GO` / `GO_WITH_ACCEPTED_RISK` / `NO_GO`.

## 8. Standard review report

Every command in Section 7 reports in this shape:

```
ROLE:
MODE:
BUILD/SHA:
ENVIRONMENT:
SCOPE:

EVIDENCE REVIEWED:
...

CHECKS:
...

PASS:
...
FAIL:
...
SKIPPED:
...

P0:
N
P1:
N
P2:
N
P3:
N

KNOWN LIMITATIONS:
...

VERDICT:
PASS
PASS_WITH_KNOWN_ISSUES
FAIL

RECOMMENDATION:
...
```

## 9. Release readiness contract

Contract/schema only -- no runtime validator, no UI, per this document's
scope. A future directive may implement a module that produces/validates
documents in this shape; this section only fixes what that shape must be.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "CutSell Release Readiness Contract",
  "type": "object",
  "required": [
    "build_sha",
    "engine_gate",
    "qa_gate",
    "security_gate",
    "reliability_gate",
    "editorial_acceptance_gate",
    "economics_gate",
    "P0_count",
    "P1_count",
    "P2_count",
    "known_risks",
    "accepted_risks",
    "release_status"
  ],
  "properties": {
    "build_sha": {"type": "string", "description": "Git commit SHA the readiness verdict applies to."},
    "engine_gate": {"$ref": "#/$defs/gate_result"},
    "qa_gate": {"$ref": "#/$defs/gate_result"},
    "security_gate": {"$ref": "#/$defs/gate_result"},
    "reliability_gate": {"$ref": "#/$defs/gate_result"},
    "editorial_acceptance_gate": {"$ref": "#/$defs/gate_result"},
    "economics_gate": {"$ref": "#/$defs/gate_result"},
    "P0_count": {"type": "integer", "minimum": 0},
    "P1_count": {"type": "integer", "minimum": 0},
    "P2_count": {"type": "integer", "minimum": 0},
    "known_risks": {
      "type": "array",
      "items": {"type": "string"}
    },
    "accepted_risks": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["description", "accepted_by", "severity"],
        "properties": {
          "description": {"type": "string"},
          "accepted_by": {"type": "string", "description": "The named canonical role accepting the risk."},
          "severity": {"type": "string", "enum": ["P2", "P3"]}
        }
      }
    },
    "release_status": {
      "type": "string",
      "enum": ["GO", "GO_WITH_ACCEPTED_RISK", "NO_GO"]
    }
  },
  "$defs": {
    "gate_result": {
      "type": "object",
      "required": ["status"],
      "properties": {
        "status": {"type": "string", "enum": ["GO", "GO_WITH_ACCEPTED_RISK", "NO_GO", "NOT_APPLICABLE"]},
        "verdict_ref": {"type": "string", "description": "Pointer to the Standard Review Report that produced this status."}
      }
    }
  }
}
```

`release_status` is `GO` only when every applicable gate is `GO`, `P0_count`
and `P1_count` are both `0`, and `accepted_risks` fully accounts for every
`P2` left outstanding. `GO_WITH_ACCEPTED_RISK` requires at least one gate at
`GO_WITH_ACCEPTED_RISK` with a corresponding `accepted_risks` entry.

## 10. Existing QA assets (reference, not duplicated)

These already exist and already do real work. This document does not
reimplement them -- it places them under the roles and gates above.

- **CleanCutBench** (`tests/test_cutsell_clean_cut_core_evaluation_suite.py`,
  `tests/test_cutsell_d050c1_5_full_cleancutbench_parity.py`) -- the general
  editorial test suite, LEGACY+AUTHORITATIVE, 54 fixtures each. Feeds
  `ENGINE_GATE` and `QA_REGRESSION`.
- **Human Gold** -- the oracle regression manifest used in every Modal
  canary since D-044/D-050. Feeds `EDITORIAL_ACCEPTANCE_GATE`; per CLAUDE.md,
  never fed into runtime production logic.
- **Architecture Validator** -- the CI check confirming the D-021 canonical
  component map is intact (see D-053 Fix 1). Feeds `ENGINE_GATE`.
- **StoryValidator** (`final_story_coherence_validation.py`) -- contradiction
  invariant, idea coverage, lost-semantic-atoms coverage ledger, lost-
  critical-claims coverage, same-idea paraphrase credit (D-061). In
  AUTHORITATIVE mode's post-resolver pass it is validation-only (D-090):
  it reports and blocks, never edits the resolver's selection; the
  `post_authority_validation.py` signature invariant enforces that. Feeds
  `ENGINE_GATE` and `QA_ENGINE`.
- **FinalEditReviewer** (`final_edit_reviewer.py`) -- the bounded PASS/FAIL
  review before Selection Freeze (D-024). Feeds `ENGINE_GATE`.
- **Selection Freeze** -- the point past which no semantic membership
  mistake may be repaired (CLAUDE.md doctrine, D-020). The authority
  `RELEASE_GATE` ultimately defers to for "was this draft ever safe to
  render."
- **PostRenderWatchListenQC** (`live_render_qc.py`) -- real ffmpeg/ffprobe-
  based physical QC on the actual rendered file (D-026/D-030). Feeds
  `EDITORIAL_ACCEPTANCE_GATE` and `RELIABILITY_GATE`.
- **Semantic Ledger integrity tests** -- coverage over `semantic_ledger.py`'s
  read-only construction and decision recording (D-050C1 family). Feeds
  `ENGINE_GATE`.
- **Unified Resolver tests** -- coverage over `realization_resolver.py`
  (D-050C1.5/C1.6/C2/C3, D-058 Phase 2). Feeds `ENGINE_GATE`.
- **LEGACY/AUTHORITATIVE regression suites** -- the dual-mode CleanCutBench
  run confirming the shadow (LEGACY) and authoritative resolver paths agree
  (D-050C1.5 parity). Feeds `QA_REGRESSION`.
- **Modal canaries** -- the one-authorized-run-at-a-time live Video00
  validation pattern used throughout D-044 through D-061. Feeds `QA_ENGINE`
  (post-canary) and `EDITORIAL_ACCEPTANCE_GATE`.
- **Canonical identity/provenance tests** -- fragment-identity and
  provenance-aware duplicate detection (D-030-era work, `canonical_identity.py`).
  Feeds `ENGINE_GATE` and `RELIABILITY_GATE`.

## 11. D-061 handoff (historical checkpoint)

Recorded at D-062; the QA_ENGINE pass it calls for was run and its verdict
is in `docs/CUTSELL_DECISIONS.md`. Kept as history under Section 12.5.

D-061 (semantic-equivalence-aware validation) has already run and is
committed on `feature/runpod-pod-on-demand`. This document does not modify,
reinterpret, or rerun it.

Per this operating model, the correct next formal action against the
existing D-061 build is:

**`RUN QA_ENGINE`** against the exact D-061 implementation/commit already on
this branch, independently reviewing: implementation, tests, negative cases,
semantic-safety guards, Freeze behavior, post-Resolver immutability,
regression evidence, claim-equivalence arbiter wiring, and same-idea
paraphrase-credit behavior.

Only after that QA_ENGINE verdict is recorded may a paid Modal canary for
D-061 be authorized.

## 12. Continuous autonomous orchestration and escalation (D-091)

This section registers how the roles above are ORCHESTRATED between
checkpoints. It changes no role's ownership and no gate's checks.

### 12.1 Canonical continuity rule

Within an already-authorized technical scope, the orchestrating agent MUST
continue autonomously. It does not wait for "continue", "proceed", "what
next?" or a newly relayed prompt when the next action is an ordinary
technical consequence of the currently authorized objective.

Canonical loop: diagnose -> reproduce -> fix -> targeted tests -> QA -> (if
QA finds an in-scope defect, return to Engineering) -> retest -> CI /
offline qualification where applicable -> analyze evidence -> continue to the
next proven root cause within scope. A technical checkpoint is not
automatically a stopping point.

### 12.2 Product Owner role

The Product Owner defines product behavior, scope, priorities, acceptable
risk, paid-compute budget, release decisions and human editorial acceptance.
The Product Owner is NOT required to choose which function to inspect, which
unit test to write, how to reproduce a bug, to relay Engineering findings to
QA or QA findings back to Engineering, or to say "continue" after normal
technical steps.

### 12.3 Engineering <-> QA auto-loop

Engineering implements; QA independently challenges (Section 2 is
unchanged). Their hand-off is orchestrated automatically: when QA returns
FAIL, or a finding within the authorized scope, the agent returns it to
Engineering, fixes it if within scope, reruns tests, then reruns QA.
Engineering never rewrites a QA verdict; diagnostics and reporting keep the
two roles' outputs separate and disclose when one session carries both.

### 12.4 Stop conditions (ask the Product Owner ONLY when at least one holds)

- A. PRODUCT DECISION REQUIRED: the fix changes product behavior, UX
  doctrine, user workflow, editorial policy, pricing, scope or acceptance
  criteria.
- B. SAFETY / AUTHORITY CHANGE REQUIRED: the next fix would weaken Freeze,
  critical-content protection, security, privacy, authorization, tenant
  isolation, claim safety or another canonical safety contract.
- C. PAID COMPUTE OUTSIDE AUTHORIZATION: a Modal/RunPod/provider run beyond
  the currently approved paid-run count/budget/scope.
- D. PROTECTED REPOSITORY ACTION: merge PR, write `main`, deploy
  production, TestFlight/App Store release, destructive repository operation.
- E. P0/P1 ACCEPTED-RISK DECISION.
- F. HUMAN EDITORIAL ACCEPTANCE: a real rendered artifact needs human
  watch/listen/product judgment.
- G. TRUE SCOPE BOUNDARY: the proven next root cause belongs to a materially
  different project objective.

Otherwise: continue.

### 12.5 Task-local STOP semantics

A directive-local "Then STOP", "No code", "No RAW" or "Report only" limits
THAT directive's authorized actions and never disables the global continuity
contract. REPORT ONLY: forensic work stops before code modification; an
implementation directive the Product Owner already authorized may then
proceed under that authorization. NO RAW: offline diagnose/fix/test/QA/CI
continue; stop only when the next required step is a paid run. NO CODE:
investigation and documentation continue; implementation waits for
authorization. Historical STOPs recorded in `docs/CUTSELL_DECISIONS.md` are
not permanent relay requirements.

### 12.6 Paid compute budget contract

Default: paid compute NOT authorized. The Product Owner may authorize a
bounded campaign (for example ONE_CANARY, THREE_RUN_STABILITY_BATTERY,
MAX_PAID_RUNS=N, MAX_BUDGET_USD=X). Within it the agent schedules exactly
the approved runs without asking again and never exceeds count, budget or
scope. A failed paid run does not stop offline diagnosis and fixing.
Additional paid compute is stop condition C.

### 12.7 Root-cause continuity

A defect exposed by a test or canary within the SAME authorized objective is
investigated and fixed automatically. An unrelated defect is recorded
separately without silently expanding scope, and the authorized objective
continues where possible. No endless D-xxx work is opened merely because
another observation exists.

### 12.8 D-xxx decision log role

D-xxx entries remain durable engineering decisions and checkpoints. They are
not manual turn-taking tokens: creating the next sequential entry does not
by itself require Product Owner intervention while the work stays within
authorized scope and crosses no stop condition.

### 12.9 Status reporting

While continuing, progress is reported compactly as: CURRENT OBJECTIVE,
CURRENT STAGE, LAST VERIFIED RESULT, CURRENT ROOT CAUSE, NEXT AUTOMATIC
ACTION, HUMAN ACTION REQUIRED (YES/NO, with the stop-condition letter).

### 12.10 What this section preserves

Repository protections (Section 12.4 D and `CLAUDE.md`), QA separation
(Section 2), security and release gates (Sections 3, 7, 9), paid-compute
limits (12.6), Human Editorial Acceptance (role 11, gate
`EDITORIAL_ACCEPTANCE_GATE`) and the current engine doctrine are unchanged.

## Change rule

Extend this document by adding new roles, gates, commands, or QA-matrix
areas as the product grows -- do not silently redefine an existing role's
ownership or an existing gate's checks. If a role's authority needs to
change, record why in `docs/CUTSELL_DECISIONS.md` the same way any other
canonical decision is recorded, and update this file's specific section --
never rewrite this document wholesale.
