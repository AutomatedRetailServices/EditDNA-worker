# CutSell.ai — Current State

Product generation: **CutSell.ai 7**

## 2026-09-24 GPT + WhisperX five-run stability qualification (authorized)

The Product Owner requested: “podrias hacerlo otra vez? hacer 5 pruebas con
este? sistema?” This authorizes FIVE new independent full Video00 trials on
`feat/gpt-whisperx-video00`, including the per-job source-cache correction at
`e1d6892983d64d12191d58df7545668d85bbfef1`. The earlier completed trial is not
one of these five. No sixth paid invocation, production promotion or PR #25
merge is authorized by this batch.

The QA transport uses one checkout and one private worker-template snapshot,
the same canonical `run_op("focused")` engine and microtrim payload, five
sequential fresh Modal apps/providers, unique benchmark IDs, L4, retries=0,
and the existing 5400-second remote bound. Claims prevent retries even after
failure; uncertain termination blocks subsequent dispatches. Transcripts are
not shared across trials. The same-source cache operates only within each
job. Each trial preserves its own video, raw result, logs, build/source proof,
technical QC and unchanged editorial/Gold QA. Paid workflow reruns are blocked.
Code under `cutsell_worker` and the canonical Modal wrapper are unchanged by
this qualification harness. **Completed:** run `36054192894`, test SHA
`99c02183246c3e2caa96c006ad81e7c966ba4a51`; five full results and five diagnostic
MP4s, all blocked by physical silence QC. Editorial scores: 6/11, 5/11,
8/11, 5/11, 8/11. The same three prior failures remain in all five. Zero
zero-duration ASR words in every trial; 65 distinct GPT requests and one
same-source cache reuse per job. Five selections differ despite lexical
disagreement of only 0–0.7764%. No production promotion or sixth run.
The runner's missing CPU ffprobe caused a separate post-render collection
error; all exact outputs were recovered and verified locally without GPU
re-execution. See `CUTSELL_GPT_WHISPERX_FIVE_TRIALS.md` for evidence and limits.

## Isolated ASR provider evaluation (not integrated)

The five RAW #127 attempts showed five distinct ASR content hashes on the
same original and decode fingerprint. On the isolated
`feat/asr-provider-evaluation-and-replay` branch, the RAW entry point now
honors the existing ASR decode selector and the ASR-only manual harness can
compare `medium`, `large-v3`, and opt-in GPT/Deepgram text candidates.
No production provider or source transcript snapshot was changed, and no
paid ASR comparison was dispatched. See
`docs/CUTSELL_ASR_PROVIDER_EVALUATION.md` for the remaining alignment,
replay and cross-video gates.


This file is the operational checkpoint. Update it whenever the active benchmark, release gate, branch state or major implementation focus changes.

## Repository

- Repository: `AutomatedRetailServices/EditDNA-worker`
- Active branch: `cutsell/mobile-v1-clean`
- Active PR: `#25` (Draft, open, unmerged)
- Base: `main`
- Base SHA remains `2fb13e5aa228e8e525b942a9b49182032b797e61`
- PR #24 remains historical/reference backup and must stay untouched.

## Watch+Listen delivery-authority integration checkpoint — D-288 through D-288.4.1 (current)

**Canonical HEAD:** `781a86346c106b1aab31674605d1cf4c32a5819c`

`cutsell/mobile-v1-clean` now contains the full D-288 → D-288.4.1 chain
(typed `watch_listen_status` delivery gate on the real export path; persistent,
recoverable HUMAN_REVIEW_REQUIRED pending reviews in private S3 + Redis; the
real authenticated pending-review HTTP handlers — query, media preview,
decision, resume — deriving identity only from the verified session; atomic,
versioned pending-record transitions with the publish claim taken BEFORE any
external effect; atomic render-version/notification reuse via one Redis Lua
primitive; sha-keyed immutable reviewed MP4 objects; the original job's
`job_started_at` preserved through a resumed delivery; the disconnected
`perceptual_repair_cycle.py` foundation with its documented preconditions and
pending issues; and the job-scoped EditorialSlotResolution observability),
brought in from `audit/watch-listen-delivery-authority` in a controlled,
Product-Owner-authorized integration gate.

Integration method and safety record:
- fast-forward only (`f012beed..781a8634`, exactly 13 commits, plain push of the
  exact SHA) — no rebase, no merge commit, no cherry-pick, no force push;
- all four expected states (canon `f012beed`, audit `781a8634`, `main`
  `2fb13e5a`, PR #25 OPEN/DRAFT/UNMERGED) and a clean worktree were verified
  before, and the three remote SHAs re-validated by `ls-remote` immediately
  before, the push;
- every workflow trigger was parsed against the whole 13-commit diff: only
  the unpaid PR CI and the paid `cutsell-video00-raw-v5-auto-microtrim.yml`
  (push to canon, path `universal_clean_cut_validation.py`) would fire; the
  latter (workflow id 342360588) was disabled by API before the push (prior
  state recorded `active` → verified `disabled_manually`), no YAML modified,
  and restored to exactly `active` after the push — its run count stayed at
  148 with no run for `781a8634`, i.e. **no paid RAW was triggered**;
- `main` remains untouched at `2fb13e5aa228e8e525b942a9b49182032b797e61`;
- PR #25 remains OPEN / DRAFT / UNMERGED (head advanced 2263 → 2276 commits).

Verified state at `781a8634`:
- **CutSell Clean Worker CI — PASS** (run 35850126074);
- **CutSell iOS CI — PASS** (run 35850126075);
- existing `mobile/ios/` work untouched; no selection/grouping/ranking policy or
  baseline changed by the integrated range;
- `perceptual_repair_cycle.py` has zero live callers — NOT activated.

**Not yet true, do not assume otherwise:**
- no RAW has been run on this head; the delivery-authority chain is proven
  offline (8841 passed, the same 5 pre-existing unrelated failures) and by CI,
  not by a rendered artifact;
- the D-288 audit's item 5 (two duplicate-realization survivals on RAW #122,
  run 35799404391 at `f012beed`) is addressed OFFLINE on the isolated branch
  `fix/editorial-realization-closure` (D-289 + D-289.1 + D-289.2: contained-restatement
  closure at retry-family formation with a faithful RAW #122 replay --
  continuation chains as one realization, preservation through the claim-
  coverage authority, the existing 0.85 pairwise bar; case 1 escalated as a
  product decision because Gold, Cut.ai and the locked baseline all KEEP the
  pimples fragments) — NOT integrated; RAW #123 (run 35866604610 on
  `0b1572a8`, D-289.3) produced NO MP4 and its result JSON could not be read
  from this environment (artifact host blocked, invalid S3 keys): the aside
  was kept ungrouped, the restatement head kept exactly as on RAW #122 and
  its tail removed before grouping — D-289.x is NOT validated on video.
  D-289.4 fixes the two verified pre-grouping causes offline (clause
  reconstruction spacing that erased a negation and made the winner's own
  CRITICAL claim read as lost; the cross-group cleanup now judges a
  continuation chain as one unit and never removes a tail alone); the
  contradiction primitive's negation scope over RAW #123's comma-run winner
  is recorded as residual R-289.4a. D-289.5 replays RAW #123 from the Product
  Owner's emitted-diagnostics package (the single Freeze blocker was the
  fused "por esono" claim; W-R was never asked; T removed as alternate 0.85),
  guards the joint removal per member and by realization preservation, and
  shows the replay removes that blocker while the restatement R+T remains a
  Level-2 co-keep pending a W-R verdict no run has recorded. D-289.6 judges
  the unit's relation at sentence granularity (an inverted causality is no
  longer "covered"), corrects D-289.5's replay fake (an omitted pair had
  invalidated the whole arbiter batch), shows the run's W-R pair was deferred
  by the per-group pair cap, and traces why even a confirming W-R verdict is
  vetoed by the contradiction primitive on this transcript -- so no RAW is
  proposed to obtain that answer. D-289.7 closes that residual (R-289.4a):
  `claim_coverage` and `contradiction_signal` now scope negation/number by
  PROPOSITION through one shared segmentation (`semantic_claims.proposition_
  units`/`proposition_scope_units`; corrective-contrast connectors split;
  relational references keep sentence scope; no connector -> fails closed as
  before), so the comma-run winner reads identically with a comma or a
  period; a one-unit-per-side bridge now takes the LATER unit as the
  newcomer (guard 8). Traced offline with labelled simulated answers: guard
  7's veto disappears, the path consults the existing claim arbiter on R+T's
  claim (0.5556, ambiguous band), and the outcome still depends on three
  unrecorded real answers (W-R pairwise, the claim verdict, the D-085 probe
  if declined) -- not integrated; no baseline change. D-289.8 closes the
  regression that segmentation introduced (a negated matrix clause "no es
  cierto que ..." was cut from its complement at a temporal connector): the
  shared splitter keeps a span whole when its left side ends on a
  complementizer/function word or a belief verb, so claims and scope keep
  the negation with what it negates; full regression 8956 passed, the same
  5 pre-existing failures. RAW #124 (run 35886004885 on `693c7b25`, D-289.9,
  the one authorized Modal run): Freeze passed and a deliverable-candidate
  MP4 was produced; on the ladder W and R form ONE family, W wins and the
  restatement R+T is discarded (Level 3, matching Human Gold) -- the
  restatement duplicate is gone from the rendered edit for the first time;
  but the 18-check Human Gold QA FAILED (inferred: the pimples family's
  monolith won over the later take, 20.1 s Level 1) and the abandoned
  stomach attempt was kept ungrouped (5.9 s Level 1): Level-1 selection
  44.9 s vs 21.69 s on RAW #122. Offline, both decisions' deterministic
  inputs are byte-identical at RAW #122's code and this head (run-varying
  hybrid labels/arbiter answers inferred, not proven). The MP4, result JSON
  and QA report are unreachable from the session container (artifact host
  blocked, invalid S3 keys): HUMAN WATCH+LISTEN and the QA report are
  pending Product Owner attachment. Not integrated; not relaunched.
  D-289.10 (from the Product Owner's RAW #124 evidence packages): the
  pimples complete-window conflict (two complete contexts choosing opposite
  winners, D-150 ABSTAIN_CONFLICT) no longer falls to a NON_DECISIVE
  DeliveryScore pick -- the ladder stops (`unresolved_semantic_winner_
  conflict`, CONFLICTED), the Ledger carries each window's winner verdict
  and the Resolver's existing conflict branch returns REVIEW_REQUIRED, which
  blocks the render for that idea pending human choice (Product Owner
  decision: accept the block or authorize the bounded finalist authority);
  the stomach split is fixed at its demonstrated cause (a deterministic
  measured-pause restart merge lost its kind on the way to the cohesion pass
  and was re-judged by the component probe) -- one family, no probe, G kept,
  the abandoned attempt discarded through the authoritative resolver and the
  KEEP/DISCARD fold; the QA harness gains `required_realization` (presence
  of the realization, not shared content); baselines untouched. No RAW.
  D-289.11 (from the updated MP4 review, user-reported CTA repetition): the
  standalone CTA no longer re-opens with the words the preserved conclusion
  closed on ("… Así que cuídate." then "Por eso cuídate, aliméntate …"):
  the pre-Freeze boundary owner (`final_boundary_authority`) trims the
  LATER clip's re-opened closing at its first remaining word start, bounded
  by recency, by an ASR punctuation/pause break, by a content floor, and
  refused for numbers/negations/distinct additions (every refusal recorded);
  the conclusion is never edited; the QA harness gains `repeated_closing_
  absent` plus an unconditional warning scan (presence is not uniqueness);
  baselines untouched; offline proof on recorded texts with SYNTHETIC word
  timings. The aside A remains an editorial review item. No RAW.

## D-290 editorial acceptance and controlled input comparison (isolated)

Branch `fix/video00-stable-editorial-oracle` (based on D-289.11 at
`98a961a4`) adds an independent, CPU-only Video00 acceptance manifest,
evidence-aware QA checks and a read-only run comparator. The historic
18-check Human Gold manifest and Selection Lock are unchanged. Default-OFF
finalist flags remain unchanged; D-290.1 below contains the opt-in prosodic
comparison correction. This branch has not been integrated or run on Modal/RunPod.

The recorded runs demonstrate **two separate selection defects**: RAW #115
kept both pimples deliveries in DIFFERENT retry families (no competition);
RAW #118/#124 placed them in competition but an internally NON_DECISIVE
DeliveryScore favored the rejected monolith; RAW #122 had two agreeing
semantic winners for the later clean take and selected it. The video output
alone does not establish an independent, reliable tie-breaker. The one
0.71-second source pause plus hand movement in the rejected take fails the
existing physical-cleanliness proof bar (1.2-second interior silence or
an independently corroborated disengagement). Lowering that floor would
turn normal expressive gestures into fabricated failed takes.

Old QA falsely passed RAW #118 on the pimples monolith because a comma
changed the forbidden literal, and it marked the later preferred take
present when only shared words were selected. The new manifest checks
rejected-realization identity and source interval independently, the
later preferred take by the union of selected source coverage (valid
re-chunking is allowed), the abandoned
stomach attempt, the full later gynecologist take, sonography opening,
one-delivery `No quiero` (editorial continuity; the old composite retains
semantic negation), the user's single-percentage preference (Cut.ai differs
from Gold), and a unique `cuídate` WITH final actions preserved. The CTA
checks allow the D-289.11 trim when the earlier conclusion says `cuídate`
once and the later CTA keeps the advice. Results on the recorded selections:
RAW #115 4/11, #118 5/11, #122 6/11 and #124 3/11 (the last comparison
uses the rendered-segment source intervals reconstructed from the MP4
evidence package; no complete #124 result JSON was available locally).
These are separate acceptance checks, not retroactive changes to prior QA.

The source key and ASR config match #115/#118/#122, but their canonical ASR
hashes differ and no source-video byte checksum is in those historical
results. Future benchmark runs now hash the downloaded SOURCE bytes before
ASR and persist the exact timed raw ASR segments/words in the full result
JSON. This also goes into the S3 result object and GitHub human-review
artifact; verify their actual access policy before treating raw speech as
confidential. The compact job result surfaces only the checksum, never the
full raw transcript. A future exact replay still needs every consulted
semantic arbiter's request/verdict (including declines) bound to those
inputs; the ASR snapshot by itself does not prove a selection result.
The new read-only comparator reports
source-interval differences and refuses to
attribute them to a code regression without matching immutable inputs and
a controlled replay. The latest selection fixes D-289.10/11 remain offline
and unvalidated by a new MP4. A reliable automatic BestTake needs a
candidate-level audiovisual comparison with negative controls for natural
pauses and meaning preservation; a post-render Watch+Listen verdict cannot
choose a take that has already been discarded. Do not claim the video is
fixed on the strength of an improved QA manifest.

D-192 previously changed this same pimples family's winner to the preferred
later take when its bounded finalist, prosodic diagnostics and authority
flags were ON; all were OFF in #115/#118/#122/#124. The three prosodic
categories (continuity, hesitation and restart) can all result from one
measured pause starting at 0.60 s, so D-192 alone does not establish safety
for a natural rhetorical pause. D-289.10 also routes complete-window winner
disagreement to a later REVIEW_REQUIRED gate. The next offline gate must
prove independent interruption evidence and trace a meaning-safe resolution
through that existing authority. No flags or protected branches changed.

### D-290.1 — pause-only finalist preference contained locally

The synthetic audio reproduction confirmed that one ordinary 0.71 s pause
could generate continuity, hesitation and restart votes and flip the
winner through D-184/D-191. D-188 now retains the descriptors but does not
certify a preference from them. Differing/unknown Phase-A descriptors yield
INSUFFICIENT_EVIDENCE (`independent_in_span_disruption_evidence`); equal known
descriptors remain NEAR_EQUAL. The unscoped language-restart boolean is not
independent proof of a failure inside the candidate. Actual V2 evidence
remains eligible through the existing authority; defaults remain OFF.

This is a safety correction, not an automatic answer to the pimples family:
D-192's historical prosody-only success is no longer a qualified enabling
argument, and D-289.10's conflicting semantic verdict remains visible.
No source-specific exception, threshold change, RAW or integration. Work
remains local; the prior push permission stop has not been bypassed.
Verification on code commit `dbdbc65`: 254 targeted tests passed;
`tests/test_cutsell_*.py` plus `tests/test_video00_regression_qa.py` completed
with **8425 passed, 10 skipped, zero failures** (511.96 s), on a clean,
unchanged committed tree. Independent review found no in-scope blocker.
This is the complete named regression set, not every test in `tests/`.
No rendered-video improvement is claimed.

## Mobile backend integration checkpoint — D-277 through D-282A

**Canonical HEAD:** `86a058d20567078af918d7f8dc8d02da84a85417`

`cutsell/mobile-v1-clean` now contains the full D-277 through D-282A timeline/
mobile-backend contract chain (timeline architecture/composition, manual
B-roll + layered audio/voice-over composition foundation, timeline asset
registry, faceless/product visual-mode safety, live asset persistence, the
FastAPI mobile timeline bridge, and its upload/export authority hardening),
brought in from `feature/runpod-pod-on-demand` in a controlled integration
gate.

Integration method and safety record:
- fast-forward only (`git merge --ff-only`) — no rebase, no merge commit, no
  cherry-pick, no force push, at any point in this checkpoint;
- the branch was a strict git ancestor of the backend source before this
  merge, so the fast-forward was structurally conflict-free;
- the prior session's local-only commit `9f97531` (a RAW workflow
  evidence-preservation fix, unrelated to this integration) was explicitly
  **not** recovered, rebased, cherry-picked, or pushed — it remains
  quarantined under the local tag `quarantine/9f97531-raw-evidence-fix`,
  outside this branch's history;
- `main` remains untouched at `2fb13e5aa228e8e525b942a9b49182032b797e61`;
- PR #25 remains OPEN / DRAFT / UNMERGED throughout.

Three real, pre-existing environment/CI gaps were exposed by running the
existing "CutSell Clean Worker CI" job against this combined code for the
first time (none are editorial/Selection logic, none were introduced by this
integration — each was root-caused and fixed as its own commit, verified
against a from-scratch CI-parity environment before pushing):
1. `numpy`/`Pillow` were hard transitive imports of `cutsell_worker` (since
   D-187/D-274b/c) but never declared in `requirements.cutsell.api.txt`.
2. The hosted CI runner no longer ships `ffmpeg` by default; CI now installs
   it explicitly, and one test helper that crashed instead of skipping when
   ffmpeg was absent was fixed to fail closed.
3. Eight pre-existing test files hardcoded this Claude Code sandbox's own
   absolute checkout path instead of computing the repo root; fixed to
   compute `Path(__file__).resolve().parents[1]`, verified from an unrelated
   working directory.

Verified state at `86a058d2`:
- **CutSell Clean Worker CI — PASS** (full `tests/test_cutsell_*.py`,
  compileall, staging API container health check);
- **CutSell iOS CI — PASS** (real Xcode Simulator build);
- existing native SwiftUI `mobile/ios/` work (29 files: auth, upload,
  projects, processing, preview, timeline editor, export) preserved
  unmodified by this integration;
- no RunPod/GPU/provider spend — two GPU-cost workflows
  (`CutSell Video00 Unified Selection RAW`, `Round24 Serverless GPU Gate`)
  were auto-triggered as a path-filter side effect of the fast-forward and
  cancelled immediately while still in their Docker-build stage, before any
  paid compute began.

**Not yet true, do not assume otherwise:**
- calibration has **not** started;
- Cut.ai parity is **not** proven for anything in this checkpoint;
- TestFlight/production readiness is **not** complete (`cutsell-ios-ci.yml`
  still builds Simulator-only, `CODE_SIGNING_ALLOWED=NO`, no
  signing/archive/App Store Connect step exists);
- no mobile UI was added or changed by this integration — the backend/API
  contracts are now available on this branch, ready for the mobile Timeline
  UI implementation gate, but that implementation has not started.

## Mobile D-279 iOS registry integration checkpoint (current working gate)

D-279's formal timeline asset registry is now wired into the existing native
Swift consumer without conflating it with the older draft-embedded preview
artifacts:

- `TimelineAssets.swift` now defines typed, client-safe D-279 asset roles,
  media kinds, qualification states and list-response models matching the
  hardened `GET /v1/projects/{project_id}/timeline-assets` contract;
- the client requests that list with the authenticated project/user scope and
  rejects a response whose `project_id` does not match the requested project;
- only `READY` assets enter the editable B-roll, voice-over and primary-source
  role catalogs;
- raw storage references and technical metadata references remain server-only;
- the prior draft `filmstrip`/waveform reader is now named
  `SourcePreviewAssetCatalog`, making its preview-only authority explicit;
- `DraftEditorViewModel.load()` loads the existing draft and the formal asset
  registry together, preserving the current editor while making the registry
  available for the Mobile V1 Timeline UI gate.

Directed verification: **135 passed** across the new iOS source-contract tests,
D-279 registry tests, D-281 live-wiring tests and D-282 bridge tests. A real
Swift/Xcode Simulator build remains required on the macOS CI runner before this
checkpoint is accepted.

**Exact next product gate after Xcode CI:** bind the ready role catalogs to the
canonical Figma editor track/add-sheet interactions (Main video / Voice-over /
Overlay), while keeping upload creation, placement mutation and keyframes as
their own explicitly tested contract steps rather than simulating unsupported
runtime behavior.

## Current focus

**Clean Cut Core V1 migration (idea-first, SWAP out of scope).** See
`docs/CUTSELL_DECISIONS.md` D-019 (SWAP out of scope until explicitly reintroduced)
and D-020 (idea-first architecture). Superseded the prior "Universal Clean Cut →
Unified Whole-Video Selection" mission: an accepted offline audit of a real RAW run
(the whole-video Gemini reasoner era) found pairwise, fixed-budget retry-family
discovery systematically under-covers retries later in a video, and separately Human
Watch+Listen found that architecture editorially insufficient. The whole-video Unified
Selection reasoner is now deactivated in the active path (kept only behind
`CUTSELL_CLEAN_CUT_CORE_V1=0` for rollback); Gemini is a bounded semantic arbiter only.

As of this checkpoint: Clean Cut Core V1's first controlled RAW (33345946000, head
`0ea0adf`) ran and was Human Watch+Listen reviewed. Result: the flagship hereditary-
cancer contradiction resolved cleanly and semantic-equivalence merges improved 8x, but
real content loss was found (a papillary-cancer diagnosis confirmation, a sonography
transition, and a pimples/rash symptom beat all missing from the final KEEP timeline)
that `final_story_coherence_validation` did not catch. See D-022 for the full root
cause (Hybrid session cleanup deletes per-clip before IdeaClusterer ever runs, with no
idea-coverage awareness of its own, and StoryValidator's coverage check was scoped only
to post-grouping state) and the general fix implemented (a lost-semantic-atoms coverage
ledger checking every discarded clip's content directly against the final KEEP text,
independent of which stage discarded it). Full consolidation of the legacy hybrid_*
authorities into one canonical CompositeResolver component is now DONE (D-023 --
turned out to be 19 hooks, not the ~14 first estimated; composite_resolver.py calls
each hook's own real installer once, in the historical order, rather than
hand-transcribing their logic, after hand-transcription was found to have missed
five of them). The coverage ledger remains the structural backstop regardless.

CanonicalEditPlan and a bounded FinalEditReviewer (D-024) now run before Selection
Freeze -- FinalEditReviewer independently catches an unresolved (non-contradictory)
duplicate idea that StoryValidator itself does not treat as freeze-blocking. Three
interface-only contracts (PostRenderWatchListenQC, the Sales/TikTok Shop
StyleProfile extension points, Finishing) are defined but not implemented or
activated anywhere.

Human review is the quality gate. Workflow success alone is not an editorial pass.
Sales-funnel/storytelling work remains intentionally separate until Clean Cut reaches reliable real-video quality.

## Benchmark history that matters

### Benchmark #39 — bad human baseline

- 16/16 processed technically.
- Human review: editorial failure.
- Core failure class: repeated failed attempts surviving, fragmentary edits, over-cutting and weak Best Take behavior.

### Benchmark #48 — first large editorial improvement

- 16/16 processed.
- 0 execution failures / 0 provider failures.
- Hybrid availability: 21/22 = 95.45%.
- Human review:
  - 00 still wrong: repeated sonography + hereditary-cancer delivery.
  - 02 major improvement, almost ready to deliver.
  - 03 cut ending too early.
  - 04 correct.
  - 05 correct.

### Benchmark #49 — technically clean, editorially failed

Exact worker:
- source `499d1c59e11ea1abe550678ae334cd46573312d1`
- image digest `sha256:01dc103a7742054b525b4ff71e720d4af220ff8a12b1a0938ed858b0bff250c5`
- workflow run `32318014133` (visible run #51)
- Pod `o5flfm9ioegdt6`, cleanup PASS

Technical:
- 16/16
- 0 execution failures
- 0 provider failures
- Hybrid 21/22 = 95.45%

Human review:
- 00 still kept competing versions instead of one winner.
- 02 regressed and reintroduced malformed failed speech.
- 03 ended at `te protegen` and cut the complete idea.
- 04 stayed correct.
- 05 stayed correct.

### Benchmark #50 — meaningful recovery, but 00 not fully solved

Exact worker:
- source `e596002dfc550c9da6d8d10a9eb47b3363272f69`
- image digest `sha256:ba14a4344e000e3a2200a1c74da32569ff53f823dd74688182f25947f7cb3431`
- benchmark run `32324942712` (visible run #52)
- Pod `uoz7exmx8tdf11`, cleanup PASS

Technical:
- 16/16
- 0 execution failures
- 0 provider failures
- Hybrid 21/22 = 95.45%

Observed editorial result:
- 00 improved strongly in sonography retries, but hereditary-cancer close still duplicated.
- 02 recovered the #48-quality path and removed `I people It was very funny`.
- 03 preserved the complete ending through `te reparan la barrera`.
- 04 remained correct/clean.
- 05 remained correct/stable.

### Benchmark #51 — architecture experiment exposed a production integration bug

Exact worker under validation:
- source `65fe16519bbf72562805d5dac0333b993c43c179`
- image digest `sha256:4593f402deb56e9128b1d058efd24515c3b91ae0712eac84aff172bc4536d99f`
- benchmark run `32345888694` (visible run #53)
- Pod `goi51dpki0m153`
- Pod cleanup step PASS

Technical result:
- 16/16 processed
- 0 execution failures
- 0 provider failures
- report and preview artifacts uploaded

Key editorial finding:
- 02/03/04/05 preserved the expected good behavior.
- 00 still selected the complete hereditary-cancer delivery **plus** the split retry (`prefix` + `continuation`).

Production diagnostics proved why:
- the new final sibling reconciliation logic itself worked in unit/regression tests;
- however `safe_group_takes_by_sessions` partitions the video into mini-sessions first and calls grouping independently inside each session;
- in Video 00 the complete hereditary delivery, split retry prefix and continuation ended in three distinct session-scoped groups;
- therefore the sibling layer never saw the three candidates together in production.

This means Benchmark #51 did **not** disprove the sibling-family architecture. It exposed that the reconciliation was installed one level too low.

## Current architecture decision

Recover the useful earlier EditDNA invariant without rolling back the modern Clean Cut architecture:

**same audience-facing idea + competing deliveries -> one sibling/retry family -> Best Take chooses exactly one selected winner**

Important:
- losers are not destroyed;
- they remain available as alternates / Swap Take;
- Composer should receive only the selected winner from a retry family;
- Story/funnel logic is not part of this decision.

The newer Clean Cut capabilities remain:
- whole-video understanding;
- attempt reconstruction;
- session boundaries;
- Hybrid/Gemini semantic reasoning;
- story-preservation guards;
- temporal completion/boundary repair;
- immutable source identity.

## Fix implemented after Benchmark #51

A new **global post-session sibling bridge** now wraps the final output of `safe_group_takes_by_sessions`:

1. session-scoped grouping still runs normally to protect true compilation boundaries;
2. all final groups are then presented to conservative sibling reconciliation at the whole-source level;
3. only strong competing-delivery evidence can merge groups;
4. no take is deleted by this bridge;
5. the existing TakeJudge/Hybrid Best Take authority chooses the one selected winner.

A production-path regression now forces the three Video 00 hereditary takes into three separate mini-sessions and requires the final grouping output to collapse them into one sibling family.

## Current validated code checkpoint

Current code head:

`140bec542dd9f0b429093810e4b85b541e35f589`

Validation:
- **CutSell Clean Worker CI #1474 — PASS**
- **CutSell iOS CI #1255 — PASS**
- new cross-session production-path sibling regression — PASS

PR #25 remains Draft and unmerged. `main` remains unchanged.

## Next gate

Code-level validation is green, but the new post-session sibling fix has **not yet been proven on real RAW video**.

Before the next paid validation:
1. keep current head green;
2. build an immutable worker image from the exact validated source;
3. stop for explicit user approval before creating a new paid RunPod Pod / benchmark;
4. after approval, run exactly one controlled 16-video RAW benchmark;
5. verify deletion of the exact Pod;
6. present 00/02/03/04/05 for human review.

Success criteria:
- **00:** hereditary-cancer competing deliveries collapse to one winner; no duplicate sonography regression.
- **02:** keeps the #48/#50 improvement and no malformed retry fragment.
- **03:** retains the complete natural ending.
- **04:** stays correct.
- **05:** stays correct.

## Sales Funnel sequencing

Do **not** reintroduce rigid funnel structure while Clean Cut is still being validated.

After Clean Cut is reliable, resume the commercial layer as a flexible narrative-understanding stage that can recognize different selling structures (for example story -> discovery -> experience -> recommendation, pain -> demo -> proof, hook -> solution -> CTA) rather than forcing every video into one fixed funnel template.

## Current execution rule

Continue automatically after each non-paid fix/status block. Stop only for:
- explicit approval before new paid infrastructure/benchmark spend; or
- human review when new preview videos are ready.

## Update rule

Whenever work advances to a later benchmark or major Brain checkpoint, update this file in the same development cycle instead of reconstructing state from chat history.

### D-291 — family-scoped confirmation of a complete-window label conflict

Same isolated branch, off `59b0a188`. The pimples family's winner flipped
across RAW #118/#122/#124 because the two overlapping 10-candidate windows
labelled it differently (M winner / L winner, or M winner / no winner); the
provider answers are data, the code defect was that a conflicted layer-5
(arbiter) answer went straight to layer 7's NON_DECISIVE DeliveryScore
(#118, #124 before D-289.10) or layer 11's review block (#124 after). D-291
asks the SAME editorial judge ONE bounded family-scoped question when D-150
abstains on a conflict (family members + one known neighbour each side,
same prompt/temperature/budget ledger, max 4 per video, `CUTSELL_FAMILY_
CONFLICT_CONFIRMATION=0` disables) and honours only an unambiguous single-
winner answer through the UNCHANGED ladder and its vetoes; anything else
leaves D-289.10's block. Symmetric (a confirmed monolith is honoured too).
Proven offline with LABELLED FAKE answers only (18 tests incl. end to end
through `build_flow_b_draft`); the real judge's family-level verdict is
unknown and the MP4 is not claimed improved. RAW #115's family-formation
defect cannot be replayed without its result JSON (`take_group_id`,
`semantic_idea_equivalence`, `distinct_idea_grouping_safety.edge_trace`).
No RAW launched; the test RAW is specified in the decision entry.

### D-291.1 — RAW #125 (run 35921819172 on `98a6b82b`): selection observed, diagnostics unreachable

The one authorized run completed (Modal OK, teardown confirmed) but the
workflow downloaded no MP4 (`PREVIEW_URI` empty) and its QA/acceptance/
pacing steps failed; only the ladder tail is reachable from this
container (artifact host denied by the network policy, AWS key invalid).
OBSERVED selection: pimples A1 + later take with the monolith discarded
(but M and L were NOT in one family -- D-291 did not decide it), gastritis
only, later gynecologist take, R+T out, aside A kept, "No"/"quiero" still
split, CTA still re-opens with "Por eso cuídate,", the acne take dropped
with a stranded "resorcina." (new Level-1 regression). Level-1 selection
21.96 s (#124 44.9, #121 37.09, #122 21.69). Nothing rendered is proven;
consistency is not proven. Next: obtain the run's artifacts (network policy
or attachment) and record the D-291 rows, the acne resolver basis, the CTA
boundary rows, the QA reports and the delivery status.

### D-291.2 — continuation chain = one realization for Ledger/Resolver (RAW #125 acne)

Reproduced on `a53459dd` with the recorded RAW #125 texts/spans and the
recorded RAW #122 labels for the same clips: the Ledger registered the acne
head and its "resorcina." tail as two realizations, the Resolver applied
the head's pre-fold `failed` window label, kept the tail alone and waived
the sentence. Fixed at the handoff: tails carry the head's `realization_id`
(`parent_realization_id` records the join) and the Ledger registers a
multi-clip realization with the complete sentence, its claims, span and
completeness. Proven offline through the real pipeline and the full path
to Freeze (boundary authority reached). No id/phrase/threshold/restoration.
Recorded, not fixed: the Resolver's id-order tie when the family's winner
label agrees with the local winner (no Ledger evidence). The RAW #125
attachment did not reach the container; the CTA on real ASR timings is
still unevaluated. No RAW launched.

### D-291.3 — RAW #125 JSON read; report corrected; chain fix verified on real data

RAW #125 blocked Freeze (`IDEA_COVERAGE_LOST` on the acne family after the
Resolver kept "resorcina." alone); `final_boundary_authority` never ran; D-291
executed once (gynecologist family, no single winner). A zero-deviation replay
from the run's own labels, scores, answers, events and timed words reproduces
the block on the pre-fix tree and, on the fixed tree (D-291.2 + the
CanonicalEditPlan realization-level validation), reaches Freeze, keeps the
acne chain, resolves stomach and percentage as RAW did, runs the boundary
authority and trims the CTA at "aliméntate" with the real timings. Ready for
the real-video run; nothing rendered is proven yet.

### D-291.4 — RAW #126 (run 35931561397 on `636bab5e`): MP4 rendered (148.03 s)

Observed from the ladder tail: acne head + "resorcina." kept (chain fix on
real media), stomach gastritis only, aside dropped, CTA clip starts at
358.17 without "Por eso cuídate," (trim on real media), W intact -- but the
pimples family formed as {A1, M} with M winning and L separate (A1 lost, M
kept), and R+T were kept (no family with W): Level-1 selection 31.66 s vs
21.96 s in #125. Provider grouping variance, not D-291.2. Artifacts (JSON,
MP4, validator reports) unreachable from the container; QA/acceptance
failed for reasons not yet read. No Watch+Listen yet. No relaunch.

### D-291.5 — kept complementary deliveries compete for the composite Best Take

Mechanism shown with #125's JSON and #126's log: in #125 the monolithic
skin retry M was deleted, restored as a unique tail and only THEN judged
against A1 + L by `hybrid_composite_best_take` (M out); in #126 M was never
deleted, so no authority compared it with the two complete complementary
deliveries and the {A1, M} family label decided (A1 out, M kept 12 s). Fix
in the same authority: a kept complete complementary delivery is a
composite candidate on the complementary guard's own association
criterion; the composite criteria are unchanged. 10 refuting tests; D-291
harness uses a genuine two-take family. Offline only; not proven on media.

### D-291.6 — live post-render repair: in-window clamp + word floor

Executed path audited (`apply_post_freeze_boundary_pass` -> `render_with_
post_render_qc` -> Watch+Listen v1 advisory; `perceptual_repair_cycle`
still has zero importers). Defect: the live technical repair trimmed a
straddling silence by its whole duration from one edge (up to 0.6 s of
speech) and consulted no word boundary on either edge. Fix in
`live_boundary_repair` + `live_render_qc`: trim only the defect measured
inside the segment's window; never enter a word (frozen draft words);
zero-extent join findings need evidence of room or are refused
(`PHYSICAL_FAIL_UNREPAIRABLE`, recorded). 13 refuting tests; D-097.4
behaviour without word evidence unchanged. Pending: 4 `NOT_IMPLEMENTED`
Watch+Listen capabilities, the disconnected perceptual repair cycle, no
mid-segment repair, no gesture/pose model for pause-vs-restart.

### D-291.7 — RAW retrieval block

Artifact host denied by the network policy, AWS keys invalid, logs capped
at 5,000 lines; `*.githubusercontent.com` reachable. A relay workflow to
release assets was refused by the session's permission layer and NOT
added. The ONE authorized RAW is unspent until the Product Owner allows the
host, attaches run 35931561397's artifacts, or approves the relay.

### D-291.8 – D-291.11 — RAW #126 read for real; approved supervised edit as reference; three engine corrections

RAW #126's JSON was attached; its MP4 and the original were read from the
project bucket (unsigned reads work through the proxy -- security
observation E for the Product Owner). Real #126: technical QC PASS,
Watch+Listen BLOCKED on six hand-motion "debris" findings that overlap
spoken words; thyroid T1 (abandoned attempt) chosen over T2 (`winner`
0.95) by the case-B gesture count; skin M over A1 by DeliveryScore
tie-break; "resorcina." and "No" rendered as silent clips because Whisper
placed them inside measured silence; symptoms take cut twice inside
speech at ASR word gaps. The approved edit (18 segments, evaluation only)
maps one-to-one onto engine takes. Fixes: D-291.5.1 (positive label +
later-piece order for kept composite candidates; gynecologist regression
found by the #126 replay), D-291.9 (ASR word spans reconciled against
measured silence in `audio_silence.py`, wired in `flow_b`), D-291.10
(gesture vs reset in case-B materiality, interior gap trim, Watch+Listen
edge debris). D-291.6 verified on the real path; the two fixed-SHA guard
edits justified by empty range diffs. Not proven on video yet: the ONE
authorized RAW runs on the pushed head; retrieval via S3 + attached JSON.

### D-291.12 — RAW #127 (run 35945070839 on `b76bcc9b`): automatic MP4 reaches human review

`DELIVERABLE_PENDING_HUMAN_WATCH_LISTEN` (technical QC PASS, Watch+Listen 0
FAIL / 8 UNCERTAIN, Freeze reached), 145.97 s. Proven on the MP4: thyroid
retake T2, acne sentence with its last word, "No quiero…" with its
negation, no gesture BLOCK. Still wrong: symptoms take cut inside speech
by `human_boundary_polish_v5` (micro gaps, no measured pause) and by a
`pause_plus_strong_reset` boundary at a mid-sentence hold; gynecologist X
over Z (CASE B counted a post-speech gesture inside the ASR-padded
delivery span); skin M over A1 (A1 `failed` 0.8 excluded by D-291.5.1);
closing aside repeated after W (never competed, open). Corrected offline
(D-291.5.2, D-291.12a-c) with tests; NO further RAW authorized -- the next
run is the Product Owner's call. Human Watch+Listen of run 35945070839's
MP4 still required (F).


## 2026-09-24 isolated ASR comparison update

The worker project now permits `gpt-transcribe`; the existing worker-key
synthetic probe returned HTTP 200 (run 36022529122, job 107763146014).
The authorized Video00 ASR-only `gpt-transcribe` versus deterministic
Faster-Whisper `medium` comparison completed successfully in run 36040003462
on `feat/asr-provider-evaluation-and-replay` (test SHA bb80b0d7). GPT returned
624 whitespace-delimited words in 12.578 s; Medium returned 643 such words
(650 normalized), 45 segments, in 28.301 s. Medium has a triple repeated acne
sentence including 15 words in 0.44 s; GPT has one complete version. Critical
lexical ambiguities remain unverified by direct listening. Re-reading prior
run 36020263828 confirms the same Medium transcript/hash, correcting the
earlier 623-word attribution to that run. The workflow is manual-only again
at d47221f9; no further paid run was launched.
See `CUTSELL_ASR_PROVIDER_EVALUATION.md` for qualification and live evidence.
Canonical/production editing remains unchanged.


Deepgram follow-up (2026-09-24 18:32 UTC): requested Video00 comparison
stopped at credential preflight in run 36041858862 because the live
`EditDNA-Worker-2` template still lacks `DEEPGRAM_API_KEY`. No GPU or Deepgram
request occurred. The comparison awaits that credential; manual-only workflow
restored at 6b97dfc2. See the ASR evaluation document for exact evidence.


## 2026-09-24 authorized GPT + WhisperX full-engine experiment

The Product Owner now explicitly authorized one complete Video00 run with GPT
transcription and WhisperX alignment (supersedes the older pending-authorization
statement only for this experiment). Branch `feat/gpt-whisperx-video00`, primary
ASR opt-in only; production/default Medium and all editorial authorities stay
unchanged. Implementation passed 135 offline tests. Run `36045181502`, exact
head `23845fa143b27e80cf972c85ed1812b228960ceb`, completed; manual-only
trigger restored at `212a5a4b8d74fe2f599ccb863efe77d8e60624de`. Result: 643 aligned words, zero zero-duration words, 138.2-second MP4, technical QC PASS; editorial acceptance failed 3/11 checks (stomach false start, full gynecologist take, repeated percentage). Human Watch+Listen remains required. No production switch. A post-run per-source cache correction prevents the observed duplicate full-source GPT call; offline verified, no second paid run.
See `CUTSELL_GPT_WHISPERX_FULL_ENGINE_EVALUATION.md`.


## 2026-09-24 uploaded 55-second source comparison (authorized)

The user supplied `v12044gd0000d46k2m7og65re0trr1rg.MP4` after requesting
a full-engine test and accepting a Medium versus GPT + WhisperX comparison.
Exactly two new sequential L4 calls are authorized, one per provider on the
same source and checkout. Source SHA-256: `c9d892629f19d49058246e79bb57eb0af10bdf123416122307aa0bcd5c91cb14`,
12,429,383 bytes, 55.401 seconds. One frozen template snapshot; no engine
changes, no Video00-specific acceptance oracle, no retries or production
promotion. Source upload uses a short-lived one-object PUT capability encrypted
to an ephemeral public key; cloud credentials remain in the existing runner.
Source bytes verified before both GPU calls. Completed run `36062782881` at
`89bb9c9bceb40264d5911677de790088b204232a`; both Modal apps stopped.
Medium: 168 words, ten zero-duration words, four selected fragments, 16-second
MP4; technical QC PASS but perceptual repeated-content FAIL blocks delivery.
GPT + WhisperX: 157 words, zero zero-duration words, ASR passed (two requests);
one selected 3.66-second fragment, coherence/content-loss review blocked
freeze and render. No GPT MP4 exists. Neither result is approved.
Manual-only workflow restored at `0be700b`; no further paid calls or promotion.
See `CUTSELL_UPLOADED_ASR_COMPARISON.md`.


## 2026-09-24 uploaded GPT repeat completed

User authorized one additional GPT + WhisperX full-engine run. Run
`36064364263`, test SHA `349b00cd3cc430521f7e8a06cd2c024e91719f93`,
completed and Modal app stopped. Same source and engine package SHA as previous
comparison. Two new GPT requests; 157 words, zero zero-duration words; same
50.70015–54.36 selected fragment, 15 discards. Coherence/content loss again
blocked freeze and render. No new MP4. Elapsed 75.722 s. No further paid calls
authorized. Manual-only restored at `2d7ab16`. Production unchanged.


## 2026-09-24 new 97-second MOV GPT test

User authorized one GPT + WhisperX full-engine test on the newly uploaded MOV.
Run `36065470982`, test SHA `c580fad32e8de421450aea90ae48930f617e88d5`,
completed and Modal stopped. Source SHA256
`5ae3cffb9034cc9aebbe4f645c4f1f34538f9113f05990291215c86b7163b681`,
68,730,266 bytes, 97.433 s. Same engine package as previous trials.
136 words, zero zero-duration words, three GPT requests; four selected clips,
six discards, 17.567 s MP4, technical PASS, human review required.
Selected transcript retains a setup remark, Yeah, and repeated CTA. GPT returned
empty text for 0–60 s despite nonzero source audio (mean -24.9 dB, max -5.6 dB);
speech/music not adjudicated. Do not claim full transcript coverage or editorial
approval. Report artifact 10836610270, video part 10836405775; hashes verified.
Full local decode passed. No further paid calls. Manual-only restored at 9cede8a.


## 2026-09-24 Deepgram newly authorized source test

DEEPGRAM_API_KEY now present in canonical RunPod template and valid.
CPU-only run 36066724857 / test ccd3014e75ec1cca201158a9d00fe7d70b41e527
completed one Nova-3 multilingual request on the same 97-second MOV.
268 words; 130 before 60 s, first word 0.08 s, last 96.99 s. Prior GPT had
136 words and zero before 60 s. Deepgram timestamps retained in raw response.
This is coverage evidence, not human-reference accuracy or editorial approval.
Deepgram uses full mono16k MP3; prior GPT used mono16k WAV chunks, so transport
and provider effects are not isolated. No editor invocation, GPU or production
change. Manual-only restored at 283b11a. Artifact 10836188777 verified SHA256
8987fa7cb8b8cfcc44e5d81d4b45b5fbeefb39d99ddcb23796333d9bf7b79352.
No further paid request authorized.


## Authorized Deepgram full-engine experiment in progress

Product Owner requested the edited video from Deepgram. Added opt-in RAW
Deepgram provider, native word timing validation, per-job successful source
cache, no fallback or invented timestamps. Defaults and editorial authorities
unchanged. One full-engine L4 run on the 97-second MOV authorized. Nova-3 multi
with punctuation for sentence grouping, no WhisperX. Native positive overlapping
word ranges are retained and counted, not silently altered (observed in prior
real payload); segment bounds enclose all words. Prior real 268-word response
preserves every word. Results pending.


Deepgram full-engine completed: run 36067890052, job 107861800043, exact test
01a7acde8243c8410bb6f3d6823515c99f084c14. One new provider request, cache hit
once, 268 words, one native overlapping pair, 13 selected fragments/15 discarded.
39.5 s MP4, technical PASS, HUMAN_REVIEW_REQUIRED; elapsed 140.9 s.
Selected transcript retains setup/blooper remarks (How am I supposed to say,
Turn it around, I'm already gonna mess it up), so not editorially approved.
No human listening. Modal completed/stopped. Reports artifact 10836867865; video
parts 10836573202 and 10836653262, digests verified. 18 targeted tests passed.
Manual-only restored at 9214068; no further paid run authorized or production switch.


## Watch & Listen forensic after Deepgram output

Read-only audit traced retained bts 0.80/0.75 to kept_fail_open and singleton
resolution safety floors. Offline actual _semantic_best_take checks reproduce
keep at bts 0.80 even with corroboration, keep at 0.90 without corroboration,
and discard at 0.90 with corroboration. Active whole-video context is an ASR
summary plus separately computed local signals; hybrid Gemini cleanup is text
only. clean_cut_provider=None; v2/guard diagnostics disabled. P2 whole-video
reasoning flag is diagnostics-only and cannot repair selection by enabling it.
No engine fix, threshold change, production change or paid run. See
CUTSELL_WATCH_LISTEN_DEEPGRAM_FORENSIC.md for evidence and correction boundary.


## 2026-09-24 authorized Video00 Deepgram full run completed

Run 36069441576 / job 107866693967, test 9aa5e5180a534e27833eb8d5ee717368b2203d3c.
Original Video00 source hash verified. Deepgram: 651 words, zero zero-duration
words, seven native overlapping pairs; one request and one intra-job cache hit.
20 selected/20 discarded, output 143.334 s, engine elapsed 431.011 s.
Editorial criteria 10/11: full gynecologist and percentage-restatement checks
pass, abandoned stomach attempt remains. Historical gold 17/18 (exact pimples
micro-2 mismatch). Technical QC NEEDS_HUMAN_REVIEW for 1.5034 s accidental
silence at output 42.9635–44.4669; diagnostic MP4 only. No human listening or
universal ASR superiority claim. Modal stopped. Artifact reports 10837364760;
parts 10837424599/10837324883; SHA digests verified. Manual-only restored at
1bfe30f8. No Watch & Listen fix or production change; no further paid call.
