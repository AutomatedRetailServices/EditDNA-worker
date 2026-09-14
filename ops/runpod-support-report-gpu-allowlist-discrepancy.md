# RunPod support report — GPU allowlist discrepancy

**Endpoint:** `xxu7autt8mv2rn`

**Configured `gpuTypeIds`** (set at endpoint creation on 2026-08-24 by our
`cutsell-serverless-gpu-gate.yml` workflow's `POST /v1/endpoints` call, and
confirmed unchanged today — the endpoint is at `version: 232`, i.e. roughly
232 subsequent updates, none of which have ever included `gpuTypeIds` in
their PATCH body):
- NVIDIA GeForce RTX 4090
- NVIDIA L4
- NVIDIA A40
- NVIDIA RTX A6000

**Observed worker (from our own dashboard log inspection):**
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- Compute capability: sm_120
- This GPU class is **not** in the endpoint's configured `gpuTypeIds` list
  above.

**Current worker image:** `madiator2011/better-pytorch:cuda12.4-torch2.6.0`
(the image's PyTorch/CUDA build does not include Blackwell/sm_120 compiled
kernels — that is expected and not itself the question; PyTorch's Blackwell
support requires >=2.7 with CUDA >=12.8).

## The question for RunPod support

Was this worker actually scheduled for endpoint `xxu7autt8mv2rn`? If so, why
was it placed on a GPU type (`NVIDIA RTX PRO 6000 Blackwell Server Edition`)
outside that endpoint's configured `gpuTypeIds` allowlist
(`RTX 4090` / `L4` / `A40` / `RTX A6000`)? We rely on `gpuTypeIds` as a hard
compatibility filter — our production image cannot use Blackwell hardware —
so a placement outside that list, if genuine, is a scheduling behavior we'd
like RunPod to confirm and, if possible, prevent going forward (e.g. whether
`gpuTypeIds` is meant to be a strict allowlist or only a soft preference
under some conditions).

## Supporting evidence we have on our side

- The endpoint's live configuration, re-confirmed via a direct, read-only
  `GET /v1/endpoints/xxu7autt8mv2rn` call today, still shows exactly the
  four-class `gpuTypeIds` list above — unchanged since creation.
- Two independent health-check jobs submitted to this same endpoint on
  2026-09-01 (job ids `a6f37082-d130-4961-ada0-09cf1051ed67-u2` and
  `3c0ca828-1138-43b8-8966-88a82065a0fe-u1`) each sat in `IN_QUEUE` for a
  full 5 minutes with no worker ever assigned at all — a different failure
  shape from the Blackwell case (worker assigned, then crashed on CUDA
  init), included here in case it's relevant context for the same
  underlying placement/capacity investigation.

## Fresh-endpoint isolation test (2026-09-01, new)

To rule out the existing endpoint being stale/unhealthy, we created a
brand-new, temporary, zero-history endpoint using the exact same compatible
GPU allowlist and current production image, submitted exactly one health
job, and tore the endpoint down immediately after:

- Temp endpoint: `fuwdxy36rvtt2e` (deleted after the test — HTTP 204)
- Temp template: `koechktwze` (deleted after the test — HTTP 204)
- `gpuTypeIds`: `NVIDIA GeForce RTX 4090`, `NVIDIA L4`, `NVIDIA A40`,
  `NVIDIA RTX A6000` (identical to the production endpoint's own allowlist)
- Endpoint created 06:29:33Z, reported ready 06:29:34Z (essentially instant)
- Health job `d3c789f1-1a4a-4f99-a23e-e89169f7b456-u2` submitted 06:29:34Z
- Status stayed `IN_QUEUE` for the full 300.6s test window (~65 consecutive
  polls at ~5.1s intervals) — **no worker was ever assigned** (`workerId`
  null on every poll)
- We cancelled the job and deleted both temp resources at 06:34:35–37Z

**This brand-new endpoint — with no history of its own — hit the identical
failure as `xxu7autt8mv2rn`: a job accepted into queue that no worker ever
picked up, across a 4-class allowlist that should have ample capacity.**
This is evidence against the existing endpoint being uniquely stale/
unhealthy, and points toward an account-level or provider-side capacity/
placement issue affecting these GPU classes for this account right now,
independent of which specific endpoint is used.

## workersMax verification (2026-09-01, new — rules out our own orchestration)

Before escalating further we verified, from our own API responses (not
assumption), that `workersMax` was genuinely `1` at the moment each health
job was submitted, on all three occasions, via a direct read-after-write
`GET /v1/endpoints/{id}` immediately following each PATCH/create:

- **Production, attempt 1**: PATCH → GET confirms `workersMax: 1` (`01:19:29.08Z`)
  → health job `a6f37082-d130-4961-ada0-09cf1051ed67-u2` submitted
  (`01:19:29.47Z`) → `IN_QUEUE` for **~304.5s**, `worker_id` null on every
  poll → job cancelled, no worker ever assigned.
- **Production, attempt 2**: re-PATCH (retry) → GET confirms `workersMax: 1`
  (immediate) → health job `3c0ca828-1138-43b8-8966-88a82065a0fe-u1`
  submitted → `IN_QUEUE` for **~303.8s**, `worker_id` null on every poll →
  job cancelled, no worker ever assigned.
- **Fresh temporary endpoint**: created with `workersMax: 1` in the
  `POST /v1/endpoints` payload → GET confirms `workersMax: 1` (`06:29:34.15Z`)
  → health job `d3c789f1-1a4a-4f99-a23e-e89169f7b456-u2` submitted
  (`06:29:34.46Z`) → `IN_QUEUE` for **~300.6s**, `worker_id` null on every
  poll → job cancelled, no worker ever assigned.
- **Teardown (`workersMax` → `0` / endpoint delete) happened only after each
  failure was already concluded** — never during the wait.
- We also checked for any other process that could have modified the
  endpoint mid-wait: no other automation in our repository ran during any
  of the three wait windows, and our only scheduled RunPod-adjacent job
  (an idle-pod reaper) has never once successfully executed a RunPod API
  call in any of its runs (unrelated pre-existing bug, tracked separately —
  not touched here) and in any case targets RunPod's Pods product, not
  Serverless endpoints.

**Additional account-side facts we've verified directly in the RunPod
dashboard**: billing is funded (balance $11.28, spend rate $0/hr, spend-rate
limit $80/hr, recent successful $20 reloads); the account currently shows
**0/15 workers deployed**; the selected GPU groups for this endpoint show
**"High Supply"** in the RunPod UI.

**In short: jobs are being accepted into queue against a correctly-configured,
API-confirmed `workersMax: 1` endpoint, with funded billing, unused worker
capacity (0/15 deployed), and GPU groups the UI itself reports as High
Supply — yet no worker is ever assigned. This is not explained by anything
under our control.**

## What we need from RunPod

Please inspect, specifically:

1. **Account-level Serverless worker provisioning** — why jobs are accepted
   into `IN_QUEUE` but no worker is ever assigned, despite `workersMax: 1`
   being API-confirmed at submission time and the account showing 0/15
   workers deployed.
2. **Scheduler placement** for this account/endpoint against the "High
   Supply" GPU groups shown in the UI.
3. **Quota/concurrency enforcement** — whether any account-level limit is
   silently preventing worker assignment even though billing is funded and
   the spend-rate limit ($80/hr) is nowhere near being hit ($0/hr current).
4. **GPU capacity allocation** for `RTX 4090` / `L4` / `A40` / `RTX A6000`
   specifically for this account, in whatever region(s) it can reach.
5. **The separate Blackwell/allowlist discrepancy**: confirm whether the
   `NVIDIA RTX PRO 6000 Blackwell Server Edition` worker we observed via our
   own dashboard log was genuinely dispatched against endpoint
   `xxu7autt8mv2rn`, and if so why it was placed outside the endpoint's
   configured `gpuTypeIds` allowlist (`RTX 4090` / `L4` / `A40` /
   `RTX A6000`) — whether `gpuTypeIds` is meant to be a strict allowlist or
   only a soft preference under some conditions, and guidance on enforcing
   it as a hard filter going forward.

## SDK version check (2026-09-01, new — rules out a known SDK issue)

Support flagged `runpod` Python SDK versions `1.7.11`–`1.10.0` as having a
known job-pulling/tracking issue. Checked directly from two independent
build logs (production RAW build and an earlier diagnostic probe build,
both on 2026-09-01): the worker image installs `runpod==1.12.0` (pinned in
`Dockerfile.cutsell.serverless` as `runpod>=1.7,<2`, which resolves to the
newest available release at build time). **1.12.0 is above the flagged
range — not the cause here.** Our handler's only use of the SDK
(`runpod.serverless.start(...)`) is exactly the job-pulling loop the
warning refers to, so this check is directly relevant, not incidental.

## Warm-worker (workersMin=1) isolation test (2026-09-01, new — decisive)

To test whether cold/on-demand provisioning specifically is the
bottleneck, we created a brand-new, temporary endpoint with
**`workersMin=1`** (forcing a standing worker independent of any job
arrival) instead of the usual `workersMin=0`, keeping everything else
identical (same compatible-only `gpuTypeIds` allowlist, same current
production image, FlashBoot enabled, no region restriction):

- Temp endpoint: `42jpheqrr9ly7w`, temp template: `l1bkopw88u` (both
  deleted after the test — HTTP 204)
- `workersMin: 1` / `workersMax: 1` **confirmed via a direct read-after-write
  GET** immediately after creation, before any job was submitted
- Health job `f6c36789-6c9a-4fcd-9c97-3f34891f2cfc-u2` submitted `15:12:16.94Z`
- **Every one of ~61 status polls over the full 301.85s test window shows
  `IN_QUEUE`, `worker_id: null` — the job never transitioned to
  `IN_PROGRESS` even once**
- Job cancelled, both temp resources deleted (204) immediately after

**Even a forced warm/standing worker request never resulted in a worker
being assigned.** This directly rules out "cold/on-demand provisioning is
the bottleneck" as an explanation — `workersMin=1` should provision a
worker regardless of whether a job has arrived, and it did not, for over 5
minutes, on a completely fresh endpoint. Combined with the evidence above
(API-confirmed `workersMax`/`workersMin`, funded billing, 0/15 workers
deployed, "High Supply" GPU groups, and the fresh-endpoint result before
this one), we consider this conclusive on our end: **no configuration
change available to us explains the lack of worker assignment.**

## Warm-worker retest (2026-09-01, ~2.5 hours later — confirms persistent, not transient)

To rule out a momentary capacity blip, we repeated the exact same
`workersMin=1` test above, unchanged, on another brand-new temporary
endpoint roughly 2.5 hours later:

- Temp endpoint: `vfc9ga1y7olpsh`, temp template: `nnh70hipp6` (both
  deleted after the test — HTTP 204)
- `workersMin: 1` / `workersMax: 1` **confirmed via read-after-write GET**
  before the health job was submitted
- Health job `c9921268-503a-4b9a-a176-48c406f42cc2-u2` submitted `17:42:39Z`
- **Every status poll over the full 300.56s window again shows `IN_QUEUE`,
  `worker_id: null` — no transition to `IN_PROGRESS`, identical to the
  first warm-worker test**

**Two independent warm-worker tests, on two independent fresh endpoints,
hours apart, produced the identical result.** This rules out a momentary
capacity fluctuation — we are treating this as a **persistent** RunPod
Serverless worker-provisioning/account-placement issue, not a transient
one.

We are holding off on further probing (no additional endpoints, no
Video00 runs, no configuration changes to the production endpoint, no
further health retries) until Support responds or gives a specific
configuration change to test.
