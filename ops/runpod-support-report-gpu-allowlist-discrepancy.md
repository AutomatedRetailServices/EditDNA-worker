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

## What we need from RunPod

1. Confirm whether the Blackwell worker was genuinely dispatched against
   endpoint `xxu7autt8mv2rn`.
2. If so, an explanation of why `gpuTypeIds` did not exclude it.
3. Any guidance on ensuring `gpuTypeIds` is enforced as a hard filter for
   this endpoint going forward.
4. Given a brand-new endpoint with the same allowlist also failed to
   provision a worker within 5 minutes (evidence above), please check
   current capacity/availability for `RTX 4090` / `L4` / `A40` / `RTX A6000`
   in whatever region(s) this account can reach, and whether anything
   account-side (quota, spend-rate limit, placement policy) is preventing
   worker assignment. (Account balance and spend-rate limit were manually
   verified as healthy on our side prior to this test: balance $11.28,
   spend rate $0/hr, spend-rate limit $80/hr — billing is not the leading
   explanation.)
