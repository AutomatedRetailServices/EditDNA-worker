# Speed diagnosis and changes — 28 Oct 2026 (PO request)

Nothing here is merged to `main`. Server + engine speed changes DEPLOYED 28 Oct with PO OK:
production = b5eb3e1 on `feat/simple-engine-v1` (Render cutsell-api + Modal cutsell-worker).
Rollback ready, NOT applied: branch `rollback/velocidad` (62718e9 = f097956).

## Measured (live server, synthetic videos, audio-first path like the iPhone)
Workflows on `perf/diagnostico`: `cutsell-perf-cut-timing.yml`, numbers in `perf-results/cut/`.

| Raw | Ready in the editor | Biggest step |
|---|---|---|
| 30 s / 60 s, warm | ≈ 23 s / ≈ 32 s | Claude 32 % / 47 % |
| 4–5 min, warm | ≈ 38 s | Claude 36 % |
| first video after idle (cold) | +15–20 s | wake + container start ≈ 14 s |

No server render is needed to see a cut: the engine returns the draft JSON ("recipe") and the iPhone
plays it from its own copy; export is on the iPhone (DeviceExport). Modal logs keep ~1 day and the
engine logs no per-call duration (a per-step timing log would need a deploy).

## Changes (three branches)
1. **App** `perf/velocidad-app` (TestFlight build 60 = `build/split-and-speed`, with Split):
   - 4K / heavy gallery videos start the cut from the voice; the lighter copy is made afterwards and
     uploaded to the place reserved for the video (existing server `restore_uri`; checked live:
     `cutsell-perf-restore-check.yml` PASS).
   - Ready seen sooner: poll 1.5 s (was 3 s), project + job requested together.
   - The finished draft is kept on the phone at "ready", so the editor opens at once.
2. **Server** `perf/velocidad-servidor` (DEPLOYED in b5eb3e1):
   - The cut's submit no longer waits for the worker wake call (as every other job already did).
   - Modal worker: memory snapshot with the engine loaded; wake/sweep on a small image. Still
     scales to zero (no min_containers / keep_warm). Tests: 244 passed.
3. **Engine** `perf/velocidad-motor` (DEPLOYED in b5eb3e1):
   - The editor review, the sale review and the word fix start together on the decided cut; a review
     that changes the cut is followed exactly as before. Prompts and keep/hide rules unchanged.
   - Independent QA: first FAIL (loose pieces skipped on the early path) → fixed in 6fe12cd →
     PASS: 486-case sweep identical to f097956 except extra (counted) calls when a review changes
     the cut. Gold/Mix guard baselines moved to 6fe12cd (asked-for change).

Estimated before deploy: 4–5 min raw ≈ 38 s → ≈ 25–30 s warm; first of the day ≈ 46 s → ≈ 30 s.

## Measured after the deploy (same workflow, ready = job progress 94)
| Cut | Before (f097956) | After run 3 | After run 4 (`cut/20261028T163523Z.json`) |
|---|---|---|---|
| 4 min, first (cold) | 40.8 s | 66.7 s | 41.5 s |
| 5 min | 34.0 s | 48.2 s | 28.1 s |
| 5 min, again | 31.2 s | 63.9 s | 30.5 s |

Run 3 = first cuts right after the deploy (one 34 s wait for the first container, likely the memory
snapshot being made) plus slow Claude answers; run 4 is the normal state. Result: warm 5 min raw
≈ 3–4 s faster (submit no longer waits for the wake call: ~2 s); cold start unchanged (~41 s);
parallel reviews saved little because the two sequential Claude calls (pass → decision) dominate.
Going lower would mean making the decision call think less — not recommended without a quality check.
PO decision: keep the deploy (28 Oct).
