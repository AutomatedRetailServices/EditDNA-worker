# CutSell worker on Modal (CPU, pay per use)

Status: code written and exercised locally. **Not deployed.** Nothing in Modal or Render was changed.

## Shape

```
iPhone app ──> cutsell-api (Render, unchanged) ──> Redis queue "cutsell"
                         │                              ▲
                         └── POST wake ──> Modal: wake ─┴─> drain_queue (one job per container)
                                           Modal: sweep (every 5 min, safety net)
```

- The API still enqueues RQ jobs exactly as before; job status, progress and results are read from
  Redis as before. The mobile app needs no change.
- `drain_queue` runs the same job functions the always-on RQ worker ran. It takes one job and exits,
  so idle time costs nothing and several videos are processed in parallel (up to 8).
- With `CUTSELL_ENGINE=simple` the worker needs no GPU. Image: Debian slim + ffmpeg +
  `requirements.cutsell.worker.cpu.txt`.

## Files

| File | Role |
|---|---|
| `modal_cutsell_worker.py` | The Modal app: `wake`, `drain_queue`, `sweep` |
| `requirements.cutsell.worker.cpu.txt` | CPU-only dependencies |
| `cutsell_worker/worker_wake.py` | Best-effort wake call made by the API after each enqueue |
| `cutsell_worker/queueing.py` | Calls `wake_worker()` after `enqueue_flow_b`, `enqueue_export`, `enqueue_batch_item` |

## Configuration (names only; values live in Modal / Render secrets)

Modal secret `cutsell-worker`:

| Name | Same value as |
|---|---|
| `REDIS_URL` | the Render API service |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `S3_BUCKET` | the Render API service |
| `DATABASE_URL`, `SENTRY_DSN` | the Render API service, if it has them |
| `ANTHROPIC_API_KEY`, `DEEPGRAM_API_KEY` | new |
| `CUTSELL_WORKER_WAKE_TOKEN` | new: any long random string |

The image itself sets `CUTSELL_ENGINE=simple` and `CUTSELL_EXPORT_DELIVERY_GATE=technical_qc`.

Render service `cutsell-api`, two new variables:

| Name | Value |
|---|---|
| `CUTSELL_WORKER_WAKE_URL` | the `wake` URL that `modal deploy` prints |
| `CUTSELL_WORKER_WAKE_TOKEN` | the same string as in the Modal secret |

Without `CUTSELL_WORKER_WAKE_URL` the API behaves exactly as before (no wake call).

## Deploy

```
modal secret create cutsell-worker ...      # once, with the values above
modal deploy modal_cutsell_worker.py        # prints the wake URL
```
Then set the two variables on `cutsell-api` and redeploy it from a branch that contains this code.

## What was verified, and what was not

Verified locally (real API + RQ + Redis, S3 stand-in, real Deepgram and Claude calls):
- upload -> cut with the simple engine -> draft -> export -> delivered file, on two sales videos;
- the same flow with a worker that has only the CPU requirements installed;
- `drain_queue` run locally against a Redis queue: takes one job, finishes it, exits.

Not verified:
- an actual `modal deploy`, the image build, the `wake` endpoint over the network, or `sweep` on schedule;
- the real Redis and S3 of the Render deployment;
- cold-start time and real cost per video on Modal.

## Known limits

- A job retried through RQ's own requeue does not call `wake`; `sweep` picks it up within 5 minutes.
- `drain_queue` allows `JOB_TIMEOUT_SEC + 600` seconds; a job that needs more is cut off.
- Export is still 1080x1920 at 30 fps and rendered on the server.
