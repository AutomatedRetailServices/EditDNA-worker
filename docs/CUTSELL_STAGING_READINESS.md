# CutSell staging readiness

Snapshot for the clean `cutsell/mobile-v1-clean` release branch. This is evidence only; it does not authorize production deployment or merge.

## Clean release branch

- Draft PR: #25
- Base: `main` (`2fb13e5aa228e8e525b942a9b49182032b797e61`)
- Clean branch does not inherit the PR #23 / PR #24 recovery history.
- PR #24 remains untouched as a historical/reference backup.

## Passed gates

- Clean Worker CI: 167+ CutSell tests passed; CI also builds and boots the staging API container.
- API container: `Dockerfile.cutsell` builds successfully.
- API container boot: `/v1/healthz` returns `ok=true` and `service=cutsell-api`.
- iOS Xcode Simulator build passed on the clean branch.
- Golden S3 real-video benchmarks passed structural and known-reject gates on the source implementation.
- Real RunPod GPU smoke passed using `Bloppers15.mp4`; temporary Pod was deleted successfully after the run.
- Slim CutSell worker image builds and pushes successfully without reinstalling the base PyTorch/CUDA stack.
- Validated worker image: `ghcr.io/automatedretailservices/cutsell-worker:f1bc3adc4f2f8df0ae005fa551116dc4d08b30e6`.
- Validated worker image digest: `sha256:41f20a15a70a301ee82990a6b18f6a03ff340b5838cafbc6a493e34855a8e09a`.
- Anonymous GHCR pull passed from a fresh runner, and the container successfully imported CutSell/Faster-Whisper while retaining the base Torch 2.6 runtime.
- Render staging API is deployed separately from legacy `editdna-web` at `https://cutsell-api.onrender.com`.
- Render staging health is fully ready: `queue_ready=true`, `storage_ready=true`, `semantic_ready=true`.
- Real staging E2E passed through the public API and temporary RunPod GPU: beta auth -> presigned upload -> Flow B -> Draft recovery -> audio edit -> autosave -> export -> download -> ffprobe.
- Successful E2E evidence: project `staging-e2e-77fc9106c0e7`, Flow B job `27e17b28-1cbc-4f63-a5cf-4b7af767c834`, export job `916c642e-aea5-41bd-81c6-038ee0fc68c9`, Draft revision 2, 4 selected clips, export size 2,533,472 bytes, export duration 5.722333 sec, render version `rv_b0a2dca4f1384eb1b6eda5559f883d3f`.
- The E2E temporary RunPod Pod was deleted successfully with HTTP 204 after verification.
- Persisted Draft recovery passed after the GPU process was deleted.
- Public API cancel -> retry -> cancel behavior passed without a GPU worker present.
- Render rollback gate passed: the API rolled back to the previous successful deploy, passed full health, restored the current deploy, and passed full health again.
- RunPod worker rollback gate passed: both the previous worker image and the current worker image booted, registered on the `cutsell` RQ queue, and were deleted after verification.
- RunPod rollback cleanup finished with zero rollback Pods remaining; the rollback preflight also confirmed there was no persistent `cutsell-staging-worker` before the test.
- Consumed one-shot staging, diagnostic, E2E, recovery, Render and worker rollback trigger workflows/markers were removed from the release checkpoint after successful verification.

## Runtime defaults frozen for staging

- `CUTSELL_AUTH_REQUIRED=1`
- `CUTSELL_CLEAN_CUT_JUDGE=0`
- `CUTSELL_ASR_MODEL=medium`
- Semantic/Visual/Take Judge: `gpt-4o-mini`
- Render runs `cutsell-api` only.
- RunPod runs the GPU RQ worker when processing is required.
- API and worker share the same Redis and S3 configuration.

## Staging secret inventory

- `REDIS_URL`: present in GitHub Actions secrets.
- `RUNPOD_API_KEY`: present in GitHub Actions secrets.
- `RENDER_API_KEY`: present and validated against the Render API.
- AWS/S3/OpenAI values are loaded from the existing `EditDNA-Worker-2` RunPod template vault and were successfully installed into the isolated Render staging API.

## Still required before beta

1. Re-run final Clean Worker CI and iOS CI on this cleaned release checkpoint.
2. Apple Developer signing / App ID / provisioning / TestFlight setup.
3. Physical iPhone beta checks: camera, Photos import, background multipart resume, app kill/resume, Flow B, playback/timeline, Swap Take, trim/split, overlays, captions, recovery, export, Save to Photos and share.

PR #25 remains Draft. Production deployment, final merge/release, repo deletion/archive and Apple/TestFlight actions still require explicit approval at their respective gates.

Passing this checklist is not equivalent to declaring the editing brain perfect. Real-video golden QA and later human edit-quality review remain release gates.
