# CutSell staging readiness

Snapshot for the clean `cutsell/mobile-v1-clean` release branch. This is evidence only; it does not authorize paid deployment or merge.

## Clean release branch

- Draft PR: #25
- Base: `main` (`2fb13e5aa228e8e525b942a9b49182032b797e61`)
- Clean branch does not inherit the PR #23 / PR #24 recovery history.
- PR #24 remains untouched as a historical/reference backup.

## Passed gates

- Clean Worker CI: 167 CutSell tests passed.
- API container: `Dockerfile.cutsell` builds successfully.
- API container boot: `/v1/healthz` returns `ok=true` and `service=cutsell-api`.
- iOS Xcode Simulator build passed on the clean branch.
- Golden S3 real-video benchmarks passed structural and known-reject gates on the source implementation.
- Real RunPod GPU smoke passed on NVIDIA A40 using `Bloppers15.mp4`; temporary Pod was deleted successfully after the run.
- Slim CutSell worker image builds and pushes successfully without reinstalling the base PyTorch/CUDA stack.
- Validated worker image: `ghcr.io/automatedretailservices/cutsell-worker:f1bc3adc4f2f8df0ae005fa551116dc4d08b30e6`.
- Validated worker image digest: `sha256:41f20a15a70a301ee82990a6b18f6a03ff340b5838cafbc6a493e34855a8e09a`.
- Anonymous GHCR pull passed from a fresh runner, and the container successfully imported CutSell/Faster-Whisper while retaining the base Torch 2.6 runtime.
- Paid GPU smoke, worker-image build and GHCR-pull one-shot trigger markers were removed after validation; the reusable workflows are manual-only.

## Runtime defaults frozen for staging

- `CUTSELL_AUTH_REQUIRED=1`
- `CUTSELL_CLEAN_CUT_JUDGE=0`
- `CUTSELL_ASR_MODEL=medium`
- Semantic/Visual/Take Judge: `gpt-4o-mini`
- Render runs `cutsell-api` only.
- RunPod runs the GPU RQ worker.
- API and worker must share the same Redis and S3 configuration.

## Still required before beta

1. Explicit approval for paid staging infrastructure.
2. Deploy the Render staging API with secrets and auth enabled.
3. Deploy one RunPod staging GPU worker from an immutable CutSell image.
4. Verify API health with real Redis/S3/OpenAI readiness.
5. Run staging E2E: upload -> Flow B -> Draft recovery -> edit -> export.
6. Verify cancel/retry and recovery across process/app interruption.
7. Perform rollback test for API git SHA and worker image SHA.
8. Only after staging is stable: Apple Developer signing / TestFlight and physical-iPhone beta.

Passing this checklist is not equivalent to declaring the editing brain perfect. Real-video golden QA and later human edit-quality review remain release gates.
