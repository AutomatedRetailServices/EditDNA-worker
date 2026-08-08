# CutSell staging readiness

Snapshot for the clean `cutsell/mobile-v1-clean` release branch. This is evidence only; it does not authorize paid deployment or merge.

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
- Real RunPod GPU smoke passed on NVIDIA A40 using `Bloppers15.mp4`; temporary Pod was deleted successfully after the run.
- Slim CutSell worker image builds and pushes successfully without reinstalling the base PyTorch/CUDA stack.
- Validated worker image: `ghcr.io/automatedretailservices/cutsell-worker:f1bc3adc4f2f8df0ae005fa551116dc4d08b30e6`.
- Validated worker image digest: `sha256:41f20a15a70a301ee82990a6b18f6a03ff340b5838cafbc6a493e34855a8e09a`.
- Anonymous GHCR pull passed from a fresh runner, and the container successfully imported CutSell/Faster-Whisper while retaining the base Torch 2.6 runtime.
- Paid GPU smoke, worker-image build, GHCR-pull and infra-preflight one-shot trigger markers were removed after validation; reusable workflows are manual-only.
- Paid staging RunPod create/delete workflows are prepared but not executed. Repository tests require manual dispatch plus explicit approval phrases and block accidental duplicate/deletion targets.

## Runtime defaults frozen for staging

- `CUTSELL_AUTH_REQUIRED=1`
- `CUTSELL_CLEAN_CUT_JUDGE=0`
- `CUTSELL_ASR_MODEL=medium`
- Semantic/Visual/Take Judge: `gpt-4o-mini`
- Render runs `cutsell-api` only.
- RunPod runs the GPU RQ worker.
- API and worker must share the same Redis and S3 configuration.

## Staging secret inventory

Presence-only preflight on the clean branch confirmed:

- `REDIS_URL`: present in GitHub Actions secrets.
- `RUNPOD_API_KEY`: present in GitHub Actions secrets.
- AWS access key, AWS secret, AWS region, S3 bucket and OpenAI key: not stored as direct GitHub Actions secrets.
- Those AWS/S3/OpenAI values are currently available through the existing `EditDNA-Worker-2` RunPod template vault and have already been used successfully by real-video QA and GPU smoke workflows.

The RunPod staging worker workflow can load AWS/S3/OpenAI from that template vault. The Render staging API still needs those same AWS/S3/OpenAI values configured in Render before real E2E testing.

## Still required before beta

1. Explicit approval for paid staging infrastructure.
2. Configure the Render staging API with Redis/AWS/S3/OpenAI values and auth enabled.
3. Deploy the Render staging API.
4. Deploy one RunPod staging GPU worker from an immutable CutSell image.
5. Verify API health with real Redis/S3/OpenAI readiness.
6. Run staging E2E: upload -> Flow B -> Draft recovery -> edit -> export.
7. Verify cancel/retry and recovery across process/app interruption.
8. Perform rollback test for API git SHA and worker image SHA.
9. Only after staging is stable: Apple Developer signing / TestFlight and physical-iPhone beta.

Passing this checklist is not equivalent to declaring the editing brain perfect. Real-video golden QA and later human edit-quality review remain release gates.
