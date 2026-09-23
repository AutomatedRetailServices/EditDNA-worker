# CutSell staging deployment contract

This document freezes the staging topology before any paid/production deployment.

## Topology

- **Render:** `cutsell-api` only. No CPU background worker.
- **RunPod:** one GPU RQ worker running the immutable CutSell worker image.
- **Redis:** shared queue/state backend used by API and GPU worker.
- **S3:** shared media, drafts/assets, previews, exports and smoke artifacts.
- **OpenAI:** Semantic, Visual and Take Judge providers. Experimental Clean Cut Judge remains disabled by default.

## Required production flags

- `CUTSELL_AUTH_REQUIRED=1`
- `CUTSELL_CLEAN_CUT_JUDGE=0`
- `CUTSELL_ASR_MODEL=medium`
- `CUTSELL_SEMANTIC_MODEL=gpt-4o-mini`
- `CUTSELL_VISUAL_MODEL=gpt-4o-mini`
- `CUTSELL_TAKE_JUDGE_MODEL=gpt-4o-mini`

## Worker image

Build from `Dockerfile.cutsell.worker` and deploy by immutable SHA tag:

`ghcr.io/automatedretailservices/cutsell-worker:<git-sha>`

Never deploy staging/production from a floating source checkout when an immutable image is available.

## Deployment gates

1. CutSell Clean Worker CI green.
2. CutSell iOS CI green.
3. Golden real-video benchmark green.
4. Real RunPod GPU smoke green.
5. Immutable worker image successfully built and pushed.
6. Render API health reports queue/storage/provider readiness.
7. Staging E2E: upload -> Flow B -> Draft recovery -> editor changes -> export.
8. Rollback test: previous API commit and previous worker image SHA can be restored.

## Spend / irreversible changes

Creating or replacing a paid RunPod worker, creating paid Render services, production deployment, merging the release branch, or deleting/archive-changing legacy infrastructure requires explicit user approval.

## Rollback

- API: redeploy the previous known-good git SHA.
- Worker: terminate the new RunPod worker and recreate from the previous known-good `cutsell-worker:<sha>` image.
- Do not delete prior known-good image tags during beta.
