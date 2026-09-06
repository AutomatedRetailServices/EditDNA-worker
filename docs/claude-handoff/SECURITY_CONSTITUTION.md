# CutSell.ai — Security Constitution for Claude Code

Security is a continuous engineering gate, not a pre-launch cleanup task.

## Core rule
Every material code change must preserve confidentiality, integrity, authorization boundaries, secret hygiene, dependency safety, and least privilege.

## Protected surfaces
Claude must explicitly consider security impact when touching:
- authentication/session/JWT/Apple auth;
- user/project/source/draft ownership;
- API routes and authorization checks;
- multipart and media upload paths;
- FFmpeg/media parsing;
- S3 object keys, signed URLs and bucket access;
- Redis/RQ/job payloads;
- RunPod/GPU worker inputs and outputs;
- database/storage access;
- notifications/webhooks;
- Docker images and GitHub Actions;
- environment variables and secrets;
- third-party SDKs/providers;
- account/project deletion and data retention.

## Never allow
- secrets committed to Git;
- secrets printed in logs;
- client-controlled `user_id`/`project_id` trusted without server authorization;
- cross-user project/media access;
- public S3 objects by default;
- path traversal in uploads/artifacts;
- shell/command injection through filenames or user input;
- unrestricted file types/sizes without validation;
- unsafe direct SQL construction;
- permissive CORS in production without explicit need;
- silent auth bypass/fallback;
- force-push/main bypass as an automation convenience;
- production credentials inside prompts/docs;
- dependency upgrades without reviewing security/regression impact;
- disabling a security check merely to make CI green.

## Security review loop
For material changes:
1. identify the trust boundary touched;
2. identify attacker-controlled inputs;
3. verify authorization/ownership;
4. verify secret handling;
5. verify validation/escaping;
6. add/update security regression tests where applicable;
7. run static/dependency/container checks when available;
8. report remaining risks.

## Required security baseline
Progressively install and maintain, without derailing the active Clean Cut blocker:
- GitHub secret scanning / push protection where the account plan supports it;
- CodeQL or equivalent SAST;
- dependency scanning / Dependabot;
- Python dependency audit such as `pip-audit` or OSV equivalent;
- container image scanning such as Trivy or equivalent;
- branch protection / required CI before merge;
- least-privilege GitHub Actions permissions;
- immutable container images for GPU deployment;
- security regression tests for auth/ownership and media access.

Security-tooling changes should be scoped/reviewed and must not silently create recurring paid infrastructure.

## Claude permissions doctrine
Prefer read/test/edit permissions. Treat destructive/irreversible commands as approval-gated.
Never use blanket unrestricted permission bypass for routine work.

Require explicit approval for:
- production deploy;
- deleting cloud resources outside temporary benchmark teardown;
- rotating/moving secrets;
- IAM changes;
- bucket public-access-policy changes;
- destructive database operations;
- force pushes;
- merging PR #25;
- writing to `main`;
- Apple/TestFlight release actions.

## Secrets doctrine
- secrets belong in secret stores/environment configuration, never source files;
- redact secrets from logs/errors;
- never paste real credentials into handoff/docs/tests;
- provider key presence may be validated, but key values must never be returned in benchmark artifacts;
- treat a previously exposed secret as compromised and rotate it.

## Authorization doctrine
Every user-owned resource access must be authorized server-side. Resource IDs are not authorization.

Verify ownership for:
- Project;
- SourceAsset;
- DraftTimeline;
- RenderVersion;
- upload/multipart state;
- overlay/media objects;
- notifications and downloadable artifacts.

## Media/upload doctrine
Media processing is an attack surface.
- validate extension/MIME/size/duration independently of client claims;
- sanitize filenames/paths;
- never interpolate user-controlled paths into shell commands unsafely;
- constrain temporary storage;
- avoid public bucket exposure;
- use scoped/signed URLs;
- preserve per-user source identity;
- test malformed/hostile upload behavior.

## CI/CD doctrine
- use least privilege for workflow tokens;
- do not print cloud/provider secrets;
- pin critical actions/dependencies appropriately;
- preserve immutable image digests for controlled GPU execution;
- teardown temporary benchmark resources;
- security gates should fail closed where appropriate, while still preserving safe diagnostics.

## Security status vocabulary
Keep security state separate from editorial state:
- SECURITY REVIEWED
- SECURITY TESTS PASS
- SAST PASS
- DEPENDENCY SCAN PASS
- CONTAINER SCAN PASS
- RUNTIME AUTHORIZATION VERIFIED

A feature is not “secure” merely because no vulnerability was noticed.
