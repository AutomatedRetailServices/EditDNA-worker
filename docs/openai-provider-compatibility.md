# OpenAI provider compatibility decision

The installed constraint remains `openai>=1.7.1`. An attempt to access current official
documentation returned an authorization error, so this refactor deliberately does not
claim model/API compatibility that could not be verified and does not change the constraint.

All five repository operations—semantic classification, visual bad-take detection,
boundary refinement, Take Judge, and standalone multimodal clause scoring—retain Chat
Completions. Existing `image_url` data-URL inputs are preserved. Multimodal clause scoring
retains its existing JSON-object response format; the other operations retain legacy JSON
parsing plus strict internal validation. No operation is migrated to Responses API or SDK
strict structured outputs. The adapter boundary permits a separately tested migration later.

The shared client uses a 60-second timeout and zero SDK retries by default. Zero retries
preserves the previous effective single-call behavior and therefore call frequency. Both
values are validated and can be overridden with `OPENAI_TIMEOUT_SECONDS` and
`OPENAI_MAX_RETRIES`.
