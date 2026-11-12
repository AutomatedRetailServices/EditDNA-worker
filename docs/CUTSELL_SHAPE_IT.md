# CutSell Shop board -- "Shape it" (server)

Status (11 Nov 2026): CODE + MOCK TESTS on branch `feature/shop-board-server`. NOT deployed, no real provider call
made yet, output quality of the three models NOT evaluated. Off by default (`CUTSELL_SHAPE_ENABLED=0`).

Code: `cutsell_app/shop_routes.py` (HTTP), `cutsell_app/shop_shape.py` (validation, prompt, providers,
transparent finish, limits, cost), tests `tests/test_cutsell_shop_shape.py`, real comparison
`scripts/shape_it_compare.py` (paid, run by hand only).

## What it does

The phone removes the background of ONE garment (or a pair of shoes) and sends a PNG with alpha. The server asks
one image model for the SAME item with a body shape -- "ghost mannequin" for clothes (worn by an invisible body),
filled and standing for shoes -- and returns a transparent PNG on the same canvas size as the input.

## Endpoint contract

`POST /v1/shop/shape` -- bearer session required (AuthScopeMiddleware), real accounts only.

Request
```json
{"image_png_base64": "<PNG with alpha, base64; a data:image/png;base64, prefix is accepted>",
 "kind": "clothing" | "shoes"}
```

| Status | Body | When |
|---|---|---|
| 200 | `{"image_png_base64": "...", "provider": "seedream"\|"gpt-image"\|"nano-banana", "cost_usd": 0.018}` | done; PNG RGBA, same width x height as the input |
| 400 | `{"error": "bad_kind"}` / `"bad_image"` / `"image_needs_transparency"` / `"bad_request"` | kind not clothing/shoes; not a PNG / undecodable / empty; no alpha or background not removed; body not a JSON object |
| 401 | `{"detail": "bearer token required"}` (middleware) or `{"error": "sign_in_required"}` | signed out, or a guest / test-probe account (no Apple identity bound) |
| 413 | `{"error": "image_too_large", "max_bytes": 6000000, "max_side": 2048}` | decoded PNG over max bytes, or a side over max px. The middleware answers `{"detail": "request body too large"}` for a body over 8 MiB |
| 429 | `{"error": "shape_daily_limit", "limit": 20}` | 21st call of this account in this UTC day |
| 502 | `{"error": "shape_provider_error"\|"shape_no_image", "provider": "<name>"}` | provider failed (after one retry) / answered without an image. Upstream messages are never forwarded |
| 503 | `{"error": "shape_disabled"}` / `"shape_provider_not_configured"` / `"shape_meter_unavailable"` | switch off; unknown model or its key missing; Redis unavailable (fails closed) |

"Signed in" = the account has an Apple identity bound in Redis (`cutsell:v1:user-apple:<hash>`). A guest that
later signed in with Apple keeps its `usr_` id and counts as signed in; guest and probe sessions do not.

Order: auth -> signed-in -> switch -> provider configured (known model + its key set) -> input validation ->
daily counter -> provider.

## Limits

- 20 per account per UTC day: Redis `INCR cutsell:v1:shape-day:<sha256(user)[:20]>:<YYYYMMDD>` (expires after
  2 days), same hashed-id pattern as `usage_limits._month_key`. The 21st is refused and not counted. A call that
  produced no image gives its slot back. If Redis cannot be read the call is refused (503), like the meter.
- Max decoded image 6,000,000 bytes and 2048 px per side, checked before pixels are decoded. 6 MB is the most
  that fits through the middleware's 8 MiB JSON cap once base64-encoded; going higher needs that cap raised
  (security-relevant, PO decision).
- Free for everyone signed in during testing; no credits system.

## Environment variables

| Name | New? | Default | Meaning |
|---|---|---|---|
| `CUTSELL_SHAPE_ENABLED` | new | `0` (off) | `1` turns the endpoint on. Off by default so a deploy never starts spending by itself; flip it on Render without a code change |
| `CUTSELL_SHAPE_MODEL` | new | `nano-banana` | `seedream` \| `gpt-image` \| `nano-banana` (aliases: `seedream-5.0-flash`, `gpt-image-2.5`, `openai`, `nano-banana-2-lite`, `gemini`) |
| `CUTSELL_SHAPE_DAILY_LIMIT` | new | `20` | per account per UTC day |
| `CUTSELL_SHAPE_MAX_BYTES` | new | `6000000` | decoded PNG bytes |
| `CUTSELL_SHAPE_MAX_SIDE` | new | `2048` | px |
| `CUTSELL_SEEDREAM_API_KEY` | new | -- | BytePlus ModelArk API key (BytePlus calls it `ARK_API_KEY`) |
| `CUTSELL_SEEDREAM_MODEL_ID` | new | `seedream-5-0-flash-260915` | confirm in the ModelArk console (one listing shows `dola-seedream-5-0-flash-260915`) |
| `CUTSELL_SEEDREAM_BASE_URL` | new | `https://ark.ap-southeast.bytepluses.com/api/v3` | EU: `https://ark.eu-west.bytepluses.com/api/v3` |
| `CUTSELL_SEEDREAM_SIZE` | new | `2K` | `1K` / `1.5K` / `2K` (flat price) |
| `OPENAI_API_KEY` | EXISTING name (already in `render.cutsell.yaml`) | -- | OpenAI key |
| `CUTSELL_SHAPE_OPENAI_MODEL` | new | `gpt-image-2.5-flare` | or `gpt-image-2.5-sunburst` (same price, slower, more precise) |
| `CUTSELL_SHAPE_OPENAI_QUALITY` | new | `high` | `low`..`max`; cost scales with it |
| `GEMINI_API_KEY` | EXISTING name (used by `cutsell_worker/brain_runtime.py` in the worker) | -- | Gemini API key; must ALSO be set on the Render API service |
| `CUTSELL_SHAPE_GEMINI_MODEL` | new | `gemini-3.1-flash-lite-image` | Nano Banana 2 Lite |
| `REDIS_URL` | existing | -- | counter, sign-in check, cost ledger |

## Providers (plain HTTPS via `requests`, timeouts 10 s connect / 150 s read, one retry on network/408/429/5xx)

1. **Seedream 5.0 Flash** (ByteDance, BytePlus ModelArk) -- `POST {base}/images/generations`, `Authorization:
   Bearer`, JSON `{model, prompt, image: data-URL, size, output_format: png, response_format: b64_json,
   watermark: false}`; reads `data[0].b64_json`, `usage.generated_images`. No alpha output, so the garment is sent
   on a solid key colour and the key is removed server-side (below).
2. **GPT Image 2.5 Flare** (OpenAI) -- `POST https://api.openai.com/v1/images/edits` (JSON form), `{model, prompt,
   images: [{image_url: data-URL of the RGBA PNG}], background: transparent, output_format: png, quality, size
   (1024x1024 / 1024x1536 / 1536x1024 from the input's aspect), n: 1}`; reads `data[0].b64_json` and `usage`
   (exact token cost). Native transparency.
3. **Nano Banana 2 Lite** (Google) -- `POST https://generativelanguage.googleapis.com/v1/models/
   gemini-3.1-flash-lite-image:generateContent`, header `x-goog-api-key`, `contents.parts = [text, inline_data
   (PNG)]`, `generationConfig.responseModalities = [TEXT, IMAGE]`; reads the first `inlineData` part and
   `usageMetadata` (exact token cost). 1K output only, no alpha -> key colour removal.

Transparent finish (`finish_output`): if the model returned an opaque image, pixels close to the key colour AND
connected to the image border become transparent (soft edge ramp), so a white logo inside a dark sweater stays.
The key is white behind dark/mid items and near-black (#181818) behind light items (mean luminance > 170). The
result is fitted, centred, into the input's own canvas size. Known risk to check in the real comparison: items
whose edge colour is very close to the key, and soft shadows the model may still paint.

## Prompt (`build_prompt`)

Clothing: professional e-commerce ghost-mannequin edit -- garment keeps its natural 3D shape as if worn by an
invisible body of average build (filled chest/shoulders/waist/hips/legs as fits, natural drape, inside back
visible through the neckline/waist opening); NO person, mannequin, hanger, skin or body parts.
Shoes: standing upright on their soles as if filled by invisible feet, no collapsed areas, laces as they are,
same angle; NO feet, legs, socks, shoe trees, stuffing or props.
Both: identity rules (identical colours, fabric/texture, print/logo/text in the same position and scale,
stitching, buttons, zips, pockets, laces, soles, trims, labels; no redesign, recolour, added/removed detail, no
change of proportions); background (transparent for GPT Image; a flat uniform key colour edge to edge with no
floor/shadow/gradient for the others); same framing, centred, nothing cropped, soft even light that does not
change colours, no text/watermark/border. The full text is in `cutsell_app/shop_shape.py`.

## Costs (list prices checked 11 Nov 2026)

| Model | Price basis | Est. per Shape it | Sources |
|---|---|---|---|
| Seedream 5.0 Flash (`seedream`) | $0.018 per generated image, reference images free | **$0.018** | [APIYI launch note (says it matches BytePlus official)](https://docs.apiyi.com/en/news/seedream-5-0-flash-launch), [LLMReference BytePlus listing, refreshed 2026-10-17](https://www.llmreference.com/provider/byteplus/seedream-v5-flash). The official BytePlus pricing page (https://docs.byteplus.com/en/docs/ModelArk/1544106) could not be opened from here: NOT confirmed first-hand |
| GPT Image 2.5 Flare (`gpt-image`), quality high, 1024x1024 | text in $5/M, image in $8/M, image out $30/M tokens | **~$0.066** (out ~1,756 tok = $0.053 + input image/text ~$0.012). Actual cost comes from the response's `usage` | [OpenAI model page](https://developers.openai.com/api/docs/models/gpt-image-2.5-flare) (token prices, official), [Apidog per-image estimates, 9 Sep 2026](https://apidog.com/blog/gpt-image-2-5-flare-vs-sunburst-vs-gpt-image-2/) (OpenAI publishes no per-image table for 2.5; input-image token count is my estimate) |
| Nano Banana 2 Lite (`nano-banana`) | input $0.25/M, image out $30/M tokens = $0.0336 per 1K image | **~$0.034**. Actual cost from `usageMetadata` | [Gemini API pricing (official)](https://ai.google.dev/gemini-api/docs/pricing), [model page, updated 2026-10-29](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite-image) |

Every call logs one structured line `cutsell_event name=shop.shape.call` with provider, kind, cost_usd, status,
latency_ms, hashed user and day (never image bytes or keys), and adds to the Redis hash
`cutsell:v1:shape-cost:<YYYYMMDD>` (`total_usd`, `<provider>_usd`, `<provider>_calls`; kept 400 days).

Worst case per account per day: 20 x $0.066 = $1.32 (GPT Image), 20 x $0.034 = $0.68, 20 x $0.018 = $0.36.

`scripts/shape_it_compare.py` (9 calls: 3 photos x 3 providers) -- estimated **~$0.35** total
(3 x $0.018 + 3 x $0.066 + 3 x $0.034). Dry run by default; `--yes-spend` to call; `--out` must be outside the repo.

## To deploy (PO decides; nothing here is deployed)

Service: the Render web service `cutsell-api` (`render.cutsell.yaml`, `autoDeploy: false`, manual deploy of
the server branch). The endpoint is API-only: nothing goes to Modal (no `cutsell_worker/` file changed, so the
Modal deploy workflow is not affected). Set on Render:
1. `CUTSELL_SHAPE_ENABLED=1` (when the app is ready to call it);
2. `CUTSELL_SHAPE_MODEL` = the chosen provider;
3. the chosen provider's key: `CUTSELL_SEEDREAM_API_KEY` (new BytePlus account + key), or `OPENAI_API_KEY`
   (check it is set on the API service), or `GEMINI_API_KEY` (today it is a worker key -- add it to the API
   service);
4. optional: limits and model ids above. `REDIS_URL` is already there.
Then merge `feature/shop-board-server` into the production server branch and deploy `cutsell-api` by hand.

## Prueba real (12 nov 2026, 4 fotos del PO: camisa, pantalón, tenis, suéter)
- **Nano Banana 2 Lite:** 4 de 4 bien, US$0.1376 en total (~$0.034 cada una), ~4.4 s cada una. Respeta el producto
  (colores, botones, bolsillo, suela); el cambio de forma es leve porque las fotos ya eran de catálogo.
- **OpenAI:** no se pudo probar. El proyecto de OpenAI de la llave del motor no tiene acceso a `gpt-image-2.5-flare`
  ni a `gpt-image-2` (HTTP 403 "does not have access to model"). Costo US$0 (OpenAI no cobra los rechazos).
  Para usarlo hay que darle acceso a modelos de imagen a ese proyecto en platform.openai.com (normalmente pide
  verificar la organización). `gpt-image-2` no acepta fondo transparente: el servidor ya le pide fondo de color
  sólido y lo quita después (`CUTSELL_SHAPE_OPENAI_BACKGROUND`).
- Seedream: sin llave, no se probó (decisión del PO: saltarlo).
- Fotos y resultados: solo en el bucket privado, `private-qa/shape-compare/20261112/`. Workflow:
  `.github/workflows/cutsell-shape-it-compare.yml` (corre solo al subir a `shape-compare/request`, un intento, máx. 8 llamadas).
