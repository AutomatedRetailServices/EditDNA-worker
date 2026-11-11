"""Shop board "Shape it": give ONE garment (or pair of shoes) a body shape -- the "ghost mannequin" look.

The phone removes the background and sends a PNG with alpha. One image model returns the SAME garment as if worn
by an invisible body (shoes: filled and standing). The garment must stay identical: colour, fabric, print, details.

Everything here is plain HTTPS through `requests` (already pinned for the API). Image bytes and keys are never
logged. Three providers, chosen with CUTSELL_SHAPE_MODEL (see docs/CUTSELL_SHAPE_IT.md):

    seedream     -- Seedream 5.0 Flash, BytePlus ModelArk    key CUTSELL_SEEDREAM_API_KEY (new)
    gpt-image    -- GPT Image 2.5 Flare, OpenAI images/edits key OPENAI_API_KEY (existing name)
    nano-banana  -- Nano Banana 2 Lite, Gemini API           key GEMINI_API_KEY (existing name)

The route (cutsell_app/shop_routes.py) owns HTTP; this module owns validation, the prompt, the provider calls,
the transparent-background finish, the per-account daily counter and the cost ledger.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import requests

# ------------------------------------------------------------------------------------------------- settings
_TRUTHY = {"1", "true", "yes", "on"}

KINDS = ("clothing", "shoes")

DEFAULT_DAILY_LIMIT = 20
# The auth middleware caps every JSON body at 8 MiB (MAX_JSON_BODY). Base64 grows bytes by 4/3, so 6 MB of PNG
# is the largest image that still fits through it; raising this above ~6.2 MB also needs that cap raised.
DEFAULT_MAX_BYTES = 6_000_000
DEFAULT_MAX_SIDE = 2048

PROVIDER_ALIASES = {
    "seedream": "seedream",
    "seedream-5.0-flash": "seedream",
    "seedream-5-0-flash": "seedream",
    "gpt-image": "gpt-image",
    "gpt-image-2.5": "gpt-image",
    "openai": "gpt-image",
    "nano-banana": "nano-banana",
    "nano-banana-2-lite": "nano-banana",
    "gemini": "nano-banana",
}
DEFAULT_PROVIDER = "nano-banana"   # its key name (GEMINI_API_KEY) already exists in CutSell

# Public list prices (USD), checked 11 Nov 2026 -- sources in docs/CUTSELL_SHAPE_IT.md.
SEEDREAM_PRICE_PER_IMAGE = 0.018            # Seedream 5.0 Flash, per generated image (reference images free)
OPENAI_TEXT_IN_PER_M = 5.00                 # gpt-image-2.5-flare / -sunburst
OPENAI_IMAGE_IN_PER_M = 8.00
OPENAI_IMAGE_OUT_PER_M = 30.00
OPENAI_ESTIMATE_PER_CALL = 0.066            # high, 1024x1024: ~1,756 out tokens ($0.053) + input image/text
GEMINI_IN_PER_M = 0.25                      # gemini-3.1-flash-lite-image, standard tier
GEMINI_TEXT_OUT_PER_M = 1.50
GEMINI_IMAGE_OUT_PER_M = 30.00
GEMINI_ESTIMATE_PER_CALL = 0.034            # $0.0336 per 1K image + input tokens

CONNECT_TIMEOUT = 10
READ_TIMEOUT = 150
MAX_ATTEMPTS = 2                            # one retry at most


def _env(name: str, default: str = "") -> str:
    return str(os.environ.get(name, "") or "").strip() or default


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(_env(name, str(default))))
    except ValueError:
        return default


def shape_enabled() -> bool:
    """Off unless CUTSELL_SHAPE_ENABLED=1: a deploy never starts spending money by itself."""
    return _env("CUTSELL_SHAPE_ENABLED", "0").lower() in _TRUTHY


def daily_limit() -> int:
    return _env_int("CUTSELL_SHAPE_DAILY_LIMIT", DEFAULT_DAILY_LIMIT)


def max_bytes() -> int:
    return _env_int("CUTSELL_SHAPE_MAX_BYTES", DEFAULT_MAX_BYTES)


def max_side() -> int:
    return _env_int("CUTSELL_SHAPE_MAX_SIDE", DEFAULT_MAX_SIDE)


def provider_name(value: str | None = None) -> str:
    raw = (value if value is not None else _env("CUTSELL_SHAPE_MODEL", DEFAULT_PROVIDER)).strip().lower()
    name = PROVIDER_ALIASES.get(raw)
    if not name:
        raise ShapeConfigError("unknown CUTSELL_SHAPE_MODEL")
    return name


# ------------------------------------------------------------------------------------------------- errors
class ShapeError(Exception):
    """Base. `code` is the safe string returned to the phone."""
    code = "shape_error"

    def __init__(self, code: str | None = None, message: str = ""):
        super().__init__(message or code or self.code)
        if code:
            self.code = code


class ShapeInputError(ShapeError):          # 400
    code = "bad_image"


class ShapeTooLargeError(ShapeError):       # 413
    code = "image_too_large"


class ShapeConfigError(ShapeError):         # 503
    code = "shape_provider_not_configured"


class ShapeProviderError(ShapeError):       # 502
    code = "shape_provider_error"


# ------------------------------------------------------------------------------------------------- input
@dataclass
class GarmentImage:
    rgba: Any                       # PIL.Image in RGBA
    width: int
    height: int
    key_rgb: tuple[int, int, int]   # solid background colour used when a provider cannot output alpha
    key_name: str

    def flattened_png(self) -> bytes:
        """The garment on a solid key colour (for models that do not read or write transparency)."""
        from PIL import Image
        base = Image.new("RGBA", self.rgba.size, self.key_rgb + (255,))
        base.alpha_composite(self.rgba)
        return _png_bytes(base.convert("RGB"))

    def rgba_png(self) -> bytes:
        return _png_bytes(self.rgba)


def _png_bytes(image) -> bytes:
    out = io.BytesIO()
    image.save(out, format="PNG", optimize=False)
    return out.getvalue()


def validate_kind(kind: Any) -> str:
    if not isinstance(kind, str) or kind.strip().lower() not in KINDS:
        raise ShapeInputError("bad_kind")
    return kind.strip().lower()


def decode_image(image_b64: Any, *, limit_bytes: int | None = None, limit_side: int | None = None) -> GarmentImage:
    """Strict: base64 -> PNG -> RGBA with a real alpha channel, within the byte and side limits."""
    limit_bytes = limit_bytes or max_bytes()
    limit_side = limit_side or max_side()
    if not isinstance(image_b64, str) or not image_b64.strip():
        raise ShapeInputError("bad_image")
    text = image_b64.strip()
    if text.startswith("data:"):
        head, _, text = text.partition(",")
        if "image/png" not in head or ";base64" not in head:
            raise ShapeInputError("bad_image")
    # Cheap size check before decoding anything.
    if (len(text) * 3) // 4 > limit_bytes + 3:
        raise ShapeTooLargeError("image_too_large")
    try:
        raw = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError):
        raise ShapeInputError("bad_image") from None
    if len(raw) > limit_bytes:
        raise ShapeTooLargeError("image_too_large")
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ShapeInputError("bad_image")
    from PIL import Image
    try:
        image = Image.open(io.BytesIO(raw))          # reads the header only
        width, height = image.size
    except Exception:
        raise ShapeInputError("bad_image") from None
    if image.format != "PNG" or width < 16 or height < 16:
        raise ShapeInputError("bad_image")
    if max(width, height) > limit_side:          # checked before the pixels are decoded (no decompression bomb)
        raise ShapeTooLargeError("image_too_large")
    has_alpha = image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info)
    if not has_alpha:
        raise ShapeInputError("image_needs_transparency")
    try:
        rgba = image.convert("RGBA")
    except Exception:
        raise ShapeInputError("bad_image") from None
    import numpy as np
    pixels = np.asarray(rgba)
    alpha = pixels[..., 3]
    opaque = alpha > 128
    if opaque.mean() < 0.01:                      # nothing there
        raise ShapeInputError("bad_image")
    if opaque.mean() > 0.995:                     # background was not removed
        raise ShapeInputError("image_needs_transparency")
    rgb = pixels[..., :3][opaque].astype("float32")
    luma = float((rgb @ np.array([0.299, 0.587, 0.114], dtype="float32")).mean())
    # The key must be far from the garment: white behind dark/mid clothes, near-black behind light ones.
    if luma > 170:
        key_rgb, key_name = (24, 24, 24), "solid near-black (#181818)"
    else:
        key_rgb, key_name = (255, 255, 255), "solid pure white (#FFFFFF)"
    return GarmentImage(rgba=rgba, width=width, height=height, key_rgb=key_rgb, key_name=key_name)


# ------------------------------------------------------------------------------------------------- prompt
_IDENTITY_RULES = (
    "The item must stay EXACTLY the same product: identical colours and shades, identical fabric or material and "
    "its texture, identical print, pattern, logo, text and graphics (same position, scale and orientation), "
    "identical stitching, seams, buttons, zips, pockets, laces, eyelets, soles, trims and labels. Do not redesign, "
    "restyle, recolour, clean up, add or remove any detail, and do not change the size or proportions of the item."
)

_CLOTHING = (
    "Edit this product photo of ONE clothing item into a professional e-commerce 'ghost mannequin' (invisible "
    "mannequin) shot: the garment keeps its natural 3D shape as if worn by an invisible body of average build -- "
    "filled out volume at the chest, shoulders, waist, hips or legs as fits the garment, natural drape and gentle "
    "folds, sleeves and legs hanging naturally, and where the neckline or waist opening is visible show the inside "
    "back of the garment through it. There is NO person, NO mannequin, NO hanger, NO skin, NO hands, NO head and "
    "NO body parts anywhere -- only the garment."
)

_SHOES = (
    "Edit this product photo of a pair of shoes (or one shoe) into a professional e-commerce shot: the shoes "
    "stand upright on their soles as if worn and filled by invisible feet -- the upper filled out with natural "
    "volume, no collapsed or flattened areas, laces neatly tied as they already are, viewed from the same angle. "
    "There are NO feet, NO legs, NO socks, NO shoe trees, NO stuffing paper and NO props -- only the shoes."
)


def build_prompt(kind: str, *, background: str) -> str:
    """`background` is "transparent" (model writes alpha) or the name of the solid key colour."""
    subject = _SHOES if kind == "shoes" else _CLOTHING
    if background == "transparent":
        bg = ("Output a PNG with a fully transparent background: no backdrop, no floor, no surface, no shadow, "
              "no reflection, no gradient -- only the item with clean, sharp edges.")
    else:
        bg = (f"Keep the background a perfectly flat, uniform {background} from edge to edge: no gradient, no "
              "floor, no surface, no shadow, no reflection, no vignette, no texture. The item has clean, sharp "
              "edges against it.")
    framing = ("Keep the same framing: same camera angle, same orientation, the item centred and filling the frame "
               "about as much as in the input, nothing cropped. Photorealistic, soft even studio lighting that does "
               "not change the item's colours. No text, no watermark, no border.")
    return " ".join((subject, _IDENTITY_RULES, bg, framing))


# ------------------------------------------------------------------------------------------------- providers
@dataclass
class ProviderResult:
    image_bytes: bytes
    cost_usd: float
    usage: dict = field(default_factory=dict)


def _post(url: str, *, headers: dict, json_body: dict, provider: str, session=None) -> dict:
    """One POST with at most one retry (network error, 429 or 5xx). Never logs the body or the key."""
    client = session or requests
    last = "unavailable"
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = client.post(url, headers=headers, json=json_body, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        except requests.RequestException as exc:
            last = exc.__class__.__name__
        else:
            status = int(getattr(response, "status_code", 0) or 0)
            if status == 200:
                try:
                    return response.json()
                except ValueError:
                    raise ShapeProviderError("shape_provider_error", f"{provider}: invalid JSON") from None
            last = f"HTTP {status}"
            if status not in (408, 429) and status < 500:
                break                               # 4xx: retrying will not help
        if attempt < MAX_ATTEMPTS - 1:
            time.sleep(1.5)
    raise ShapeProviderError("shape_provider_error", f"{provider}: {last}")


def _data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _b64(value: Any, provider: str) -> bytes:
    try:
        return base64.b64decode(str(value), validate=False)
    except (binascii.Error, ValueError):
        raise ShapeProviderError("shape_provider_error", f"{provider}: bad image data") from None


class SeedreamProvider:
    """Seedream 5.0 Flash on BytePlus ModelArk: POST {base}/images/generations (edit = prompt + `image`)."""
    name = "seedream"
    key_env = "CUTSELL_SEEDREAM_API_KEY"
    writes_alpha = False

    def __init__(self, session=None):
        self.session = session

    def model_id(self) -> str:
        return _env("CUTSELL_SEEDREAM_MODEL_ID", "seedream-5-0-flash-260915")

    def build_request(self, garment: GarmentImage, kind: str, key: str) -> tuple[str, dict, dict]:
        base = _env("CUTSELL_SEEDREAM_BASE_URL", "https://ark.ap-southeast.bytepluses.com/api/v3").rstrip("/")
        body = {
            "model": self.model_id(),
            "prompt": build_prompt(kind, background=garment.key_name),
            "image": _data_url(garment.flattened_png()),
            "size": _env("CUTSELL_SEEDREAM_SIZE", "2K"),
            "output_format": "png",
            "response_format": "b64_json",
            "watermark": False,
        }
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        return f"{base}/images/generations", headers, body

    def parse(self, data: dict) -> ProviderResult:
        items = data.get("data") or []
        if data.get("error") or not items or not items[0].get("b64_json"):
            raise ShapeProviderError("shape_no_image", "seedream: no image returned")
        usage = dict(data.get("usage") or {})
        images = int(usage.get("generated_images") or 1)
        return ProviderResult(_b64(items[0]["b64_json"], self.name), round(images * SEEDREAM_PRICE_PER_IMAGE, 6),
                              {"generated_images": images})


class OpenAIImageProvider:
    """GPT Image 2.5 Flare: POST https://api.openai.com/v1/images/edits (JSON, image as a base64 data URL)."""
    name = "gpt-image"
    key_env = "OPENAI_API_KEY"
    writes_alpha = True

    def __init__(self, session=None):
        self.session = session

    def model_id(self) -> str:
        return _env("CUTSELL_SHAPE_OPENAI_MODEL", "gpt-image-2.5-flare")

    @staticmethod
    def size_for(garment: GarmentImage) -> str:
        ratio = garment.width / garment.height
        if ratio > 1.2:
            return "1536x1024"
        if ratio < 1 / 1.2:
            return "1024x1536"
        return "1024x1024"

    def build_request(self, garment: GarmentImage, kind: str, key: str) -> tuple[str, dict, dict]:
        base = _env("CUTSELL_SHAPE_OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        body = {
            "model": self.model_id(),
            "prompt": build_prompt(kind, background="transparent"),
            "images": [{"image_url": _data_url(garment.rgba_png())}],
            "background": "transparent",
            "output_format": "png",
            "quality": _env("CUTSELL_SHAPE_OPENAI_QUALITY", "high"),
            "size": self.size_for(garment),
            "n": 1,
        }
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        return f"{base}/images/edits", headers, body

    def parse(self, data: dict) -> ProviderResult:
        items = data.get("data") or []
        if not items or not items[0].get("b64_json"):
            raise ShapeProviderError("shape_no_image", "gpt-image: no image returned")
        usage = dict(data.get("usage") or {})
        return ProviderResult(_b64(items[0]["b64_json"], self.name), self.cost(usage), usage)

    @staticmethod
    def cost(usage: dict) -> float:
        if not usage or "output_tokens" not in usage:
            return OPENAI_ESTIMATE_PER_CALL
        details = dict(usage.get("input_tokens_details") or {})
        image_in = int(details.get("image_tokens") or 0)
        text_in = int(details.get("text_tokens") or max(0, int(usage.get("input_tokens") or 0) - image_in))
        out = int(usage.get("output_tokens") or 0)
        return round((text_in * OPENAI_TEXT_IN_PER_M + image_in * OPENAI_IMAGE_IN_PER_M
                      + out * OPENAI_IMAGE_OUT_PER_M) / 1e6, 6)


class GeminiImageProvider:
    """Nano Banana 2 Lite (gemini-3.1-flash-lite-image): POST {base}/models/{model}:generateContent."""
    name = "nano-banana"
    key_env = "GEMINI_API_KEY"
    writes_alpha = False

    def __init__(self, session=None):
        self.session = session

    def model_id(self) -> str:
        return _env("CUTSELL_SHAPE_GEMINI_MODEL", "gemini-3.1-flash-lite-image")

    def build_request(self, garment: GarmentImage, kind: str, key: str) -> tuple[str, dict, dict]:
        base = _env("CUTSELL_SHAPE_GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1").rstrip("/")
        body = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": build_prompt(kind, background=garment.key_name)},
                    {"inline_data": {"mime_type": "image/png",
                                     "data": base64.b64encode(garment.flattened_png()).decode("ascii")}},
                ],
            }],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"], "candidateCount": 1},
        }
        headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
        return f"{base}/models/{self.model_id()}:generateContent", headers, body

    def parse(self, data: dict) -> ProviderResult:
        image = None
        for candidate in data.get("candidates") or []:
            for part in ((candidate.get("content") or {}).get("parts") or []):
                inline = part.get("inlineData") or part.get("inline_data") or {}
                if inline.get("data"):
                    image = inline["data"]
                    break
            if image:
                break
        if not image:
            raise ShapeProviderError("shape_no_image", "nano-banana: no image returned")
        usage = dict(data.get("usageMetadata") or {})
        return ProviderResult(_b64(image, self.name), self.cost(usage), usage)

    @staticmethod
    def cost(usage: dict) -> float:
        if not usage or "candidatesTokenCount" not in usage:
            return GEMINI_ESTIMATE_PER_CALL
        prompt = int(usage.get("promptTokenCount") or 0)
        out = int(usage.get("candidatesTokenCount") or 0)
        thoughts = int(usage.get("thoughtsTokenCount") or 0)
        image_out = out
        text_out = 0
        details = usage.get("candidatesTokensDetails") or []
        if details:
            image_out = sum(int(d.get("tokenCount") or 0) for d in details if str(d.get("modality")).upper() == "IMAGE")
            text_out = max(0, out - image_out)
        return round((prompt * GEMINI_IN_PER_M + image_out * GEMINI_IMAGE_OUT_PER_M
                      + (text_out + thoughts) * GEMINI_TEXT_OUT_PER_M) / 1e6, 6)


PROVIDERS: dict[str, type] = {
    SeedreamProvider.name: SeedreamProvider,
    OpenAIImageProvider.name: OpenAIImageProvider,
    GeminiImageProvider.name: GeminiImageProvider,
}


def get_provider(name: str | None = None, session=None):
    return PROVIDERS[provider_name(name)](session=session)


def provider_key_present(provider) -> bool:
    return bool(_env(provider.key_env))


def call_provider(provider, garment: GarmentImage, kind: str) -> ProviderResult:
    key = _env(provider.key_env)
    if not key:
        raise ShapeConfigError("shape_provider_not_configured", f"{provider.name}: {provider.key_env} missing")
    url, headers, body = provider.build_request(garment, kind, key)
    data = _post(url, headers=headers, json_body=body, provider=provider.name, session=provider.session)
    if not isinstance(data, dict):
        raise ShapeProviderError("shape_provider_error", f"{provider.name}: unexpected response")
    return provider.parse(data)


# ------------------------------------------------------------------------------------------------- output finish
def finish_output(image_bytes: bytes, garment: GarmentImage) -> bytes:
    """The provider's image -> a transparent PNG on the input's own canvas size (same framing for the phone).

    If the model wrote no transparency, the solid key background is removed: only key-coloured pixels CONNECTED
    to the image border become transparent, so key-coloured areas inside the garment are kept."""
    from PIL import Image
    import numpy as np
    try:
        out = Image.open(io.BytesIO(image_bytes))
        out.load()
        out = out.convert("RGBA")
    except Exception:
        raise ShapeProviderError("shape_provider_error", "output is not an image") from None
    alpha = np.asarray(out)[..., 3]
    if (alpha < 250).mean() < 0.02:              # practically opaque: key the background out
        out = _key_out(out, garment.key_rgb)
    # Fit into the input's canvas, centred.
    canvas = Image.new("RGBA", (garment.width, garment.height), (0, 0, 0, 0))
    scale = min(garment.width / out.width, garment.height / out.height)
    size = (max(1, round(out.width * scale)), max(1, round(out.height * scale)))
    if size != out.size:
        out = out.resize(size, Image.LANCZOS)
    canvas.alpha_composite(out, ((garment.width - size[0]) // 2, (garment.height - size[1]) // 2))
    return _png_bytes(canvas)


def _key_out(image, key_rgb: tuple[int, int, int], *, hard: float = 28.0, soft: float = 60.0):
    """Background removal for a solid key colour, border-connected only (numpy, no extra dependency)."""
    from PIL import Image
    import numpy as np
    rgb = np.asarray(image.convert("RGB")).astype("float32")
    dist = np.sqrt(((rgb - np.array(key_rgb, dtype="float32")) ** 2).sum(axis=-1))
    candidate = dist < soft
    # Connectivity on a small copy (fast), then back at full size.
    small_side = 256
    h, w = candidate.shape
    factor = max(1, int(np.ceil(max(h, w) / small_side)))
    sh, sw = -(-h // factor), -(-w // factor)
    padded = np.zeros((sh * factor, sw * factor), dtype=bool)
    padded[:h, :w] = candidate
    small = padded.reshape(sh, factor, sw, factor).mean(axis=(1, 3)) > 0.5
    reach = np.zeros_like(small)
    reach[0, :], reach[-1, :], reach[:, 0], reach[:, -1] = small[0, :], small[-1, :], small[:, 0], small[:, -1]
    for _ in range(sh + sw):
        grown = reach.copy()
        grown[1:, :] |= reach[:-1, :]
        grown[:-1, :] |= reach[1:, :]
        grown[:, 1:] |= reach[:, :-1]
        grown[:, :-1] |= reach[:, 1:]
        grown &= small
        if (grown == reach).all():
            break
        reach = grown
    # One cell of margin so the garment's edge pixels get the soft ramp.
    margin = reach.copy()
    margin[1:, :] |= reach[:-1, :]
    margin[:-1, :] |= reach[1:, :]
    margin[:, 1:] |= reach[:, :-1]
    margin[:, :-1] |= reach[:, 1:]
    region = np.repeat(np.repeat(margin, factor, axis=0), factor, axis=1)[:h, :w]
    ramp = np.clip((dist - hard) / (soft - hard), 0.0, 1.0)
    alpha = np.where(region & candidate, ramp * 255.0, 255.0).astype("uint8")
    rgba = np.dstack([rgb.astype("uint8"), alpha])
    return Image.fromarray(rgba, mode="RGBA")


# ------------------------------------------------------------------------------------------------- limits + costs
def _redis():
    from cutsell_worker.config import load_runtime_config
    config = load_runtime_config()
    if not config.redis_url:
        raise RuntimeError("REDIS_URL is not configured")
    from redis import Redis
    return Redis.from_url(config.redis_url)


def _now() -> float:
    return time.time()


def _day(now: float | None = None) -> str:
    return time.strftime("%Y%m%d", time.gmtime(_now() if now is None else now))


def user_scope(user_id: str) -> str:
    return hashlib.sha256(str(user_id).encode()).hexdigest()[:20]


def day_key(user_id: str, now: float | None = None) -> str:
    return f"cutsell:v1:shape-day:{user_scope(user_id)}:{_day(now)}"


def cost_key(now: float | None = None) -> str:
    return f"cutsell:v1:shape-cost:{_day(now)}"


def is_signed_in(user_id: str, client) -> bool:
    """A real account = an Apple identity bound to it. Guest and test-probe accounts have none.

    (A guest that later signed in with Apple keeps its usr_ id, so the binding is checked, not the id prefix.)"""
    from cutsell_worker.auth import user_apple_key
    return client.get(user_apple_key(user_id)) is not None


def take_daily_slot(user_id: str, client, *, limit: int | None = None) -> tuple[bool, int]:
    """Atomically count one Shape it for this account today (UTC). Over the limit -> refused and not counted."""
    limit = limit or daily_limit()
    key = day_key(user_id)
    count = int(client.incr(key))
    if count == 1:
        client.expire(key, 2 * 24 * 3600)
    if count > limit:
        client.decr(key)
        return False, count - 1
    return True, count


def give_back_daily_slot(user_id: str, client) -> None:
    """A call that produced no image does not use up one of her 20."""
    try:
        key = day_key(user_id)
        if int(client.decr(key)) < 0:
            client.set(key, 0)
    except Exception:
        pass


def record_cost(*, client, provider: str, kind: str, cost_usd: float, user_id: str, status: str,
                latency_ms: int, log: Callable[..., None] | None = None) -> bool:
    """One structured line per call (never image bytes or keys) and today's running total in Redis."""
    if log is None:
        from cutsell_worker.commercial_observability import capture_operational_event as log
    log("shop.shape.call", provider=provider, kind=kind, cost_usd=round(float(cost_usd), 6), status=status,
        latency_ms=int(latency_ms), user=user_scope(user_id), day=_day())
    if cost_usd <= 0:
        return True
    try:
        key = cost_key()
        pipe = client.pipeline()
        pipe.hincrbyfloat(key, "total_usd", float(cost_usd))
        pipe.hincrbyfloat(key, f"{provider}_usd", float(cost_usd))
        pipe.hincrby(key, f"{provider}_calls", 1)
        pipe.expire(key, 400 * 24 * 3600)
        pipe.execute()
        return True
    except Exception:
        return False


def shape(garment: GarmentImage, kind: str, *, provider=None) -> tuple[bytes, ProviderResult, Any]:
    """Run one provider and finish its output. Used by the route and by scripts/shape_it_compare.py."""
    provider = provider or get_provider()
    result = call_provider(provider, garment, kind)
    return finish_output(result.image_bytes, garment), result, provider
