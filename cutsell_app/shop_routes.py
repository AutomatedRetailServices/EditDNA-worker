"""Shop board "Shape it" (PO, Nov 2026).

    POST /v1/shop/shape   {"image_png_base64": "<PNG with alpha>", "kind": "clothing"|"shoes"}
      200 {"image_png_base64": "...", "provider": "<name>", "cost_usd": <float>}
      400 {"error": "bad_kind"|"bad_image"|"image_needs_transparency"}
      401 signed out (no/invalid bearer: AuthScopeMiddleware) or {"error": "sign_in_required"} (guest/probe)
      413 {"error": "image_too_large", "max_bytes": .., "max_side": ..}
      429 {"error": "shape_daily_limit", "limit": 20}
      502 {"error": "shape_provider_error"|"shape_no_image", "provider": "<name>"}
      503 {"error": "shape_disabled"|"shape_provider_not_configured"|"shape_meter_unavailable"}

Free during testing (no credits); 20 per account per UTC day. Contract and env vars: docs/CUTSELL_SHAPE_IT.md.
"""
from __future__ import annotations

import base64
import json
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from cutsell_app import shop_shape as shape

router = APIRouter()


def _error(status: int, code: str, **extra) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": code, **extra})


@router.post("/v1/shop/shape")
async def shop_shape(request: Request):
    user_id = getattr(request.state, "auth_user_id", None)
    if not user_id:
        return _error(401, "sign_in_required")
    try:
        payload = json.loads(await request.body() or b"null")
    except ValueError:
        return _error(400, "bad_request")
    if not isinstance(payload, dict):
        return _error(400, "bad_request")
    return await run_in_threadpool(_shape_sync, str(user_id), payload)


def _shape_sync(user_id: str, payload: dict):
    try:
        client = shape._redis()
        signed_in = shape.is_signed_in(user_id, client)
    except Exception:
        return _error(503, "shape_meter_unavailable")
    if not signed_in:
        return _error(401, "sign_in_required")
    if not shape.shape_enabled():
        return _error(503, "shape_disabled")
    try:
        provider = shape.get_provider()
    except shape.ShapeConfigError:
        return _error(503, "shape_provider_not_configured")
    if not shape.provider_key_present(provider):
        return _error(503, "shape_provider_not_configured")

    try:
        kind = shape.validate_kind(payload.get("kind"))
        garment = shape.decode_image(payload.get("image_png_base64"))
    except shape.ShapeTooLargeError:
        return _error(413, "image_too_large", max_bytes=shape.max_bytes(), max_side=shape.max_side())
    except shape.ShapeInputError as exc:
        return _error(400, exc.code)

    limit = shape.daily_limit()
    try:
        allowed, _count = shape.take_daily_slot(user_id, client, limit=limit)
    except Exception:                                   # fail closed, like the monthly meter
        return _error(503, "shape_meter_unavailable")
    if not allowed:
        return _error(429, "shape_daily_limit", limit=limit)

    started = time.monotonic()
    try:
        result = shape.call_provider(provider, garment, kind)
    except shape.ShapeConfigError:
        shape.give_back_daily_slot(user_id, client)
        return _error(503, "shape_provider_not_configured")
    except shape.ShapeError as exc:
        shape.give_back_daily_slot(user_id, client)
        shape.record_cost(client=client, provider=provider.name, kind=kind, cost_usd=0.0, user_id=user_id,
                          status=exc.code, latency_ms=int((time.monotonic() - started) * 1000))
        code = exc.code if exc.code in ("shape_no_image",) else "shape_provider_error"
        return _error(502, code, provider=provider.name)
    except Exception:
        shape.give_back_daily_slot(user_id, client)
        return _error(502, "shape_provider_error", provider=provider.name)

    latency_ms = int((time.monotonic() - started) * 1000)
    try:
        finished = shape.finish_output(result.image_bytes, garment)
    except Exception:
        # The provider was paid: the cost is recorded, the slot stays used.
        shape.record_cost(client=client, provider=provider.name, kind=kind, cost_usd=result.cost_usd,
                          user_id=user_id, status="bad_output", latency_ms=latency_ms)
        return _error(502, "shape_provider_error", provider=provider.name)
    shape.record_cost(client=client, provider=provider.name, kind=kind, cost_usd=result.cost_usd,
                      user_id=user_id, status="ok", latency_ms=latency_ms)
    return {
        "image_png_base64": base64.b64encode(finished).decode("ascii"),
        "provider": provider.name,
        "cost_usd": round(float(result.cost_usd), 6),
    }
