"""Shape it through kie.ai (PO 16 Nov 2026: "podemos correr el modelo de imagen nano banana por kie api").

The kie.ai key lives only in the Modal secret, so this runs on Modal (modal_cutsell_worker.shape_kie); the Render
API reaches it through that endpoint with the wake token (cutsell_app/shop_shape.KieProvider).

kie.ai flow (docs.kie.ai, checked 16 Nov 2026):
  1. POST https://kieai.redpandaai.co/api/file-base64-upload  {base64Data, uploadPath, fileName}  -> data.downloadUrl
     (kie deletes uploaded files after 24 h)
  2. POST https://api.kie.ai/api/v1/jobs/createTask  {model, input: {prompt, image_urls, output_format, image_size}}
     -> data.taskId
  3. GET  https://api.kie.ai/api/v1/jobs/recordInfo?taskId=..  until data.state is "success" (data.resultJson =
     '{"resultUrls": [..]}') or "fail" (data.failMsg)
  4. GET the first result URL (kie keeps it ~24 h).
Price: google/nano-banana-edit = 4 credits ~= US$0.02 per image (kie's own page, 16 Nov 2026).

Only `requests` is imported here: the small Modal image that serves the endpoint has nothing else.
"""
from __future__ import annotations

import base64
import json
import os
import time
import uuid

import requests

UPLOAD_URL = "https://kieai.redpandaai.co/api/file-base64-upload"
CREATE_URL = "https://api.kie.ai/api/v1/jobs/createTask"
RECORD_URL = "https://api.kie.ai/api/v1/jobs/recordInfo"
DEFAULT_MODEL = "google/nano-banana-edit"
PRICE_PER_IMAGE = {   # approximate (kie.ai credits; checked per run in the probe)
    "google/nano-banana-edit": 0.02, "nano-banana-2": 0.04, "nano-banana-pro": 0.09,
    "bytedance/seedream-v4-edit": 0.0175, "qwen/image-edit": 0.02,
}
DEFAULT_PRICE = 0.02
CONNECT_TIMEOUT, READ_TIMEOUT = 10, 60
POLL_FIRST_SEC, POLL_MAX_SEC = 2.0, 5.0
TOTAL_WAIT_SEC = 240.0


class KieError(Exception):
    """`code` is the Shape it error code ("shape_provider_error" / "shape_no_image" / "shape_provider_not_configured")."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(detail or code)
        self.code = code
        self.detail = detail


def model_id() -> str:
    return os.environ.get("CUTSELL_SHAPE_KIE_MODEL", "").strip() or DEFAULT_MODEL


def price(model: str | None = None) -> float:
    return PRICE_PER_IMAGE.get(model or model_id(), DEFAULT_PRICE)


def _check(response, step: str) -> dict:
    status = int(getattr(response, "status_code", 0) or 0)
    if status != 200:
        raise KieError("shape_provider_error", f"kie {step}: HTTP {status}")
    try:
        data = response.json()
    except ValueError:
        raise KieError("shape_provider_error", f"kie {step}: invalid JSON") from None
    code = data.get("code") if isinstance(data, dict) else None
    if code not in (None, 200, "200"):
        msg = str((data or {}).get("msg") or "")[:120]
        raise KieError("shape_provider_error", f"kie {step}: code {code} {msg}".strip())
    return data.get("data") or {}


def task_input(model: str, prompt: str, image_url: str) -> dict:
    """Each kie.ai image model names its fields differently (docs, 16 Nov 2026)."""
    if model in ("nano-banana-2", "nano-banana-pro"):
        return {"prompt": prompt, "image_input": [image_url], "aspect_ratio": "auto", "resolution": "1K",
                "output_format": "png"}
    if model == "qwen/image-edit":
        return {"prompt": prompt[:2000], "image_url": image_url, "output_format": "png", "num_images": "1"}
    if model.startswith("bytedance/seedream"):
        return {"prompt": prompt, "image_urls": [image_url], "image_resolution": "1K", "max_images": 1}
    return {"prompt": prompt, "image_urls": [image_url], "output_format": "png", "image_size": "auto"}


def shape_png(png: bytes, prompt: str, *, key: str | None = None, session=None, sleep=time.sleep,
              clock=time.monotonic, model: str | None = None) -> tuple[bytes, dict]:
    """The garment PNG (on its solid key colour) -> the model's image bytes, plus {"model", "cost_usd", "ms"}."""
    key = (key if key is not None else os.environ.get("KIE_API_KEY", "")).strip()
    if not key:
        raise KieError("shape_provider_not_configured", "KIE_API_KEY missing")
    client = session or requests
    auth = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    model = model or model_id()
    started = clock()
    try:
        uploaded = _check(client.post(UPLOAD_URL, headers=auth, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), json={
            "base64Data": "data:image/png;base64," + base64.b64encode(png).decode("ascii"),
            "uploadPath": "cutsell/shape", "fileName": f"{uuid.uuid4().hex}.png"}), "upload")
        image_url = str(uploaded.get("downloadUrl") or uploaded.get("fileUrl") or "")
        if not image_url.startswith("https://"):
            raise KieError("shape_provider_error", "kie upload: no URL")
        task = _check(client.post(CREATE_URL, headers=auth, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), json={
            "model": model, "input": task_input(model, prompt, image_url)}), "createTask")
        task_id = str(task.get("taskId") or "")
        if not task_id:
            raise KieError("shape_provider_error", "kie createTask: no taskId")
        wait = POLL_FIRST_SEC
        while True:
            if clock() - started > TOTAL_WAIT_SEC:
                raise KieError("shape_provider_error", "kie: timed out")
            sleep(wait)
            wait = min(POLL_MAX_SEC, wait * 1.5)
            record = _check(client.get(RECORD_URL, headers=auth, params={"taskId": task_id},
                                       timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)), "recordInfo")
            state = str(record.get("state") or "")
            if state == "fail":
                raise KieError("shape_no_image", f"kie task failed: {str(record.get('failMsg') or '')[:120]}")
            if state != "success":
                continue
            try:
                urls = json.loads(record.get("resultJson") or "{}").get("resultUrls") or []
            except (TypeError, ValueError):
                urls = []
            if not urls:
                raise KieError("shape_no_image", "kie: success without an image")
            result = client.get(str(urls[0]), timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
            if int(getattr(result, "status_code", 0) or 0) != 200 or not result.content:
                raise KieError("shape_provider_error", "kie: result download failed")
            return result.content, {"model": model, "cost_usd": price(model), "ms": int((clock() - started) * 1000)}
    except requests.RequestException as exc:
        raise KieError("shape_provider_error", f"kie: {exc.__class__.__name__}") from None
