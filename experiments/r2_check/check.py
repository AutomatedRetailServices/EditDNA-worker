"""R2 compatibility check for CutSell storage (test only, no production change).

Reads R2 credentials from the environment as the standard AWS_* variables plus
AWS_ENDPOINT_URL_S3, exactly as Render/Modal would after the switch, and builds
boto3 clients the same way the product code does (no endpoint argument). Uses the
product's own presign functions where they need no database.

Prints only pass/fail lines and HTTP status codes. Never prints keys, URLs or
signatures. Everything written goes under a test prefix and is deleted at the end.
"""
from __future__ import annotations

import json
import os
import sys
import uuid

import boto3
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

os.environ.setdefault("CUTSELL_UPLOAD_PREFIX", "pruebas/r2-check/uploads/")
BUCKET = os.environ["S3_BUCKET"]
RESULTS: list[dict] = []
CREATED: list[str] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append({"prueba": name, "ok": ok, "detalle": detail})
    print(f"{'OK   ' if ok else 'FALLA'} {name}" + (f" -- {detail}" if detail else ""), flush=True)


def short_error(exc: Exception) -> str:
    code = getattr(exc, "response", {}).get("Error", {}).get("Code") if hasattr(exc, "response") else None
    return code or type(exc).__name__


def body_error(resp: requests.Response) -> str:
    text = resp.text or ""
    for tag in ("Code", "Message"):
        start, end = text.find(f"<{tag}>"), text.find(f"</{tag}>")
        if start >= 0 and end > start:
            return f"HTTP {resp.status_code} {text[start + len(tag) + 2:end][:120]}"
    return f"HTTP {resp.status_code}"


def client():
    # Same construction as cutsell_worker (region from AWS_REGION, no endpoint arg).
    return boto3.client("s3", region_name=os.environ.get("AWS_REGION") or "us-east-1")


def check_endpoint_is_r2() -> None:
    host = client().meta.endpoint_url or ""
    record("boto3 usa el endpoint de R2 sin cambiar codigo", host.endswith(".r2.cloudflarestorage.com"))


def check_put_get_delete() -> None:
    s3 = client()
    key = f"pruebas/r2-check/basic-{uuid.uuid4().hex}.txt"
    data = b"cutsell r2 check " + uuid.uuid4().hex.encode()
    try:
        s3.put_object(Bucket=BUCKET, Key=key, Body=data, ContentType="text/plain")
        CREATED.append(key)
        record("subir archivo", True)
    except Exception as exc:
        record("subir archivo", False, short_error(exc))
        return
    try:
        got = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        record("leer archivo", got == data)
    except Exception as exc:
        record("leer archivo", False, short_error(exc))
    try:
        url = s3.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=300)
        resp = requests.get(url, timeout=60)
        record("bajar con enlace firmado (vista previa y video final)", resp.status_code == 200 and resp.content == data,
               f"HTTP {resp.status_code}")
    except Exception as exc:
        record("bajar con enlace firmado (vista previa y video final)", False, short_error(exc))
    try:
        s3.delete_object(Bucket=BUCKET, Key=key)
        CREATED.remove(key)
        try:
            s3.head_object(Bucket=BUCKET, Key=key)
            record("borrar archivo", False, "sigue existiendo")
        except Exception:
            record("borrar archivo", True)
    except Exception as exc:
        record("borrar archivo", False, short_error(exc))


def check_multipart_put() -> None:
    """Mirrors cutsell_worker.multipart_uploads: create -> presigned PUT per part -> complete."""
    s3 = boto3.client("s3", region_name="us-east-1")  # product presign_multipart_part uses this default
    key = f"pruebas/r2-check/multipart-{uuid.uuid4().hex}.mp4"
    part1 = os.urandom(5 * 1024 * 1024)
    part2 = os.urandom(300 * 1024)
    try:
        upload_id = s3.create_multipart_upload(Bucket=BUCKET, Key=key, ContentType="video/mp4")["UploadId"]
    except Exception as exc:
        record("video principal: empezar subida en partes", False, short_error(exc))
        return
    record("video principal: empezar subida en partes", True)
    parts = []
    for number, chunk in ((1, part1), (2, part2)):
        url = s3.generate_presigned_url("upload_part", Params={
            "Bucket": BUCKET, "Key": key, "UploadId": upload_id, "PartNumber": number}, ExpiresIn=900)
        resp = requests.put(url, data=chunk, timeout=120)
        ok = resp.status_code == 200 and bool(resp.headers.get("ETag"))
        record(f"video principal: subir parte {number} con enlace firmado PUT", ok,
               "" if ok else body_error(resp))
        if not ok:
            s3.abort_multipart_upload(Bucket=BUCKET, Key=key, UploadId=upload_id)
            return
        parts.append({"PartNumber": number, "ETag": resp.headers["ETag"]})
    try:
        listed = s3.list_parts(Bucket=BUCKET, Key=key, UploadId=upload_id).get("Parts", [])
        record("video principal: listar partes (reanudar subida)", len(listed) == 2)
    except Exception as exc:
        record("video principal: listar partes (reanudar subida)", False, short_error(exc))
    try:
        s3.complete_multipart_upload(Bucket=BUCKET, Key=key, UploadId=upload_id, MultipartUpload={"Parts": parts})
        CREATED.append(key)
        size = s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
        record("video principal: juntar partes", size == len(part1) + len(part2))
    except Exception as exc:
        record("video principal: juntar partes", False, short_error(exc))


def check_multipart_variants() -> None:
    """Diagnose a failing part PUT: region and checksum settings; prints query param NAMES only."""
    from urllib.parse import parse_qs, urlparse
    from botocore.config import Config

    variants = [
        ("region us-east-1, checksum por defecto", "us-east-1", None),
        ("region auto, checksum por defecto", "auto", None),
        ("region us-east-1, checksum solo si hace falta", "us-east-1", "when_required"),
        ("region auto, checksum solo si hace falta", "auto", "when_required"),
    ]
    for label, region, checksum in variants:
        cfg = Config(request_checksum_calculation=checksum, response_checksum_validation=checksum) if checksum else None
        s3 = boto3.client("s3", region_name=region, config=cfg)
        key = f"pruebas/r2-check/variant-{uuid.uuid4().hex}.mp4"
        try:
            upload_id = s3.create_multipart_upload(Bucket=BUCKET, Key=key, ContentType="video/mp4")["UploadId"]
            url = s3.generate_presigned_url("upload_part", Params={
                "Bucket": BUCKET, "Key": key, "UploadId": upload_id, "PartNumber": 1}, ExpiresIn=900)
            names = sorted(k for k in parse_qs(urlparse(url).query) if k.lower().startswith("x-amz-") and
                           k not in ("X-Amz-Signature", "X-Amz-Credential", "X-Amz-Security-Token"))
            resp = requests.put(url, data=os.urandom(5 * 1024 * 1024), timeout=120)
            ok = resp.status_code == 200
            record(f"parte PUT [{label}]", ok, ("" if ok else body_error(resp)) + f" params={','.join(names)}")
            s3.abort_multipart_upload(Bucket=BUCKET, Key=key, UploadId=upload_id)
        except Exception as exc:
            record(f"parte PUT [{label}]", False, short_error(exc))


def try_post(name: str, presign: dict, payload: bytes, filename: str) -> None:
    files = {"file": (filename, payload, presign["fields"].get("Content-Type", "application/octet-stream"))}
    try:
        resp = requests.post(presign["upload_url"], data=presign["fields"], files=files, timeout=120)
    except Exception as exc:
        record(name, False, short_error(exc))
        return
    ok = resp.status_code in (200, 201, 204)
    record(name, ok, "" if ok else body_error(resp))
    if ok:
        CREATED.append(presign["object_key"])


def check_presigned_post() -> None:
    from cutsell_worker import overlay_uploads, uploads

    user, project = "r2-check-user", "r2-check-project"
    audio = os.urandom(64 * 1024)
    image = os.urandom(32 * 1024)
    video = os.urandom(128 * 1024)
    try:
        vo = uploads.create_presigned_voice_over_upload(project_id=project, user_id=user, original_name="voz.m4a",
                                                        content_type="audio/mp4", size_bytes=len(audio))
        try_post("voz en off: subida POST de formulario", vo, audio, "voz.m4a")
    except Exception as exc:
        record("voz en off: subida POST de formulario", False, short_error(exc))
    try:
        ov = overlay_uploads.create_overlay_presigned_upload(project_id=project, user_id=user,
                                                             original_name="foto.png", content_type="image/png",
                                                             size_bytes=len(image))
        try_post("overlay / b-roll: subida POST de formulario", ov, image, "foto.png")
    except Exception as exc:
        record("overlay / b-roll: subida POST de formulario", False, short_error(exc))
    try:
        sv = uploads.create_presigned_upload(project_id=project, user_id=user, original_name="video.mp4",
                                             content_type="video/mp4", size_bytes=len(video))
        try_post("video simple (/v1/uploads/presign, usado por prueba_real.py): POST", sv, video, "video.mp4")
    except Exception as exc:
        record("video simple (/v1/uploads/presign, usado por prueba_real.py): POST", False, short_error(exc))


def check_presigned_put_alternative() -> None:
    """What the POST routes would look like as single presigned PUT (the proposed fix)."""
    s3 = client()
    key = f"pruebas/r2-check/put-{uuid.uuid4().hex}.m4a"
    data = os.urandom(64 * 1024)
    url = s3.generate_presigned_url("put_object", Params={"Bucket": BUCKET, "Key": key, "ContentType": "audio/mp4"},
                                    ExpiresIn=900)
    resp = requests.put(url, data=data, headers={"Content-Type": "audio/mp4"}, timeout=120)
    ok = resp.status_code == 200
    record("alternativa: subida simple con enlace firmado PUT", ok, "" if ok else body_error(resp))
    if ok:
        CREATED.append(key)


def cleanup() -> None:
    s3 = client()
    for key in list(CREATED):
        try:
            s3.delete_object(Bucket=BUCKET, Key=key)
        except Exception:
            pass
    leftover = s3.list_objects_v2(Bucket=BUCKET, Prefix="pruebas/r2-check/").get("KeyCount", 0)
    record("limpieza: no queda nada de la prueba", leftover == 0, f"quedan {leftover}" if leftover else "")


def main() -> int:
    print(f"boto3 {boto3.__version__}", flush=True)
    for step in (check_endpoint_is_r2, check_put_get_delete, check_multipart_put, check_multipart_variants,
                 check_presigned_post, check_presigned_put_alternative):
        try:
            step()
        except Exception as exc:
            record(step.__name__, False, short_error(exc))
    cleanup()
    print("RESUMEN " + json.dumps(RESULTS, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
