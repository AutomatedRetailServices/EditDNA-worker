"""Resumable direct-to-S3 multipart upload sessions for mobile CutSell.

FastAPI only creates/resumes/completes sessions and signs individual S3 parts. Video
bytes travel directly between the mobile client and S3. Session metadata is kept in
Redis so ownership and expected part count can be revalidated after app restarts.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any

from .config import load_runtime_config
from .uploads import prepare_upload_target

DEFAULT_PART_SIZE = 16 * 1024 * 1024
SESSION_TTL_SEC = 24 * 60 * 60
MAX_PARTS = 10_000


def _session_key(upload_id: str) -> str:
    if not upload_id:
        raise ValueError("upload_id is required")
    digest = hashlib.sha256(upload_id.encode()).hexdigest()
    return f"cutsell:v1:multipart:{digest}"


def _redis_client(client=None):
    if client is not None:
        return client
    config = load_runtime_config()
    if not config.redis_url:
        raise RuntimeError("REDIS_URL is required for multipart upload sessions")
    from redis import Redis
    return Redis.from_url(config.redis_url)


def _s3_client(client=None, *, region: str | None = None):
    if client is not None:
        return client
    import boto3
    return boto3.client("s3", region_name=region or "us-east-1")


def _load_session(upload_id: str, *, user_id: str, project_id: str, redis_client=None) -> dict[str, Any]:
    client = _redis_client(redis_client)
    raw = client.get(_session_key(upload_id))
    if raw is None:
        raise KeyError("multipart upload session not found")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    session = json.loads(str(raw))
    if session.get("upload_id") != upload_id:
        raise ValueError("multipart upload session is invalid")
    if session.get("user_id") != user_id or session.get("project_id") != project_id:
        raise PermissionError("multipart upload session ownership mismatch")
    return session


def start_multipart_upload(
    *,
    project_id: str,
    user_id: str,
    original_name: str,
    content_type: str | None,
    size_bytes: int,
    part_size: int = DEFAULT_PART_SIZE,
    s3=None,
    redis_client=None,
) -> dict[str, Any]:
    if int(part_size) < 5 * 1024 * 1024:
        raise ValueError("multipart part_size must be at least 5 MiB")
    target = prepare_upload_target(
        project_id=project_id,
        user_id=user_id,
        original_name=original_name,
        content_type=content_type,
        size_bytes=size_bytes,
    )
    resolved_part_size = int(part_size)
    part_count = int(math.ceil(target["size_bytes"] / resolved_part_size))
    if not 1 <= part_count <= MAX_PARTS:
        raise ValueError("multipart upload requires too many parts")

    client = _s3_client(s3, region=target["region"])
    response = client.create_multipart_upload(
        Bucket=target["bucket"],
        Key=target["object_key"],
        ContentType=target["content_type"],
    )
    upload_id = str(response.get("UploadId") or "")
    if not upload_id:
        raise RuntimeError("S3 did not return a multipart upload id")
    session = {
        "schema_version": "cutsell.multipart.v1",
        "upload_id": upload_id,
        "project_id": project_id,
        "user_id": user_id,
        "bucket": target["bucket"],
        "object_key": target["object_key"],
        "source_uri": target["source_uri"],
        "content_type": target["content_type"],
        "size_bytes": target["size_bytes"],
        "part_size": resolved_part_size,
        "part_count": part_count,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expires_in": SESSION_TTL_SEC,
    }
    redis_target = _redis_client(redis_client)
    try:
        redis_target.set(_session_key(upload_id), json.dumps(session, ensure_ascii=False), ex=SESSION_TTL_SEC)
    except Exception:
        try:
            client.abort_multipart_upload(
                Bucket=target["bucket"], Key=target["object_key"], UploadId=upload_id
            )
        finally:
            raise
    return session


def presign_multipart_part(
    *,
    upload_id: str,
    user_id: str,
    project_id: str,
    part_number: int,
    expires_in: int = 900,
    s3=None,
    redis_client=None,
) -> dict[str, Any]:
    session = _load_session(
        upload_id, user_id=user_id, project_id=project_id, redis_client=redis_client
    )
    part_number = int(part_number)
    if not 1 <= part_number <= int(session["part_count"]):
        raise ValueError("multipart part_number is outside expected range")
    if not 60 <= int(expires_in) <= 3600:
        raise ValueError("multipart part URL expiry must be between 60 and 3600 seconds")
    client = _s3_client(s3)
    url = client.generate_presigned_url(
        "upload_part",
        Params={
            "Bucket": session["bucket"],
            "Key": session["object_key"],
            "UploadId": upload_id,
            "PartNumber": part_number,
        },
        ExpiresIn=int(expires_in),
    )
    return {
        "upload_id": upload_id,
        "part_number": part_number,
        "upload_url": url,
        "expires_in": int(expires_in),
    }


def list_multipart_parts(
    *, upload_id: str, user_id: str, project_id: str, s3=None, redis_client=None
) -> dict[str, Any]:
    session = _load_session(
        upload_id, user_id=user_id, project_id=project_id, redis_client=redis_client
    )
    client = _s3_client(s3)
    response = client.list_parts(
        Bucket=session["bucket"], Key=session["object_key"], UploadId=upload_id
    )
    parts = [
        {
            "part_number": int(item["PartNumber"]),
            "etag": str(item.get("ETag") or ""),
            "size": int(item.get("Size") or 0),
        }
        for item in response.get("Parts", [])
    ]
    parts.sort(key=lambda item: item["part_number"])
    return {
        "upload_id": upload_id,
        "source_uri": session["source_uri"],
        "part_count": session["part_count"],
        "uploaded_parts": parts,
        "uploaded_part_numbers": [item["part_number"] for item in parts],
    }


def complete_multipart_upload(
    *,
    upload_id: str,
    user_id: str,
    project_id: str,
    parts: list[dict[str, Any]],
    s3=None,
    redis_client=None,
) -> dict[str, Any]:
    session = _load_session(
        upload_id, user_id=user_id, project_id=project_id, redis_client=redis_client
    )
    expected = list(range(1, int(session["part_count"]) + 1))
    normalized = []
    for item in parts:
        number = int(item.get("part_number") or 0)
        etag = str(item.get("etag") or "").strip()
        if not etag:
            raise ValueError("multipart completion requires an ETag for every part")
        normalized.append({"PartNumber": number, "ETag": etag})
    normalized.sort(key=lambda item: item["PartNumber"])
    if [item["PartNumber"] for item in normalized] != expected:
        raise ValueError("multipart completion must contain every expected part exactly once")

    client = _s3_client(s3)
    client.complete_multipart_upload(
        Bucket=session["bucket"],
        Key=session["object_key"],
        UploadId=upload_id,
        MultipartUpload={"Parts": normalized},
    )
    redis_target = _redis_client(redis_client)
    redis_target.delete(_session_key(upload_id))
    return {
        "upload_id": upload_id,
        "state": "uploaded",
        "source_uri": session["source_uri"],
        "object_key": session["object_key"],
        "size_bytes": session["size_bytes"],
        "content_type": session["content_type"],
    }


def abort_multipart_upload(
    *, upload_id: str, user_id: str, project_id: str, s3=None, redis_client=None
) -> dict[str, Any]:
    session = _load_session(
        upload_id, user_id=user_id, project_id=project_id, redis_client=redis_client
    )
    client = _s3_client(s3)
    client.abort_multipart_upload(
        Bucket=session["bucket"], Key=session["object_key"], UploadId=upload_id
    )
    _redis_client(redis_client).delete(_session_key(upload_id))
    return {"upload_id": upload_id, "state": "canceled"}
