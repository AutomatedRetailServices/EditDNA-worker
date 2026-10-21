"""Editor v5 official sound-effect library (v1).

The iPhone editor offers a small, fixed set of sound effects ("Sounds"). The draft only ever names
one by `sfx_id`; the files themselves live in the CutSell media bucket under `assets/sfx/v1/`, so
the client never supplies a URI. The export downloads the file named here; the editor gets
short-lived signed links from `GET /v1/sfx/manifest`.

Durations are read from `assets/sfx/v1/manifest.json` when that object exists (written when the
files are uploaded); otherwise they stay `None` and the app measures the file it downloads.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace

from .config import load_runtime_config
from .storage_routing import presigned_get

SFX_LIBRARY_VERSION = "v1"
SFX_S3_PREFIX = f"assets/sfx/{SFX_LIBRARY_VERSION}/"
SFX_MANIFEST_KEY = f"{SFX_S3_PREFIX}manifest.json"
SFX_LICENSE = "CC0 1.0 (Kenney Interface Sounds; Keyboard Soundpack by unicaegames)"


@dataclass(frozen=True)
class SoundEffect:
    sfx_id: str
    name: str
    file: str
    duration_sec: float | None = None
    license: str = SFX_LICENSE


SFX_LIBRARY: tuple[SoundEffect, ...] = (
    SoundEffect("tap", "Tap", "tap_select.m4a"),
    SoundEffect("good", "Good", "good.m4a"),
    SoundEffect("wrong", "Wrong", "wrong.m4a"),
    SoundEffect("glass", "Glass", "glass.m4a"),
    SoundEffect("typing", "Typing", "typing_3s.m4a"),
    SoundEffect("typing_long", "Typing (long)", "typing.m4a"),
)

SFX_IDS: frozenset[str] = frozenset(item.sfx_id for item in SFX_LIBRARY)


def sound_effect(sfx_id: str) -> SoundEffect:
    for item in SFX_LIBRARY:
        if item.sfx_id == sfx_id:
            return item
    raise KeyError(f"unknown sound effect: {sfx_id}")


def sfx_object_key(sfx_id: str) -> str:
    """The S3 key of one official sound (never built from client input other than a known id)."""
    return f"{SFX_S3_PREFIX}{sound_effect(sfx_id).file}"


def _bucket() -> str:
    bucket = load_runtime_config().s3_bucket
    if not bucket:
        raise RuntimeError("S3_BUCKET is required for the sound library")
    return bucket


def _client(client=None):
    if client is not None:
        return client
    import boto3

    return boto3.client("s3", region_name=load_runtime_config().aws_region or "us-east-1")


def sfx_uri(sfx_id: str) -> str:
    return f"s3://{_bucket()}/{sfx_object_key(sfx_id)}"


def _read_manifest(client=None) -> bytes:
    """The stored manifest from the current storage; while the move from Amazon is in progress a
    missing one is read from the old bucket (storage_routing)."""
    from .storage_routing import fallback_client_factory, fallback_storage, is_not_found

    bucket = _bucket()
    try:
        return _client(client).get_object(Bucket=bucket, Key=SFX_MANIFEST_KEY)["Body"].read()
    except Exception as exc:
        fallback = fallback_storage()
        if fallback is None or not is_not_found(exc):
            raise
        return fallback_client_factory(fallback).get_object(Bucket=fallback.bucket, Key=SFX_MANIFEST_KEY)["Body"].read()


def load_library(*, client=None) -> tuple[SoundEffect, ...]:
    """The library with durations from the stored `manifest.json` when present and readable.
    Any problem reading it keeps the built-in list (durations `None`)."""
    try:
        body = _read_manifest(client)
        stored = json.loads(body.decode("utf-8") if isinstance(body, bytes) else body)
        rows = stored.get("sounds") if isinstance(stored, dict) else stored
        durations = {}
        for row in rows or ():
            if isinstance(row, dict) and row.get("id") in SFX_IDS and row.get("duration_sec") is not None:
                value = float(row["duration_sec"])
                if 0.0 < value < 600.0:
                    durations[str(row["id"])] = round(value, 3)
    except RuntimeError:
        raise
    except Exception:
        return SFX_LIBRARY
    return tuple(replace(item, duration_sec=durations.get(item.sfx_id, item.duration_sec)) for item in SFX_LIBRARY)


def signed_manifest(expires_in: int = 3600, *, client=None) -> dict:
    """{"version", "sounds": [{id, name, file, duration_sec, license, url, expires_in}]} with
    presigned GET links. Raises RuntimeError when S3 is not configured."""
    if not 60 <= int(expires_in) <= 3600:
        raise ValueError("sound library link expiry must be between 60 and 3600 seconds")
    bucket = _bucket()
    s3 = _client(client)
    sounds = []
    for item in load_library(client=s3):
        # Routed: a sound not yet on the current storage is signed from the old bucket.
        url = presigned_get(bucket, f"{SFX_S3_PREFIX}{item.file}", expires_in=int(expires_in), client=s3)
        sounds.append({
            "id": item.sfx_id, "name": item.name, "file": item.file, "duration_sec": item.duration_sec,
            "license": item.license, "url": url, "expires_in": int(expires_in),
        })
    return {"version": SFX_LIBRARY_VERSION, "sounds": sounds}
