"""Mobile Timeline API Bridge -- Upload-to-Ingest Seam (D-282).

D-281 built the live persistence/qualification service (`timeline_asset_
registry_store.py`), which ingests media from an already-local file path.
D-282 exposes that service through a real, tenant-safe mobile API -- this
module is the ONE seam that turns "a client already uploaded object X"
into "a local file D-281's own ingest functions can qualify," so that
`cutsell_app/timeline_routes.py` never has to know about storage at all.

    client presigned-upload (uploads.py, additive VO allowlist)
        -> client PUTs bytes directly to storage (never through this API)
        -> client calls the ingest route with `source_uri`
        -> THIS MODULE fetches/materializes that object to a local path
        -> D-281's own create_video_timeline_asset/create_voice_over_asset
           (completely unmodified) qualifies + persists it

## Scope discipline (binding, D-282's own scope banner)

No D-281 function is modified -- this module only calls them. No real S3
mutation in this gate's own tests (Stage 31/36): `_default_fetch_uploaded_
media` handles ONLY the `local://` fake-storage scheme this session's own
D-281 gate established; a real production wiring supplies its own
`fetch_media` callable (a real `boto3` `download_file` to a job-local
temp path -- the same pattern `exports.py`/`uploads.py` already use
elsewhere) through the exact same injection seam, never hardcoded here.
"""
from __future__ import annotations

from typing import Callable

from . import timeline_asset_registry as reg
from . import timeline_asset_registry_store as store


def _default_fetch_uploaded_media(source_uri: str) -> str:
    """For this gate's fake/local-storage integration, an already-
    'uploaded' object is referenced by a `local://` URI -- the exact
    same fake-storage scheme D-281's own `_default_materialize`
    established. Fetching it is a passthrough. A real production
    wiring supplies its own callable that downloads a real `s3://`
    object to a local temp path through this same seam."""
    if source_uri.startswith("local://"):
        return source_uri[len("local://"):]
    raise RuntimeError(f"no fetcher available for source_uri scheme: {source_uri!r}")


def ingest_broll_asset_from_upload(
    *,
    user_id: str,
    project_id: str,
    role: reg.TimelineAssetRole,
    source_uri: str,
    persist_media: Callable[[str, str], str],
    normalization_output_dir: str | None = None,
    fetch_media: Callable[[str], str] | None = None,
    client=None,
) -> reg.TimelineMediaAsset:
    """Stage 5: bridges an already-uploaded VIDEO object into D-281's own
    `create_video_timeline_asset` -- never reimplements qualification."""
    fetcher = fetch_media or _default_fetch_uploaded_media
    local_path = fetcher(source_uri)
    return store.create_video_timeline_asset(
        user_id=user_id, project_id=project_id, role=role, local_source_path=local_path,
        persist_media=persist_media, normalization_output_dir=normalization_output_dir, client=client,
    )


def ingest_voice_over_asset_from_upload(
    *,
    user_id: str,
    project_id: str,
    source_uri: str,
    persist_media: Callable[[str, str], str],
    fetch_media: Callable[[str], str] | None = None,
    client=None,
    replaces_asset_id: str | None = None,
) -> reg.TimelineMediaAsset:
    """Stage 6/10: bridges an already-uploaded AUDIO object into D-281's
    own `create_voice_over_asset` -- never reimplements qualification."""
    fetcher = fetch_media or _default_fetch_uploaded_media
    local_path = fetcher(source_uri)
    return store.create_voice_over_asset(
        user_id=user_id, project_id=project_id, local_source_path=local_path,
        persist_media=persist_media, client=client, replaces_asset_id=replaces_asset_id,
    )
