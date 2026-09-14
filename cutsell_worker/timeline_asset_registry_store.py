"""Timeline Asset Persistence / API Live Wiring -- D-281.

D-279 built `timeline_asset_registry.py`: pure types and pure functions
for asset ownership, the secure resolver, registry-aware timeline
validation, delete-vs-referenced safety, and optimistic-concurrency
save/export. This module wires that CONTRACT into the real backend --
live Redis-backed persistence for `TimelineMediaAsset` records and
`TimelinePersistenceRecord` revisions, real B-roll/VO media
qualification (reusing D-271/D-272/D-274A/D-274B verbatim, never
duplicated), and a real bridge into D-278's composition executor.

    MOBILE / FUTURE CLIENT -> authenticated project -> upload asset
    -> qualify asset -> persist TimelineMediaAsset -> save TimelineComposition
    revision -> reopen project -> resolve asset IDs securely
    -> D-278 Timeline Composition Executor -> final output

## Scope discipline (binding, D-281's own scope banner)

OFFLINE + LOCAL/FAKE-STORAGE INTEGRATION ONLY. No real S3 mutation is
performed by this module's own tests -- every storage-mutating
function takes an injected `persist_media`/`materialize` callable
(mirroring `exports.py`/`uploads.py`'s own established `client=None`
injection convention); `local_directory_persister` is the fake/local
storage implementation this gate actually exercises. No mobile UI, no
microphone capture, no AI B-roll, no D-277/D-278 semantic change (both
are imported and called exactly as they already exist), no BestTake/
P1/P2/Freeze/Boundary/Pacing/Audio-Join/Audio-Finishing/Visual-
Finishing/source-format-policy change of any kind.

## Naming note (avoiding a real, pre-existing collision)

`cutsell_worker/timeline_asset_storage.py` ALREADY EXISTS in this
codebase and is a completely different, unrelated feature (mobile
scrubber-UI filmstrip/waveform presentation assets, D-2xx era, keyed
under `cutsell/timeline-assets/`). This module is deliberately named
`timeline_asset_registry_store` -- the live-storage sibling of D-279's
own `timeline_asset_registry` -- to avoid any confusion with that
unrelated module, which this gate never touches.

## What this gate does NOT build (explicit, not silently skipped)

- FastAPI routes (Stage 40 explicitly permits a service-level contract
  alone for this gate: "Do NOT overbuild controllers if service-level
  contract is enough"). The service functions below ARE that contract;
  a thin router is the next, later, additive step.
- Real S3 upload/download (Stage 36: fake/local storage only).
- `base_edit_identity` -> real render/project media resolution (Stage 23
  says the client must not supply an arbitrary base path, but does not
  ask this gate to rebuild the EXISTING render/project lookup that
  already knows a project's own base edit; `export_timeline_revision`
  below accepts an already-resolved `base_edit_asset` from that
  existing authority as a parameter, rather than re-deriving it here
  and risking a second, competing lookup).
- A real audio presigned-upload allowlist in `uploads.py` (Stage 11's
  own "if not: return exact gap" instruction -- recorded as a gap, not
  invented silently; see `create_voice_over_asset`'s own docstring).
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from . import source_format_policy as sfp
from . import source_media_profile as smp
from . import source_normalization_executor as sne
from . import source_normalization_plan as snp
from . import timeline_asset_registry as reg
from . import timeline_composition as tc
from . import timeline_composition_executor as tce
from .config import load_runtime_config

TIMELINE_ASSET_REGISTRY_STORE_VERSION = 1
MAX_TIMELINE_REVISION_HISTORY = 20


# =============================================================================
# Redis key helpers -- mirrors project_store.py's own `_scope`/`_redis_
# client` convention with a private, per-module copy (this codebase's
# established discipline: `exports.py`/`uploads.py`/`project_store.py`/
# `render_versions.py`/`tenant_safe_delivery.py` each own their own copy
# rather than importing another module's private helper).
# =============================================================================

def _scope(value: str) -> str:
    text = str(value or "").strip()
    if not text or len(text) > 200:
        raise ValueError("identifier must contain 1 to 200 characters")
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def timeline_asset_key(*, user_id: str, project_id: str, asset_id: str) -> str:
    return f"cutsell:v1:timeline_asset:{_scope(user_id)}:{_scope(project_id)}:{_scope(asset_id)}"


def timeline_asset_index_key(*, user_id: str, project_id: str) -> str:
    return f"cutsell:v1:timeline_assets:{_scope(user_id)}:{_scope(project_id)}"


def timeline_revision_key(*, user_id: str, project_id: str) -> str:
    return f"cutsell:v1:timeline_revision:{_scope(user_id)}:{_scope(project_id)}"


def timeline_revision_history_key(*, user_id: str, project_id: str) -> str:
    return f"cutsell:v1:timeline_revision_history:{_scope(user_id)}:{_scope(project_id)}"


def _redis_client(client=None):
    if client is not None:
        return client
    config = load_runtime_config()
    if not config.redis_url:
        raise RuntimeError("REDIS_URL is required for timeline asset persistence")
    from redis import Redis
    return Redis.from_url(config.redis_url)


def _decode_id(raw) -> str:
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


# =============================================================================
# Fake/local storage (Stage 36) -- the ONE storage mutation this gate's
# own tests actually exercise. A real production wiring supplies its
# own `persist_media`/`materialize` callables (e.g. real S3 upload/
# download) through the exact same injection seam.
# =============================================================================

def local_directory_persister(durable_dir: str) -> Callable[[str, str], str]:
    """Stage 34/35/36: copies a local file into `durable_dir` under a
    server-generated key (never a user filename), returning an opaque
    `local://` reference. This is the fake/local storage stand-in for a
    real S3 upload -- no real S3 mutation happens anywhere in this
    module. `durable_dir` must be explicitly supplied by the caller
    (no silent default to a job tempdir) -- this is the concrete,
    testable form of Stage 34's 'no READY asset pointing at /tmp/...'
    invariant."""
    root = Path(durable_dir)
    root.mkdir(parents=True, exist_ok=True)

    def _persist(local_path: str, storage_key: str) -> str:
        source = Path(local_path)
        if not source.exists() or source.stat().st_size <= 0:
            raise ValueError("asset media is missing or empty")
        if ".." in storage_key.split("/") or storage_key.startswith("/"):
            raise ValueError("unsafe storage key")
        destination = root / storage_key
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(source), str(destination))
        return f"local://{destination.resolve().as_posix()}"

    return _persist


def _default_materialize(storage_reference: str) -> str:
    """Stage 20: for this gate's fake/local storage, the durable
    reference IS already a real, durable local path -- 'download' is a
    passthrough. A real S3-backed wiring supplies its own `materialize`
    callable (download-to-job-local-path) through the same seam."""
    if storage_reference.startswith("local://"):
        return storage_reference[len("local://"):]
    raise RuntimeError(f"no materializer available for storage reference scheme: {storage_reference!r}")


def _storage_key(*, user_id: str, project_id: str, asset_id: str, suffix: str) -> str:
    """Stage 35: server-generated, tenant/project-scoped, no user
    filename authority, no traversal -- built entirely from the
    server's own hashed scope + the server-generated asset_id."""
    clean_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    return f"timeline-assets/{_scope(user_id)}/{_scope(project_id)}/{asset_id}{clean_suffix}"


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# =============================================================================
# Serialization -- TimelineMediaAsset <-> dict (Redis JSON payload)
# =============================================================================

def _asset_to_dict(asset: reg.TimelineMediaAsset) -> dict:
    return {
        "asset_id": asset.asset_id,
        "user_id": asset.ownership.user_id,
        "project_id": asset.ownership.project_id,
        "role": asset.role.value,
        "media_kind": asset.media_kind.value,
        "source_media_identity": asset.source_media_identity,
        "storage_reference": asset.storage_reference,
        "duration_sec": asset.duration_sec,
        "has_audio": asset.has_audio,
        "qualification_status": asset.qualification_status.value,
        "content_sha256": asset.content_sha256,
        "technical_metadata_reference": asset.technical_metadata_reference,
        "replaces_asset_id": asset.replaces_asset_id,
        "created_at": asset.created_at,
    }


def _asset_from_dict(data: dict) -> reg.TimelineMediaAsset:
    return reg.TimelineMediaAsset(
        asset_id=data["asset_id"],
        ownership=reg.AssetOwnershipScope(user_id=data["user_id"], project_id=data["project_id"]),
        role=reg.TimelineAssetRole(data["role"]),
        media_kind=reg.TimelineMediaKind(data["media_kind"]),
        source_media_identity=data["source_media_identity"],
        storage_reference=data["storage_reference"],
        duration_sec=float(data["duration_sec"]),
        has_audio=bool(data["has_audio"]),
        qualification_status=reg.TimelineAssetQualificationStatus(data["qualification_status"]),
        content_sha256=data.get("content_sha256"),
        technical_metadata_reference=data.get("technical_metadata_reference"),
        replaces_asset_id=data.get("replaces_asset_id"),
        created_at=data.get("created_at"),
    )


def _save_asset(asset: reg.TimelineMediaAsset, *, client) -> None:
    key = timeline_asset_key(
        user_id=asset.ownership.user_id, project_id=asset.ownership.project_id, asset_id=asset.asset_id,
    )
    client.set(key, json.dumps(_asset_to_dict(asset), ensure_ascii=False))
    index_key = timeline_asset_index_key(user_id=asset.ownership.user_id, project_id=asset.ownership.project_id)
    client.zadd(index_key, {asset.asset_id: datetime.now(timezone.utc).timestamp()})


def get_timeline_asset(
    *, user_id: str, project_id: str, asset_id: str, client=None,
) -> reg.TimelineMediaAsset | None:
    """Stage 2: raw live get -- returns `None` for a missing asset
    (never raises); callers that need an authorization decision should
    go through `resolve_timeline_asset_live` instead."""
    target = _redis_client(client)
    raw = target.get(timeline_asset_key(user_id=user_id, project_id=project_id, asset_id=asset_id))
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return _asset_from_dict(json.loads(raw))


def list_timeline_assets(
    *, user_id: str, project_id: str, client=None, include_deleted: bool = False,
) -> tuple[reg.TimelineMediaAsset, ...]:
    """Stage 2/19: list, scoped by construction to the SAME user+project
    the index key was built from -- reuses D-279's own `list_project_
    assets` for the deleted-filtering policy rather than reimplementing it."""
    target = _redis_client(client)
    ids = target.zrevrange(timeline_asset_index_key(user_id=user_id, project_id=project_id), 0, -1)
    assets = []
    for raw_id in ids:
        asset_id = _decode_id(raw_id)
        asset = get_timeline_asset(user_id=user_id, project_id=project_id, asset_id=asset_id, client=target)
        if asset is not None:
            assets.append(asset)
    return reg.list_project_assets(
        requesting=reg.AssetOwnershipScope(user_id=user_id, project_id=project_id),
        assets=tuple(assets), include_deleted=include_deleted,
    )


# =============================================================================
# STAGE 6/7/8/9 -- live B-roll (VIDEO) asset ingest
# =============================================================================

def create_video_timeline_asset(
    *,
    user_id: str,
    project_id: str,
    role: "reg.TimelineAssetRole",
    local_source_path: str,
    persist_media: Callable[[str, str], str],
    normalization_output_dir: str | None = None,
    client=None,
) -> reg.TimelineMediaAsset:
    """The live B-roll ingest flow: authorized project -> uploaded media
    reference (already downloaded to `local_source_path` by an out-of-
    scope upload-completion step) -> server qualification (D-271 profile
    -> D-272 policy -> D-274A/D-274B normalization when required ->
    re-verification, all REUSED verbatim, never duplicated) ->
    `TimelineMediaAsset`, READY only after every required technical
    check passes -- never on upload receipt alone (Stage 6).

    `role` must be `SUPPLEMENTAL_BROLL` or `PRIMARY_SOURCE` -- never
    `VOICE_OVER` (use `create_voice_over_asset` for that). `persist_
    media` is the injected storage callable (Stage 36/37); this
    function performs no real S3 mutation itself. Stage 3: `asset_id`
    is always server-generated here -- the caller has no way to supply
    one."""
    ownership = reg.AssetOwnershipScope(user_id=user_id, project_id=project_id)
    asset_id = f"tla_{uuid4().hex}"

    profile = smp.probe_source_media_profile(local_source_path)
    decision = sfp.evaluate_source_format_policy(profile)

    resolved_path = local_source_path
    resolved_profile = profile
    status = reg.TimelineAssetQualificationStatus.FAILED

    if decision.decision == sfp.DECISION_REJECT:
        status = reg.TimelineAssetQualificationStatus.REJECTED
    elif decision.decision == sfp.DECISION_INSUFFICIENT_EVIDENCE:
        status = reg.TimelineAssetQualificationStatus.FAILED
    elif decision.decision == sfp.DECISION_ACCEPT:
        status = reg.TimelineAssetQualificationStatus.READY
    elif decision.decision == sfp.DECISION_NORMALIZE_REQUIRED:
        if normalization_output_dir is None:
            status = reg.TimelineAssetQualificationStatus.FAILED
        else:
            plan_result = snp.build_source_normalization_plan(asset_id, profile, decision)
            if plan_result.plan is not None:
                exec_result = sne.execute_source_normalization(
                    local_source_path, plan_result.plan, output_directory=normalization_output_dir,
                )
                if exec_result.outcome == "NORMALIZATION_SUCCEEDED" and exec_result.normalized_path:
                    resolved_path = exec_result.normalized_path
                    resolved_profile = exec_result.normalized_profile or profile
                    status = reg.TimelineAssetQualificationStatus.READY
                else:
                    status = reg.TimelineAssetQualificationStatus.FAILED
            else:
                status = reg.TimelineAssetQualificationStatus.FAILED

    duration_sec = resolved_profile.duration_sec or 0.001
    has_audio = resolved_profile.audio_presence == smp.AUDIO_PRESENT

    content_sha256 = None
    storage_reference = local_source_path
    source_media_identity = f"unqualified:{asset_id}"
    if status == reg.TimelineAssetQualificationStatus.READY:
        content_sha256 = reg.validate_content_sha256_source(_sha256_file(resolved_path), derived_from_etag=False)
        key = _storage_key(
            user_id=user_id, project_id=project_id, asset_id=asset_id,
            suffix=Path(resolved_path).suffix or ".mp4",
        )
        # Stage 9: the ORIGINAL upload reference is never overwritten --
        # `local_source_path` stays whatever it was; only the QUALIFIED
        # (possibly normalized) media is persisted durably and becomes
        # the asset's own `storage_reference`, which the timeline render
        # always resolves.
        storage_reference = persist_media(resolved_path, key)
        source_media_identity = f"sha256:{content_sha256}"

    asset = reg.TimelineMediaAsset(
        asset_id=asset_id, ownership=ownership, role=role, media_kind=reg.TimelineMediaKind.VIDEO,
        source_media_identity=source_media_identity, storage_reference=storage_reference,
        duration_sec=duration_sec, has_audio=has_audio, qualification_status=status,
        content_sha256=content_sha256, created_at=_now(),
    )
    target = _redis_client(client)
    _save_asset(asset, client=target)
    return asset


# =============================================================================
# STAGE 10/11 -- live VOICE_OVER (AUDIO) asset ingest
# =============================================================================

def create_voice_over_asset(
    *,
    user_id: str,
    project_id: str,
    local_source_path: str,
    persist_media: Callable[[str, str], str],
    client=None,
    replaces_asset_id: str | None = None,
) -> reg.TimelineMediaAsset:
    """Ingest a VOICE_OVER asset from an ALREADY-RECORDED/uploaded audio
    file. NO microphone recording happens here or anywhere in this
    module (Stage 45's firewall) -- the future mobile client records
    audio and uploads the completed file; this function only ever
    accepts that completed local file. Validates: audio presence, known
    duration, ownership, supported technical media (Stage 10) -- no
    transcript requirement for READY (D-279's own contract never
    required one).

    Stage 11 audit finding, recorded honestly rather than silently
    worked around: `uploads.py`'s existing presigned-upload allowlist
    (`ALLOWED_VIDEO_EXTENSIONS` = {.mp4,.mov,.m4v,.webm} / `ALLOWED_
    CONTENT_TYPES` = video types only) covers VIDEO containers only --
    there is NO existing audio-specific presigned-upload allowlist
    anywhere in this codebase today. This function itself does not
    depend on that allowlist (it accepts an already-local file, matching
    this gate's own 'offline + local/fake-storage integration only'
    scope), so VO ingest from an already-downloaded file works today.
    But a real mobile VO-RECORDING-upload PRESIGN endpoint (the actual
    direct-to-S3 upload step) needs `uploads.py` extended with an audio
    extension/content-type allowlist before it can safely accept a
    direct-to-S3 iPhone voice recording (typically `.m4a`/AAC) -- this
    is the exact gap Stage 11 asked to be reported, not invented away."""
    ownership = reg.AssetOwnershipScope(user_id=user_id, project_id=project_id)
    asset_id = f"tla_{uuid4().hex}"

    try:
        profile = smp.probe_source_media_profile(local_source_path)
    except Exception:
        profile = None

    if profile is None or profile.audio_presence != smp.AUDIO_PRESENT or not profile.duration_sec:
        status = reg.TimelineAssetQualificationStatus.FAILED
        duration_sec = 0.001
        has_audio = False
        storage_reference = local_source_path
        content_sha256 = None
        source_media_identity = f"unqualified:{asset_id}"
    else:
        status = reg.TimelineAssetQualificationStatus.READY
        duration_sec = float(profile.duration_sec)
        has_audio = True
        content_sha256 = reg.validate_content_sha256_source(_sha256_file(local_source_path), derived_from_etag=False)
        key = _storage_key(
            user_id=user_id, project_id=project_id, asset_id=asset_id,
            suffix=Path(local_source_path).suffix or ".m4a",
        )
        storage_reference = persist_media(local_source_path, key)
        source_media_identity = f"sha256:{content_sha256}"

    asset = reg.TimelineMediaAsset(
        asset_id=asset_id, ownership=ownership, role=reg.TimelineAssetRole.VOICE_OVER,
        media_kind=reg.TimelineMediaKind.AUDIO, source_media_identity=source_media_identity,
        storage_reference=storage_reference, duration_sec=duration_sec, has_audio=has_audio,
        qualification_status=status, content_sha256=content_sha256, created_at=_now(),
        replaces_asset_id=replaces_asset_id,
    )
    target = _redis_client(client)
    _save_asset(asset, client=target)
    return asset


def rerecord_voice_over_asset_live(
    *,
    user_id: str,
    project_id: str,
    previous_asset_id: str,
    local_source_path: str,
    persist_media: Callable[[str, str], str],
    client=None,
) -> reg.TimelineMediaAsset:
    """Stage 33: re-recording always mints a NEW asset_id -- the
    previous asset's own record is never mutated. Uses `create_voice_
    over_asset` for the new record and stamps `replaces_asset_id` for
    lineage (reusing D-279's own `rerecord_voice_over_asset` doctrine
    at the live layer)."""
    target = _redis_client(client)
    previous = get_timeline_asset(user_id=user_id, project_id=project_id, asset_id=previous_asset_id, client=target)
    if previous is None:
        raise KeyError(reg.ASSET_NOT_FOUND)
    return create_voice_over_asset(
        user_id=user_id, project_id=project_id, local_source_path=local_source_path,
        persist_media=persist_media, client=target, replaces_asset_id=previous_asset_id,
    )


# =============================================================================
# STAGE 30/31 -- delete asset (live), delete vs. placement-delete
# =============================================================================

def delete_timeline_asset_live(
    *, user_id: str, project_id: str, asset_id: str, client=None,
) -> reg.AssetDeleteResult:
    """Stage 31: blocks deletion of an asset referenced by the project's
    CURRENT timeline revision (`ASSET_REFERENCED`) -- reuses D-279's own
    `delete_timeline_asset` verbatim; this function only supplies the
    live referencing-compositions list (the project's current revision,
    when one exists)."""
    target = _redis_client(client)
    requesting = reg.AssetOwnershipScope(user_id=user_id, project_id=project_id)
    asset = get_timeline_asset(user_id=user_id, project_id=project_id, asset_id=asset_id, client=target)
    current_record = get_timeline_revision(user_id=user_id, project_id=project_id, client=target)
    referencing = (current_record.composition,) if current_record is not None else ()
    result = reg.delete_timeline_asset(requesting=requesting, asset=asset, referencing_compositions=referencing)
    if result.outcome == reg.ASSET_DELETE_SUCCEEDED and result.asset is not None:
        _save_asset(result.asset, client=target)
    return result


# =============================================================================
# STAGE 14-18 -- timeline revision persistence + optimistic concurrency
# =============================================================================

def _revision_to_dict(record: reg.TimelinePersistenceRecord) -> dict:
    comp = record.composition
    return {
        "contract_version": record.contract_version,
        "user_id": record.ownership.user_id,
        "project_id": record.ownership.project_id,
        "timeline_revision_identity": record.timeline_revision_identity,
        "composition": {
            "contract_version": comp.contract_version,
            "base_edit_identity": comp.base_edit_identity,
            "timeline_duration_sec": comp.timeline_duration_sec,
            "broll_placements": [
                {
                    "placement_id": p.placement_id,
                    "asset_id": p.asset.asset_id,
                    "source_media_identity": p.asset.source_media_identity,
                    "asset_duration_sec": p.asset.duration_sec,
                    "timeline_start_sec": p.timeline_start_sec,
                    "timeline_end_sec": p.timeline_end_sec,
                    "source_in_sec": p.source_in_sec,
                    "source_out_sec": p.source_out_sec,
                    "audio_mode": p.audio_mode.value,
                }
                for p in comp.broll_placements
            ],
            "voice_over_placements": [
                {
                    "placement_id": p.placement_id,
                    "asset_id": p.asset.asset_id,
                    "source_media_identity": p.asset.source_media_identity,
                    "asset_duration_sec": p.asset.duration_sec,
                    "timeline_start_sec": p.timeline_start_sec,
                    "timeline_end_sec": p.timeline_end_sec,
                    "source_in_sec": p.source_in_sec,
                    "source_out_sec": p.source_out_sec,
                    "transcript_reference": p.transcript_reference,
                }
                for p in comp.voice_over_placements
            ],
        },
        "saved_at": _now(),
    }


def _revision_from_dict(data: dict) -> reg.TimelinePersistenceRecord:
    comp_data = data["composition"]
    broll = tuple(
        tc.BrollPlacement(
            placement_id=p["placement_id"],
            asset=tc.TimelineAssetReference(
                asset_id=p["asset_id"], source_media_identity=p["source_media_identity"],
                duration_sec=p["asset_duration_sec"],
            ),
            timeline_start_sec=p["timeline_start_sec"], timeline_end_sec=p["timeline_end_sec"],
            source_in_sec=p["source_in_sec"], source_out_sec=p["source_out_sec"],
            audio_mode=tc.TimelineAudioMode(p["audio_mode"]),
        )
        for p in comp_data["broll_placements"]
    )
    voice_over = tuple(
        tc.VoiceOverPlacement(
            placement_id=p["placement_id"],
            asset=tc.TimelineAssetReference(
                asset_id=p["asset_id"], source_media_identity=p["source_media_identity"],
                duration_sec=p["asset_duration_sec"],
            ),
            timeline_start_sec=p["timeline_start_sec"], timeline_end_sec=p["timeline_end_sec"],
            source_in_sec=p["source_in_sec"], source_out_sec=p["source_out_sec"],
            transcript_reference=p.get("transcript_reference"),
        )
        for p in comp_data["voice_over_placements"]
    )
    composition = tc.TimelineComposition(
        contract_version=comp_data["contract_version"], base_edit_identity=comp_data["base_edit_identity"],
        timeline_duration_sec=comp_data["timeline_duration_sec"],
        broll_placements=broll, voice_over_placements=voice_over,
    )
    return reg.TimelinePersistenceRecord(
        contract_version=data["contract_version"],
        ownership=reg.AssetOwnershipScope(user_id=data["user_id"], project_id=data["project_id"]),
        composition=composition, timeline_revision_identity=data["timeline_revision_identity"],
    )


def get_timeline_revision(*, user_id: str, project_id: str, client=None) -> reg.TimelinePersistenceRecord | None:
    target = _redis_client(client)
    raw = target.get(timeline_revision_key(user_id=user_id, project_id=project_id))
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return _revision_from_dict(json.loads(raw))


def _persist_revision(record: reg.TimelinePersistenceRecord, *, client) -> None:
    payload = json.dumps(_revision_to_dict(record), ensure_ascii=False)
    client.set(
        timeline_revision_key(user_id=record.ownership.user_id, project_id=record.ownership.project_id), payload,
    )
    # Stage 32: append-only, bounded history -- a superseded revision
    # remains resolvable/auditable; the old entry is never mutated,
    # only pushed down the (bounded) list.
    history_key = timeline_revision_history_key(
        user_id=record.ownership.user_id, project_id=record.ownership.project_id,
    )
    raw_history = client.get(history_key)
    if isinstance(raw_history, bytes):
        raw_history = raw_history.decode("utf-8")
    history = json.loads(raw_history) if raw_history else []
    history.insert(0, json.loads(payload))
    client.set(history_key, json.dumps(history[:MAX_TIMELINE_REVISION_HISTORY], ensure_ascii=False))


def list_timeline_revision_history(
    *, user_id: str, project_id: str, client=None,
) -> tuple[dict, ...]:
    target = _redis_client(client)
    raw = target.get(timeline_revision_history_key(user_id=user_id, project_id=project_id))
    if raw is None:
        return ()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return tuple(json.loads(raw))


def save_timeline(
    *,
    user_id: str,
    project_id: str,
    composition: tc.TimelineComposition,
    expected_revision_identity: str | None,
    client=None,
) -> reg.TimelineSaveResult:
    """Stage 15/16/17: the live `save_timeline` service. Resolves/
    validates every referenced asset_id against the LIVE registry
    (`reg.validate_timeline_against_registry`) before ever calling
    D-277's own `validate_composition`/D-279's own optimistic-
    concurrency `save_timeline_revision` -- both reused verbatim, never
    re-derived. A brand-new project timeline (no `current_record` yet)
    never conflicts regardless of `expected_revision_identity` (Stage 17)."""
    target = _redis_client(client)
    ownership = reg.AssetOwnershipScope(user_id=user_id, project_id=project_id)
    current = get_timeline_revision(user_id=user_id, project_id=project_id, client=target)

    assets_by_id = {
        asset.asset_id: asset
        for asset in list_timeline_assets(user_id=user_id, project_id=project_id, client=target, include_deleted=True)
    }
    registry_check = reg.validate_timeline_against_registry(
        requesting=ownership, composition=composition, assets_by_id=assets_by_id,
    )
    if not registry_check.valid:
        return reg.TimelineSaveResult(outcome=reg.TIMELINE_INVALID, reason_codes=registry_check.errors)

    result = reg.save_timeline_revision(
        ownership=ownership, composition=composition, current_record=current,
        expected_revision_identity=expected_revision_identity,
    )
    if result.outcome == reg.TIMELINE_SAVE_SUCCEEDED and result.record is not None:
        _persist_revision(result.record, client=target)
    return result


# =============================================================================
# STAGE 18/19/39 -- reopen timeline, list assets, client-safe responses
# =============================================================================

def client_safe_timeline_view(record: reg.TimelinePersistenceRecord) -> dict:
    """Stage 18/39: no storage secrets -- only asset IDs, never a raw
    path/bucket/key."""
    comp = record.composition
    return {
        "contract_version": comp.contract_version,
        "base_edit_identity": comp.base_edit_identity,
        "timeline_duration_sec": comp.timeline_duration_sec,
        "timeline_revision_identity": record.timeline_revision_identity,
        "broll_placements": [
            {
                "placement_id": p.placement_id, "asset_id": p.asset.asset_id,
                "timeline_start_sec": p.timeline_start_sec, "timeline_end_sec": p.timeline_end_sec,
                "source_in_sec": p.source_in_sec, "source_out_sec": p.source_out_sec,
                "audio_mode": p.audio_mode.value,
            }
            for p in comp.broll_placements
        ],
        "voice_over_placements": [
            {
                "placement_id": p.placement_id, "asset_id": p.asset.asset_id,
                "timeline_start_sec": p.timeline_start_sec, "timeline_end_sec": p.timeline_end_sec,
                "source_in_sec": p.source_in_sec, "source_out_sec": p.source_out_sec,
                "transcript_reference": p.transcript_reference,
            }
            for p in comp.voice_over_placements
        ],
    }


def get_timeline(*, user_id: str, project_id: str, client=None) -> dict | None:
    record = get_timeline_revision(user_id=user_id, project_id=project_id, client=client)
    if record is None:
        return None
    return client_safe_timeline_view(record)


def list_timeline_assets_client_safe(
    *, user_id: str, project_id: str, client=None,
) -> tuple[dict, ...]:
    assets = list_timeline_assets(user_id=user_id, project_id=project_id, client=client)
    return tuple(reg.client_safe_asset_view(asset) for asset in assets)


# =============================================================================
# STAGE 20 -- secure resolver, wired to live persistence
# =============================================================================

def resolve_timeline_asset_live(
    *,
    user_id: str,
    project_id: str,
    asset_id: str,
    required_media_kind: reg.TimelineMediaKind,
    required_audio_mode: "tc.TimelineAudioMode | None" = None,
    client=None,
    materialize: Callable[[str], str] | None = None,
) -> reg.AssetResolutionResult:
    target = _redis_client(client)
    asset = get_timeline_asset(user_id=user_id, project_id=project_id, asset_id=asset_id, client=target)
    local_path = ""
    if asset is not None and asset.storage_reference:
        materializer = materialize or _default_materialize
        try:
            local_path = materializer(asset.storage_reference)
        except Exception:
            local_path = ""
    return reg.resolve_timeline_asset(
        requesting=reg.AssetOwnershipScope(user_id=user_id, project_id=project_id),
        asset=asset, local_path=local_path, required_media_kind=required_media_kind,
        required_audio_mode=required_audio_mode,
    )


# =============================================================================
# STAGE 21/22/23/24 -- export exact revision, D-278 bridge
# =============================================================================

def export_timeline_revision(
    *,
    user_id: str,
    project_id: str,
    revision_identity: str,
    base_edit_asset: tce.ResolvedTimelineAsset,
    output_directory: str,
    client=None,
    materialize: Callable[[str], str] | None = None,
) -> "tce.CompositionExecutionResult | reg.TimelineExportRequestResult | reg.AssetResolutionResult | tce.TimelineRenderPlanResult":
    """Stage 21: resolves the EXACT persisted revision named by
    `revision_identity` -- never 'whatever is currently latest' -- via
    D-279's own `resolve_timeline_export`. Stage 22/23: bridges every
    referenced asset_id into a real `ResolvedTimelineAsset` via `resolve_
    timeline_asset_live`, then calls D-278's own, completely unmodified
    `build_timeline_render_plan`/`execute_timeline_composition` --
    composition semantics are never rebuilt here. `base_edit_asset` is
    accepted as a parameter: resolving `base_edit_identity` into real
    render/project media is an EXISTING authority this gate does not
    re-derive (see module docstring)."""
    target = _redis_client(client)
    ownership = reg.AssetOwnershipScope(user_id=user_id, project_id=project_id)
    record = get_timeline_revision(user_id=user_id, project_id=project_id, client=target)
    export_check = reg.resolve_timeline_export(
        requesting=ownership, record=record, expected_revision_identity=revision_identity,
    )
    if export_check.outcome != reg.TIMELINE_EXPORT_RESOLVED or export_check.composition is None:
        return export_check
    composition = export_check.composition

    resolved_broll: dict[str, tce.ResolvedTimelineAsset] = {}
    for placement in composition.broll_placements:
        result = resolve_timeline_asset_live(
            user_id=user_id, project_id=project_id, asset_id=placement.asset.asset_id,
            required_media_kind=reg.TimelineMediaKind.VIDEO, required_audio_mode=placement.audio_mode,
            client=target, materialize=materialize,
        )
        if result.outcome != reg.ASSET_RESOLVED or result.resolved is None:
            return result
        resolved_broll[placement.asset.asset_id] = result.resolved

    resolved_vo: dict[str, tce.ResolvedTimelineAsset] = {}
    for placement in composition.voice_over_placements:
        result = resolve_timeline_asset_live(
            user_id=user_id, project_id=project_id, asset_id=placement.asset.asset_id,
            required_media_kind=reg.TimelineMediaKind.AUDIO, client=target, materialize=materialize,
        )
        if result.outcome != reg.ASSET_RESOLVED or result.resolved is None:
            return result
        resolved_vo[placement.asset.asset_id] = result.resolved

    plan_result = tce.build_timeline_render_plan(
        composition, base_edit_asset=base_edit_asset,
        resolved_broll_assets=resolved_broll, resolved_voice_over_assets=resolved_vo,
    )
    if plan_result.outcome != tce.PLAN_BUILD_SUCCEEDED or plan_result.plan is None:
        return plan_result
    return tce.execute_timeline_composition(plan_result.plan, output_directory=output_directory)


# =============================================================================
# STAGE 41 -- bounded API/service error mapping
# =============================================================================

SERVICE_ERROR_STATUS: dict[str, int] = {
    reg.ASSET_NOT_FOUND: 404,
    reg.ASSET_NOT_OWNED: 403,
    reg.ASSET_NOT_READY: 409,
    reg.ASSET_MEDIA_UNSUPPORTED: 422,
    reg.ASSET_HAS_NO_AUDIO: 422,
    reg.TIMELINE_REVISION_CONFLICT: 409,
    reg.TIMELINE_INVALID: 422,
    reg.ASSET_REFERENCED: 409,
}


def map_outcome_to_http_status(outcome: str) -> int:
    """Stage 41: bounded mapping -- an unrecognized outcome maps to 500
    rather than silently defaulting to 200/OK, so a future new D-279
    outcome value can never accidentally read as success."""
    return SERVICE_ERROR_STATUS.get(outcome, 500)


# =============================================================================
# STAGE 42 -- project/account deletion integration
# =============================================================================

def delete_project_timeline_assets(*, user_id: str, project_id: str, client=None) -> dict[str, Any]:
    """Additive project-deletion participation: removes every timeline-
    asset Redis record plus the project's timeline-revision pointer and
    history. Called from `account_lifecycle.delete_project_data`
    (imported there additively) -- never duplicates that module's own
    S3/durable-record cleanup authority. Real durable media bytes
    (Stage 34's `local://`/future S3 objects) are NOT deleted by this
    function -- that is a real-storage-backend concern outside this
    gate's fake/local-storage scope (Stage 43: explicit orphan
    responsibility, not silently swept)."""
    target = _redis_client(client)
    index_key = timeline_asset_index_key(user_id=user_id, project_id=project_id)
    ids = target.zrevrange(index_key, 0, -1)
    deleted = 0
    for raw_id in ids:
        asset_id = _decode_id(raw_id)
        target.delete(timeline_asset_key(user_id=user_id, project_id=project_id, asset_id=asset_id))
        deleted += 1
    target.delete(index_key)
    target.delete(timeline_revision_key(user_id=user_id, project_id=project_id))
    target.delete(timeline_revision_history_key(user_id=user_id, project_id=project_id))
    return {"timeline_assets_deleted": deleted}
