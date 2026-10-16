"""Editor v5 "Clean up voice" (build 17): ONE cleaned copy of a sound, made once on the worker and
used by BOTH the iPhone preview (it downloads and plays the file) and the export (it renders
with the same file), so what the creator hears is what the video gets.

Cleaning = background-noise reduction + even loudness:
    highpass 80 Hz -> afftdn (adaptive broadband denoise) -> level to about -20 LUFS (measured)
    -> agate (pushes the noise left between words down) -> acompressor (evens the voice)
    -> loudnorm -16 LUFS / -1.5 dBTP -> 48 kHz stereo, padded/trimmed to the input's length

The copy keeps the input's timeline (same length, no shift), so a clip plays the same window of
the cleaned file as of the original.

What gets cleaned:
    target "source": the sound of one video of the cut (whole file) -- the whole video's Clean up
                     and each video clip's own Clean up both play windows of this one copy;
    target "asset":  a voice-over / music recording (timeline asset).

Stored next to the project's uploads (deleted with the project / account):
    {upload_prefix}{user16}/{project16}/clean/v1/{target}-{sha16(id)}.m4a
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from .media_probe import probe_media

CLEAN_VERSION = "v1"
TARGETS = ("source", "asset")
#: Longest sound cleaned (a longer source is refused, never half-cleaned).
MAX_CLEAN_SECONDS = 20 * 60
PRE_LEVEL_LUFS = -20.0
GATE = "agate=threshold=0.02:ratio=4:range=0.1:attack=5:release=150"
COMPRESSOR = "acompressor=threshold=0.125:ratio=3:attack=10:release=200:makeup=1.5"
FINAL = "loudnorm=I=-16:TP=-1.5:LRA=7"
DENOISE = ("highpass=f=80", "afftdn=nr=24:nf=-30:tn=1")
_FORMAT = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"


def _scope16(value: str) -> str:
    return hashlib.sha256(str(value).encode()).hexdigest()[:16]


def clean_key(*, user_id: str, project_id: str, target: str, ident: str) -> str:
    """Inside the project's own upload prefix (uploads.scoped_upload_prefix), so deleting the
    project or the account deletes it too."""
    from .uploads import scoped_upload_prefix

    if target not in TARGETS:
        raise ValueError("clean target must be source or asset")
    if not ident or len(ident) > 200:
        raise ValueError("clean id is required")
    return (f"{scoped_upload_prefix(user_id=user_id, project_id=project_id)}clean/{CLEAN_VERSION}/"
            f"{target}-{_scope16(ident)}.m4a")


def _measure_lufs(path: str, chain: str) -> float | None:
    """Integrated loudness (LUFS) of `path` after `chain`; None when silent / unmeasurable."""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vn", "-af",
         f"{chain},loudnorm=I=-16:TP=-1.5:LRA=7:print_format=json", "-f", "null", "-"],
        capture_output=True, text=True, timeout=600,
    )
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", result.stderr, re.S)
    if not match:
        return None
    try:
        value = float(json.loads(match.group(0))["input_i"])
    except (KeyError, ValueError, TypeError):
        return None
    return value if value > -70 else None


def clean_filter_chain(measured_lufs: float | None) -> str:
    gain_db = 0.0 if measured_lufs is None else max(-20.0, min(30.0, PRE_LEVEL_LUFS - measured_lufs))
    return ",".join([*DENOISE, f"volume={gain_db:.2f}dB", GATE, COMPRESSOR, FINAL, _FORMAT])


def make_clean_audio(input_path: str, output_path: str) -> dict:
    """Write the cleaned copy of `input_path`'s sound to `output_path` (AAC .m4a, same length).
    Returns facts for the job record."""
    probe = probe_media(str(input_path))
    if not probe.has_audio:
        raise ValueError("clean_audio_no_sound")
    duration = float(probe.duration_sec or 0.0)
    if duration <= 0.0:
        raise ValueError("clean_audio_empty")
    if duration > MAX_CLEAN_SECONDS:
        raise ValueError("clean_audio_too_long")
    measured = _measure_lufs(str(input_path), ",".join(DENOISE))
    chain = clean_filter_chain(measured) + f",apad=whole_dur={duration:.6f},atrim=duration={duration:.6f}"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(input_path), "-vn", "-af", chain,
         "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2", "-movflags", "+faststart", str(output_path)],
        check=True, capture_output=True, timeout=1200,
    )
    return {"duration_sec": round(duration, 3), "measured_lufs": measured}


# --- Status record (Redis) and the worker job ----------------------------------------------------

STATUS_TTL_SEC = 7 * 24 * 3600


def status_key(*, user_id: str, project_id: str, target: str, ident: str) -> str:
    return (f"cutsell:v1:clean:{hashlib.sha256(user_id.encode()).hexdigest()[:20]}:"
            f"{_scope16(project_id)}:{target}:{_scope16(ident)}")


def read_status(client, *, user_id: str, project_id: str, target: str, ident: str) -> dict | None:
    raw = client.get(status_key(user_id=user_id, project_id=project_id, target=target, ident=ident))
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def write_status(client, *, user_id: str, project_id: str, target: str, ident: str, **fields) -> dict:
    record = dict(read_status(client, user_id=user_id, project_id=project_id, target=target, ident=ident) or {})
    record.update(fields)
    import time
    record["updated_at"] = time.time()
    key = status_key(user_id=user_id, project_id=project_id, target=target, ident=ident)
    client.set(key, json.dumps(record))
    if hasattr(client, "expire"):
        client.expire(key, STATUS_TTL_SEC)
    return record


def source_uri(snapshot: dict, source_asset_id: str) -> str | None:
    """The stored video of one source of the cut (its originals, or a B-roll video the creator
    added). None when the project has no such source."""
    rows = list(snapshot.get("sources") or ())
    draft = snapshot.get("draft") or {}
    rows += [row for row in draft.get("broll_sources") or () if isinstance(row, dict)]
    for row in rows:
        if isinstance(row, dict) and str(row.get("source_asset_id") or "") == source_asset_id and row.get("uri"):
            return str(row["uri"])
    return None


def fetch_input(*, user_id: str, project_id: str, target: str, ident: str, directory: str,
                snapshot: dict | None = None) -> str:
    """Download what is cleaned to `directory`, through the same tenant checks as the export."""
    from .storage import download_source
    from .uploads import validate_product_source_uri

    if target == "source":
        if snapshot is None:
            from .draft_store import get_draft_snapshot
            snapshot = get_draft_snapshot(user_id=user_id, project_id=project_id)
        uri = source_uri(snapshot, ident)
        if not uri:
            raise ValueError("clean_audio_unknown_source")
        validate_product_source_uri(uri, project_id=project_id, user_id=user_id)
        suffix = Path(uri).suffix or ".mp4"
        return download_source(uri, str(Path(directory) / f"clean-input{suffix}"))
    from . import timeline_asset_registry as registry
    from . import timeline_asset_registry_store as store
    from . import timeline_asset_s3

    result = store.resolve_timeline_asset_live(
        user_id=user_id, project_id=project_id, asset_id=ident,
        required_media_kind=registry.TimelineMediaKind.AUDIO,
        materialize=timeline_asset_s3.s3_fetcher(user_id=user_id, project_id=project_id,
                                                 scratch_dir=str(Path(directory) / "asset")),
    )
    if result.outcome != registry.ASSET_RESOLVED or result.resolved is None:
        raise ValueError(f"clean_audio_unknown_asset:{result.outcome}")
    return str(result.resolved.local_path)


def ensure_clean_audio(*, user_id: str, project_id: str, target: str, ident: str, directory: str,
                       local_input: str | None = None, progress=None, snapshot: dict | None = None) -> str:
    """The cleaned copy as a local file in `directory`: downloaded when it is already stored, else
    made (from `local_input` when the caller already has the file) and stored. One algorithm for
    the preview and the export."""
    from .config import load_runtime_config
    from .storage import download_source
    from . import storage_routing

    bucket = load_runtime_config().s3_bucket
    if not bucket:
        raise RuntimeError("S3_BUCKET is not configured")
    key = clean_key(user_id=user_id, project_id=project_id, target=target, ident=ident)
    Path(directory).mkdir(parents=True, exist_ok=True)
    local = str(Path(directory) / f"clean-{target}-{_scope16(ident)}.m4a")
    try:
        storage_routing.head(bucket, key)
        return download_source(f"s3://{bucket}/{key}", local)
    except Exception as exc:
        if not storage_routing.is_not_found(exc):
            raise
    report = progress or (lambda _percent: None)
    report(10)
    source = local_input or fetch_input(user_id=user_id, project_id=project_id, target=target, ident=ident,
                                        directory=directory, snapshot=snapshot)
    report(35)
    make_clean_audio(source, local)
    report(85)
    storage_routing.default_client().upload_file(local, bucket, key, ExtraArgs={"ContentType": "audio/mp4"})
    report(100)
    return local


def run_clean_audio_job(payload: dict) -> dict:
    """RQ job (Modal worker): clean one sound and store it. Progress and the outcome go to the
    status record the API reads."""
    import tempfile

    from .config import load_runtime_config

    user_id = str(payload["user_id"])
    project_id = str(payload["project_id"])
    target = str(payload["target"])
    ident = str(payload["id"])
    from redis import Redis
    redis = Redis.from_url(load_runtime_config().redis_url)

    def progress(percent: int) -> None:
        write_status(redis, user_id=user_id, project_id=project_id, target=target, ident=ident,
                     state="cleaning", progress=int(percent))

    progress(5)
    try:
        with tempfile.TemporaryDirectory(prefix="cutsell-clean-") as directory:
            ensure_clean_audio(user_id=user_id, project_id=project_id, target=target, ident=ident,
                               directory=directory, progress=progress)
    except Exception as exc:
        write_status(redis, user_id=user_id, project_id=project_id, target=target, ident=ident,
                     state="failed", error=str(exc)[:200] or exc.__class__.__name__)
        raise
    write_status(redis, user_id=user_id, project_id=project_id, target=target, ident=ident,
                 state="ready", progress=100)
    return {"state": "ready", "target": target}
