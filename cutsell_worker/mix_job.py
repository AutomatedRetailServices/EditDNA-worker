"""The Mix job: ONE job per Mix (all its clips together), resumable.

1. Clean every clip ONCE: Deepgram (its voice file when the app sent the voice first) and one mix_clip call for all
   the clips that still need it; then the cut engine's finishing per clip (no AI).
2. When every clip's video is on storage: a clean clip per clip (ffmpeg: common frame, 30 fps, -16 LUFS, 30 ms fades),
   kept on storage so a retry never makes it again.
3. Each combination = the clean clips joined in order (no AI); its own project with a normal draft (one clip per part,
   captions and Auto Zoom moved onto its timeline), shown in Edits as soon as it is made.

A clip that fails is marked with the reason; the videos that need it wait, the others are made. POST
/v1/mix/{id}/retry runs the job again for that clip only (everything already done is reused). When the videos are
still uploading the job stops ("waiting_for_videos") and /v1/projects/{mix}/video-ready starts it again.
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

from . import mix_engine as engine
from . import mix_store as store

_LOG = logging.getLogger("cutsell.mix_job")
DEEPGRAM_USD_PER_MIN = 0.0043          # Nova-3 pre-recorded, pay as you go (Deepgram price list)
VIDEO_WAIT_LIMIT_SEC = 6 * 3600        # a clip whose video never arrives fails after this
COMBO_WORKERS = 3                      # videos made side by side (a join is light; the strip + draft are I/O)


def units_per_video() -> float:
    """Plan accounting: how much one video of a Mix counts (MIX_UNITS_PER_VIDEO, default 1). Recorded, not charged."""
    try:
        return float(os.environ.get("MIX_UNITS_PER_VIDEO", "1") or 1)
    except ValueError:
        return 1.0


class Deps:
    """Everything that reaches outside (injectable in tests)."""

    def __init__(self, *, transcribe=None, llm=None, fetch=None, put=None, extract_audio=None, client=None,
                 after_combo=None):
        from .simple_engine import asr
        self.transcribe = transcribe or asr.transcribe_words
        self.extract_audio = extract_audio or asr.extract_audio
        self.llm = llm
        self.fetch = fetch or _fetch
        self.put = put or _put
        self.client = client
        self.after_combo = after_combo if after_combo is not None else _after_combo


def _fetch(uri: str, destination: str) -> str:
    from .storage import download_source
    return download_source(uri, destination)


def _put(local_path: str, key: str, content_type: str) -> str:
    from .config import load_runtime_config
    from .timeline_asset_s3 import _client
    bucket = load_runtime_config().s3_bucket
    if not bucket:
        raise RuntimeError("S3_BUCKET is required")
    _client().upload_file(str(local_path), bucket, key, ExtraArgs={"ContentType": content_type})
    return f"s3://{bucket}/{key}"


def _prefix(user_id: str, project_id: str) -> str:
    from .uploads import scoped_upload_prefix
    return scoped_upload_prefix(user_id=user_id, project_id=project_id)


def all_clips(record: dict) -> list[tuple[int, int, dict]]:
    return [(si, ci, clip) for si, section in enumerate(record["sections"]) for ci, clip in enumerate(section["clips"])]


def run_mix_job(payload: dict, deps: Optional[Deps] = None) -> dict:
    user_id, mix_id = str(payload.get("user_id") or ""), str(payload.get("mix_id") or "")
    if not user_id or not mix_id:
        raise ValueError("user_id and mix_id are required")
    deps = deps or Deps()
    if not store.lock(user_id, mix_id, client=deps.client):
        # another run of this Mix is working: it runs once more when it ends (a retry / a video that just arrived)
        store.update(user_id, mix_id, lambda d: d.update(rerun=True), client=deps.client)
        return {"state": "busy"}
    try:
        result = _run(user_id, mix_id, deps)
    finally:
        store.unlock(user_id, mix_id, client=deps.client)
        try:
            from .usage_limits import release_processing_slot
            if payload.get("slot_reserved"):
                release_processing_slot(user_id=user_id)
        except Exception:
            pass
    # a video that arrived / a retry asked while this run was working: run again instead of waiting for a new signal
    rerun = store.load(user_id, mix_id, client=deps.client).get("rerun")
    if rerun:
        store.update(user_id, mix_id, lambda d: d.pop("rerun", None), client=deps.client)
    if rerun or (result.get("state") == "waiting_for_videos" and result.get("recheck")):
        return run_mix_job({"user_id": user_id, "mix_id": mix_id}, deps)
    return result


def _set(user_id, mix_id, deps, change):
    return store.update(user_id, mix_id, change, client=deps.client)


def _run(user_id: str, mix_id: str, deps: Deps) -> dict:
    started = time.time()
    record = store.load(user_id, mix_id, client=deps.client)
    _set(user_id, mix_id, deps, lambda d: (d.setdefault("timing", {}).setdefault("started_at", started),
                                          d.update(state="processing")))
    with tempfile.TemporaryDirectory(prefix="cutsell-mix-") as directory:
        work = Path(directory)
        record = _clean_phase(user_id, mix_id, record, deps, work)
        videos, missing = _fetch_videos(record, deps, work)
        if missing:
            age = time.time() - float((record.get("timing") or {}).get("started_at") or started)
            if age > VIDEO_WAIT_LIMIT_SEC:
                def fail_missing(d):
                    for cid in missing:
                        d["clips"].setdefault(cid, {}).update(state="failed", error="video_never_arrived")
                record = _set(user_id, mix_id, deps, fail_missing)
            else:
                _set(user_id, mix_id, deps, lambda d: d.update(state="waiting_for_videos"))
                still = _fetch_videos(store.load(user_id, mix_id, client=deps.client), deps, work, probe_only=True)[1]
                return {"state": "waiting_for_videos", "missing": sorted(missing), "recheck": len(still) < len(missing)}
        record = _render_phase(user_id, mix_id, record, deps, work, videos)
        record = _combo_phase(user_id, mix_id, record, deps, work)
    return store.summary(record)


# ---------------------------------------------------------------- 1. clean every clip once

def _clean_phase(user_id, mix_id, record, deps, work: Path) -> dict:
    todo = [(si, ci, clip) for si, ci, clip in all_clips(record)
            if (record["clips"].get(clip["clip_id"]) or {}).get("state") not in ("clean", "rendered")]
    if not todo:
        return record

    def hear(item):
        si, ci, clip = item
        cid = clip["clip_id"]
        try:
            uri = clip.get("audio_uri") or clip["uri"]
            local = deps.fetch(uri, str(work / f"{cid}-voice{Path(uri).suffix or '.m4a'}"))
            audio = deps.extract_audio(local, str(work / f"{cid}.mp3"))
            heard = deps.transcribe(audio)
            from .media_probe import probe_media
            duration = float(probe_media(local).duration_sec)
            words = [w for w in heard.get("words", []) if float(w["e"]) > float(w["s"])]
            return cid, {"audio": audio, "duration": duration, "words": words, "language": heard.get("language")}
        except Exception as exc:
            _LOG.warning("mix_clip_transcribe_failed %s", exc.__class__.__name__)
            return cid, {"error": "transcription_failed:" + exc.__class__.__name__}

    with ThreadPoolExecutor(max_workers=6) as pool:
        heard = dict(pool.map(hear, todo))
    # A short clip heard alone can come back empty or in the wrong language (a 6 s English clip came back as Dutch
    # in production): every clip not heard in the Mix's language is heard again in it. The Mix's language is the one
    # most of its words were heard in (a creator records one Mix in one language).
    weight = {}
    for h in heard.values():
        if h.get("words") and h.get("language") in ("en", "es"):
            weight[h["language"]] = weight.get(h["language"], 0) + len(h["words"])
    if weight:
        main = max(weight, key=weight.get)
        for si, ci, clip in todo:
            h = heard[clip["clip_id"]]
            if not h.get("error") and (not h.get("words") or h.get("language") != main) and h.get("duration", 0) > 1.0:
                try:
                    again = deps.transcribe(h["audio"], language_hint=main)
                    h["words"] = [w for w in again.get("words", []) if float(w["e"]) > float(w["s"])]
                    h["language"] = again.get("language") or main
                    h["heard_twice"] = True
                except Exception:
                    pass
    speaking = [(si, ci, clip) for si, ci, clip in todo if heard[clip["clip_id"]].get("words")]
    answer, units, usage, ai_error = {"clips": [], "fix": [], "zoom": []}, [], {}, None
    if speaking:
        try:
            answer, units, usage = engine.ask(
                [{"section": record["sections"][si]["name"], "words": heard[clip["clip_id"]]["words"]}
                 for si, ci, clip in speaking], call=deps.llm)
        except Exception as exc:
            ai_error = "cleaning_failed:" + exc.__class__.__name__
    results = {}
    for si, ci, clip in todo:
        cid, h = clip["clip_id"], heard[clip["clip_id"]]
        label = f"{record['sections'][si]['code']}{ci + 1}"
        if h.get("error"):
            results[cid] = {"state": "failed", "error": h["error"], "section": record["sections"][si]["name"], "label": label}
            continue
        try:
            words, duration = h["words"], h["duration"]
            if not words:                              # no speech (a product shot): the whole clip, no captions
                parts, caps, moments, fixes = [(0.0, duration)], [], [], []
            elif ai_error:
                results[cid] = {"state": "failed", "error": ai_error, "section": record["sections"][si]["name"],
                                "label": label}
                continue
            else:
                n = next(i for i, item in enumerate(speaking) if item[2]["clip_id"] == cid)
                lab = engine.clip_labels(answer, n, words, units[n])
                splits = engine.finish_clip(words, lab, duration, h["audio"])
                parts = engine.kept_parts(splits) or [(0.0, duration)]
                kept = {k for k, w in enumerate(words) if any(a <= (w["s"] + w["e"]) / 2 <= b for a, b in parts)}
                fixes, marks = engine.fixes_and_moments(answer, n, words, units[n], kept)
                shown = [dict(w) for w in words]
                for k, _old, new in fixes:
                    shown[k]["w"] = new
                caps = engine.clip_captions(shown, parts)
                moments = [{"t": engine.clip_time(words[m["i"]]["s"], parts), "kind": m["kind"]} for m in marks]
                moments = [m for m in moments if m["t"] is not None]
            results[cid] = {"state": "clean", "section": record["sections"][si]["name"], "label": label,
                            "duration": round(duration, 3), "parts": [[round(a, 3), round(b, 3)] for a, b in parts],
                            "captions": caps, "moments": moments, "fixes": fixes, "language": h.get("language"),
                            "words": len(words), "kept_sec": round(sum(b - a for a, b in parts), 3)}
        except Exception as exc:
            results[cid] = {"state": "failed", "error": "finishing_failed:" + exc.__class__.__name__,
                            "section": record["sections"][si]["name"], "label": label}
    deepgram_sec = sum(h.get("duration", 0.0) for h in heard.values())

    def save(d):
        d["clips"].update(results)
        u = d.setdefault("usage", {})
        u["deepgram_sec"] = round(float(u.get("deepgram_sec", 0)) + deepgram_sec, 2)
        u["deepgram_usd"] = round(u["deepgram_sec"] / 60 * DEEPGRAM_USD_PER_MIN, 5)
        if usage:
            llm = u.setdefault("llm", {"model": usage["model"], "input_tokens": 0, "output_tokens": 0, "calls": 0,
                                       "cost_usd": 0.0})
            for k in ("input_tokens", "output_tokens", "calls"):
                llm[k] = int(llm.get(k, 0)) + int(usage.get(k, 0))
            llm["cost_usd"] = round(float(llm.get("cost_usd", 0)) + float(usage.get("cost_usd", 0)), 6)
        d.setdefault("timing", {})["cleaned_at"] = time.time()
    return _set(user_id, mix_id, deps, save)


# ---------------------------------------------------------------- 2. clean clips (videos needed)

def _fetch_videos(record, deps, work: Path, probe_only: bool = False):
    videos, missing = {}, set()
    for si, ci, clip in all_clips(record):
        cid = clip["clip_id"]
        state = (record["clips"].get(cid) or {}).get("state")
        if state != "clean" or (record["clips"][cid].get("clean_uri")):
            continue                                   # failed, or its clean clip already exists
        target = work / f"{cid}-video{Path(clip['uri']).suffix or '.mp4'}"
        if target.exists():
            videos[cid] = str(target)
            continue
        try:
            videos[cid] = deps.fetch(clip["uri"], str(target))
        except Exception:
            missing.add(cid)
    return videos, missing


def _render_phase(user_id, mix_id, record, deps, work: Path, videos: dict) -> dict:
    if not videos:
        return record
    canvas = tuple(record.get("canvas") or ())
    if not canvas:
        shapes = []
        for path in videos.values():
            try:
                shapes.append(engine.probe(path))
            except Exception:
                continue
        canvas = engine.canvas_for(shapes) if shapes else engine.CANVAS_PORTRAIT
    prefix = _prefix(user_id, mix_id)

    def render(item):
        cid, path = item
        info = record["clips"][cid]
        try:
            shape = engine.probe(path)
            out = engine.render_clean_clip(path, [tuple(p) for p in info["parts"]], canvas,
                                           str(work / f"clean-{cid}.mov"), has_audio=shape["has_audio"])
            uri = deps.put(out["path"], f"{prefix}mix/clean-{cid}.mov", "video/quicktime")
            return cid, {"state": "rendered", "clean_uri": uri, "clean_duration": round(out["duration"], 3),
                         "clean_path": out["path"],
                         "lufs_in": (out.get("loudness_in") or {}).get("input_i")}
        except Exception as exc:
            _LOG.warning("mix_clip_render_failed %s", exc.__class__.__name__)
            return cid, {"state": "failed", "error": "render_failed:" + exc.__class__.__name__}

    with ThreadPoolExecutor(max_workers=2) as pool:
        rendered = dict(pool.map(render, videos.items()))
    local = {cid: r.pop("clean_path") for cid, r in rendered.items() if r.get("clean_path")}

    def save(d):
        d["canvas"] = list(canvas)
        for cid, r in rendered.items():
            d["clips"][cid].update(r)
        d.setdefault("timing", {})["rendered_at"] = time.time()
    record = _set(user_id, mix_id, deps, save)
    record["_local_clean"] = local
    return record


# ---------------------------------------------------------------- 3. the combinations

def _clean_clip_path(record, cid, deps, work: Path) -> str:
    local = (record.get("_local_clean") or {}).get(cid)
    if local and Path(local).exists():
        return local
    target = work / f"clean-{cid}.mov"
    if not target.exists():
        deps.fetch(record["clips"][cid]["clean_uri"], str(target))
    return str(target)


def _combo_phase(user_id, mix_id, record, deps, work: Path) -> dict:
    local_clean = record.get("_local_clean") or {}
    record["_local_clean"] = local_clean
    buildable = []
    for combo in record["combos"]:
        if combo.get("state") == "ready":
            continue
        cids = [record["sections"][si]["clips"][p]["clip_id"] for si, p in enumerate(combo["picks"])]
        infos = [record["clips"].get(cid) or {} for cid in cids]
        if any(i.get("state") == "failed" for i in infos):
            n = combo["n"]
            record = _set(user_id, mix_id, deps, lambda d: _combo_state(d, n, "waiting_clip"))
            record["_local_clean"] = local_clean
            continue
        if all(i.get("state") == "rendered" for i in infos):
            buildable.append((combo, cids, infos))
    # every clean clip needed is on this machine before the videos are made side by side
    paths = {}
    for _, cids, _ in buildable:
        for cid in cids:
            if cid not in paths:
                paths[cid] = _clean_clip_path(record, cid, deps, work)

    def build(item):
        combo, cids, infos = item
        n = combo["n"]
        try:
            out = engine.join_clips([paths[cid] for cid in cids], str(work / f"combo-{n:03d}.mp4"))
            timeline = engine.combo_timeline([{"duration": i["clean_duration"], "captions": i.get("captions") or [],
                                               "moments": i.get("moments") or [], "section": i.get("section")}
                                              for i in infos])
            name = f"mix-{n:03d}-{combo['code'].replace(' · ', '-')}.mp4"
            uri = deps.put(out["path"], f"{_prefix(user_id, combo['project_id'])}{name}", "video/mp4")
            deps.after_combo(user_id=user_id, record=record, combo=combo, uri=uri, name=name, local=out["path"],
                             duration=out["duration"], timeline=timeline, infos=infos, directory=str(work))
            duration = round(out["duration"], 3)

            def ready(d):
                _combo_state(d, n, "ready", duration_sec=duration)
                t = d.setdefault("timing", {})
                t.setdefault("first_ready_at", time.time())
                t["last_ready_at"] = time.time()
            _set(user_id, mix_id, deps, ready)
            Path(out["path"]).unlink(missing_ok=True)
        except Exception as exc:
            _LOG.warning("mix_combo_failed %s", exc.__class__.__name__)
            error = "combo_failed:" + exc.__class__.__name__
            _set(user_id, mix_id, deps, lambda d: _combo_state(d, n, "failed", error=error))

    with ThreadPoolExecutor(max_workers=COMBO_WORKERS) as pool:
        list(pool.map(build, buildable))
    record = store.load(user_id, mix_id, client=deps.client)
    combos = record["combos"]
    done = all(c.get("state") == "ready" for c in combos)
    failed_clips = any(c.get("state") == "failed" for c in record["clips"].values())

    def finish(d):
        d["state"] = "ready" if done else ("needs_retry" if failed_clips else d.get("state", "processing"))
        u = d.setdefault("usage", {})
        u["videos"] = sum(1 for c in d["combos"] if c.get("state") == "ready")
        u["units_per_video"] = units_per_video()
        u["units"] = u["videos"] * u["units_per_video"]
    record = _set(user_id, mix_id, deps, finish)
    if done and not record.get("announced"):
        _announce(user_id, record, deps)
        _captions(user_id, record, deps)
    return record


def _captions(user_id, record, deps):
    """Plan my week: each video of the Mix is a Mix video, with its posting caption written once (Haiku)."""
    if deps.after_combo is not _after_combo:
        return                                      # tests / probes that do not store real projects
    from .post_captions import caption_once
    from .post_plan import record_source

    def one(combo):
        try:
            record_source(user_id, combo["project_id"], "mix")
            caption_once(user_id=user_id, project_id=combo["project_id"])
        except Exception:
            pass
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one, record["combos"]))


def _combo_state(d, n, state, **extra):
    for c in d["combos"]:
        if c["n"] == n:
            c["state"] = state
            c.update(extra)


def _announce(user_id, record, deps):
    try:
        from .notifications import publish_notification
        first = record["combos"][0]["project_id"]
        publish_notification(user_id=user_id, project_id=first, kind="draft_ready",
                             payload={"mix_id": record["mix_id"], "videos": len(record["combos"])},
                             idempotency_key="mix-" + record["mix_id"])
    except Exception:
        pass
    try:
        from .usage_limits import record_processing_minutes
        clips = record.get("clips") or {}
        record_processing_minutes(user_id=user_id, project_id=record["mix_id"],
                                  minutes=sum(float(c.get("duration") or 0) for c in clips.values()) / 60.0,
                                  metadata={"mix_id": record["mix_id"], "videos": len(record["combos"]),
                                            "units": (record.get("usage") or {}).get("units")})
    except Exception:
        pass
    store.update(user_id, record["mix_id"], lambda d: d.update(announced=True), client=deps.client)


# ---------------------------------------------------------------- one combination becomes a normal project

def _after_combo(*, user_id, record, combo, uri, name, local, duration, timeline, infos, directory) -> None:
    """The combination's project: a normal one-source draft (one clip per part, captions and Auto Zoom on its
    timeline), its strip and waveform, its product in Edits · Products with the MIX tag and code."""
    from .contracts import SCHEMA_VERSION, DraftClip, DraftTimeline, EditStrategy, JobState, ProcessingResult, Word
    from .draft_store import create_initial_draft
    from .project_tracking import safe_update_project
    from .serde import result_to_dict
    from .simple_engine_adapter import _zoom_keyframes
    from .source_identity import stable_source_id

    project_id = combo["project_id"]
    source_id = stable_source_id(project_id, 0, name)
    selected = []
    for k, ((a, b), info) in enumerate(zip(timeline["parts"], infos)):
        words = [c for c in timeline["captions"] if a - 0.01 <= (c["s"] + c["e"]) / 2 <= b + 0.01]
        text = " ".join(w["w"] for w in words)
        selected.append(DraftClip(
            clip_id=f"mx-{combo['n']:03d}-{k + 1:02d}", source_asset_id=source_id, source_order=0,
            start=float(a), end=float(min(b, duration)), text=text, caption_text=text,
            words=tuple(Word(w["w"], float(w["s"]), float(w["e"]), None) for w in words), selected=True))
    source = SimpleNamespace(source_order=0, source_asset_id=source_id)
    zooms = _zoom_keyframes(source, timeline["moments"],
                            [{"start": a, "end": b, "visible": True} for a, b in timeline["parts"]])
    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id=project_id, strategy=EditStrategy.MIXED, selected=tuple(selected),
        alternates=(), discarded=(), auto_zoom_enabled=True, zoom_keyframes=tuple(zooms),
        diagnostics={"engine": "simple", "engine_version": engine.MIX_VERSION,
                     "mix": {"mix_id": record["mix_id"], "n": combo["n"], "code": combo["code"], "picks": combo["picks"]},
                     "simple_engine": {"model": engine.model_name(), "asr": "deepgram-nova-3",
                                       "cut_duration_sec": round(duration, 3), "captions": timeline["captions"],
                                       "caption_groups": timeline["caption_groups"],
                                       "sources": [{"source_asset_id": source_id, "source_order": 0,
                                                    "duration_sec": round(duration, 3),
                                                    "cut_duration_sec": round(duration, 3),
                                                    "splits": [{"start": 0.0, "end": round(duration, 3),
                                                                "visible": True, "reason": "mix"}],
                                                    "log": {}, "usage": []}]}})
    serialized = result_to_dict(ProcessingResult(schema_version=SCHEMA_VERSION, project_id=project_id,
                                                 state=JobState.DRAFT_READY, draft=draft, stage_status={}))
    assets = {"status": "not_generated"}
    try:
        from .media_probe import probe_media
        from .timeline_asset_storage import store_timeline_assets
        from .timeline_assets import generate_filmstrip, waveform_peaks
        filmstrip = generate_filmstrip(local, str(Path(directory) / f"strip-{combo['n']}"),
                                       duration_sec=probe_media(local).duration_sec, max_frames=24, width=160)
        assets = store_timeline_assets(user_id=user_id, project_id=project_id, source_asset_id=source_id,
                                       filmstrip=filmstrip, waveform=waveform_peaks(local, buckets=256))
    except Exception as exc:
        assets = {"status": "degraded", "reason": exc.__class__.__name__}
    sources = [{"source_asset_id": source_id, "original_name": name, "source_order": 0,
                "duration_sec": round(duration, 3), "uri": uri, "timeline_assets": assets}]
    create_initial_draft(user_id=user_id, project_id=project_id, draft=dict(serialized["draft"]), sources=sources)
    try:
        from .timeline_asset_s3 import register_primary_video
        register_primary_video(user_id=user_id, project_id=project_id, source_uri=uri, original_local_path=local,
                               cut_local_path=local, scratch_dir=directory)
    except Exception:
        pass
    safe_update_project(user_id=user_id, project_id=project_id, state="draft_ready", sources=sources)
