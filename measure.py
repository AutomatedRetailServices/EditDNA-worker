"""Cost measurement on REAL stored videos, on the worker's own machine size (2 cores, 4 GiB): download, Deepgram,
the cut (Claude), and the final export render -- each timed. Nothing is written to Redis or storage: the export is
rendered to a temporary file only (its size is measured) and the cut is not saved. Report encrypted."""
import json
import os
import time

import modal

app = modal.App("cutsell-cost-measure")
secret = modal.Secret.from_name("cutsell-worker")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
    .env({"CUTSELL_ENGINE": "simple", "CUTSELL_EXPORT_DELIVERY_GATE": "technical_qc"})
    .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
)


REAL_USER = "1625a6e9"   # hashed id of the account that uploads from the real iPhone (no synthetic clips)


def _real(d):
    return d and abs(float(d) - round(float(d))) > 0.01      # synthetic test clips have whole-second lengths


@app.function(image=image, secrets=[secret], timeout=600)
def pick():
    """One stored draft per case, single-source, real (not a synthetic test clip)."""
    from redis import Redis
    r = Redis.from_url(os.environ["REDIS_URL"])
    rows = []
    for key in r.scan_iter("cutsell:v1:draft:*", count=500):
        try:
            snap = json.loads(r.get(key))
        except Exception:
            continue
        srcs = snap.get("sources") or []
        if len(srcs) != 1 or not srcs[0].get("uri"):
            continue
        d = srcs[0].get("duration_sec")
        if not _real(d):
            continue
        import hashlib
        if hashlib.sha256(str(snap.get("user_id")).encode()).hexdigest()[:8] != REAL_USER:
            continue                                   # only the creator's own iPhone uploads, no test clips
        try:
            from cutsell_worker.storage import parse_s3_uri
            from cutsell_worker.storage_routing import head
            head(*parse_s3_uri(srcs[0]["uri"]))
        except Exception:
            continue
        rows.append({"key": key.decode(), "duration": float(d), "uri": srcs[0]["uri"]})
    seen, uniq = set(), []
    for row in sorted(rows, key=lambda x: x["duration"]):
        if row["uri"] not in seen:
            seen.add(row["uri"]); uniq.append(row)
    one = min(uniq, key=lambda x: abs(x["duration"] - 60))
    three = max(uniq, key=lambda x: x["duration"])
    rest = [x for x in uniq if x["key"] not in (one["key"], three["key"]) and 15 <= x["duration"] <= 90]
    from urllib.parse import urlparse
    host = urlparse(os.environ["REDIS_URL"]).hostname or ""
    return {"one": one, "three": three, "bulk": rest[:5], "pool": [round(x["duration"], 1) for x in uniq],
            "redis_provider": ".".join(host.split(".")[-2:])}


@app.function(image=image, secrets=[secret], timeout=600, cpu=2.0, memory=4096, retries=1)
def run_case(key: str, bulk: bool):
    import tempfile
    from pathlib import Path
    from redis import Redis
    from cutsell_worker.storage import download_source, parse_s3_uri
    from cutsell_worker.storage_routing import head
    from cutsell_worker.media_probe import probe_media
    from cutsell_worker.simple_engine import asr, process, llm
    from cutsell_worker.serde import draft_from_dict
    from cutsell_worker.render_plan import build_render_plan
    from cutsell_worker.live_render_qc import render_with_post_render_qc
    from cutsell_worker.export_job import resolve_audio_mix
    from cutsell_worker.export_job import LocalMediaOverlay  # noqa: F401  (overlays are counted, not rendered)
    container_start = time.time()
    r = Redis.from_url(os.environ["REDIS_URL"])
    raw = None
    for _ in range(5):
        try:
            raw = r.get(key); break
        except Exception:
            time.sleep(3)
    snap = json.loads(raw)
    src = snap["sources"][0]
    out = {"bulk": bulk, "timings": {}, "sizes": {}}
    t = time.time()
    with tempfile.TemporaryDirectory() as d:
        path = download_source(src["uri"], str(Path(d) / ("src" + (Path(src["uri"]).suffix or ".mp4"))))
        out["timings"]["download"] = round(time.time() - t, 2)
        out["sizes"]["original_mb"] = round(os.path.getsize(path) / 1e6, 2)
        meta = src.get("metadata") or {}
        if meta.get("audio_uri"):
            try:
                b, k = parse_s3_uri(meta["audio_uri"]); out["sizes"]["voice_mb"] = round(int(head(b, k).get("ContentLength") or 0) / 1e6, 3)
            except Exception as exc:
                out["sizes"]["voice_mb"] = "n/a " + exc.__class__.__name__
        probe = probe_media(path)
        duration = float(probe.duration_sec)
        out["duration_sec"] = round(duration, 2)
        out["source"] = {"width": probe.width, "height": probe.height, "fps": probe.fps}
        t = time.time(); audio = asr.extract_audio(path, str(Path(d) / "a.mp3")); out["timings"]["extract_audio"] = round(time.time() - t, 2)
        t = time.time(); heard = asr.transcribe_words(audio); out["timings"]["deepgram"] = round(time.time() - t, 2)
        out["deepgram_audio_sec"] = heard.get("duration")
        words = [w for w in heard.get("words", []) if float(w["e"]) > float(w["s"])]
        t = time.time()
        with llm.cache_scope(bulk):
            cut = process(words, duration, path, video_path=path)
        out["timings"]["cut_engine"] = round(time.time() - t, 2)
        out["usage"] = [{k: u.get(k) for k in ("step", "input_tokens", "output_tokens", "cache_creation_input_tokens",
                                                "cache_read_input_tokens", "cache", "json_format")} for u in cut["usage"]]
        out["cut_duration_sec"] = cut["cut_duration"]
        out["kept_parts"] = sum(1 for s in cut["splits"] if s["visible"])
        # Export: the creator's own stored cut, rendered exactly as the export job renders it (to a temp file).
        draft = draft_from_dict(dict(snap["draft"]))
        clean_wanted = bool(getattr(draft, "video_clean_voice", False)) or any(getattr(c, "clean_voice", False) for c in draft.selected) \
            or any(getattr(c, "clean", False) for c in getattr(draft, "audio_clips", ()) or ())
        out["export_notes"] = {"clean_voice_skipped": clean_wanted, "media_overlays_skipped": len(draft.media_overlays),
                               "text_overlays": len(draft.text_overlays), "selected": len(draft.selected)}
        mix = None if clean_wanted else resolve_audio_mix(draft, user_id=snap["user_id"], project_id=snap["project_id"], directory=d)
        from cutsell_worker.render_plan import bound_draft_to_sources
        draft = bound_draft_to_sources(draft, {src["source_asset_id"]: path})
        plan = build_render_plan(draft, {src["source_asset_id"]: path})
        export = str(Path(d) / "export.mp4")
        t = time.time()
        qc = render_with_post_render_qc(draft, plan, export, text_overlays=draft.text_overlays,
                                        **({"audio_mix": mix} if mix is not None else {}))
        out["timings"]["export_render_qc"] = round(time.time() - t, 2)
        out["export_qc"] = qc.status
        out["sizes"]["export_mb"] = round(os.path.getsize(export) / 1e6, 2) if os.path.exists(export) else None
        out["export_duration_sec"] = round(float(probe_media(export).duration_sec), 2) if os.path.exists(export) else None
    out["timings"]["container_total"] = round(time.time() - container_start, 2)
    return out


@app.local_entrypoint()
def main():
    picks = pick.remote()
    cases = [("one", picks["one"]["key"], False), ("three", picks["three"]["key"], False)]
    cases += [(f"bulk{i + 1}", p["key"], True) for i, p in enumerate(picks["bulk"])]
    t0 = time.time()
    results = []
    for x in run_case.starmap([(k, b) for _, k, b in cases], return_exceptions=True):
        results.append(x if isinstance(x, dict) else {"error": str(x)[-600:]})
    rep = {"cases": [{"case": c[0], **x} for c, x in zip(cases, results)],
           "wall_sec": round(time.time() - t0, 1), "pool": picks["pool"], "redis_provider": picks["redis_provider"]}
    with open("engine_check.json", "w") as f:
        json.dump(rep, f, ensure_ascii=False, default=str)
    print(f"::notice title=measure::{len(cases)} cases, errors {sum(1 for x in results if "error" in x)}")
