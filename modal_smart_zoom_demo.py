"""Smart zoom on a REAL video (PO 16 Nov 2026: "ok"). PAID, on request only, ~US$0.05 (one Sonnet cut, no new ASR).

One video of the private bucket (Yaskira/04: "Y04" in the Gold set, whose words are already transcribed) goes through
the real engine on this branch -- the cut, the word fix and the AI's zoom moments (kind, move, body part) -- on Modal
(the Anthropic key stays in the Modal secret). The result is rendered with captions and a small label on each zoom
(what she says -> why it zooms), and saved to the PRIVATE bucket only. Nothing she says is printed in this public log.
Run (CI only): `modal run modal_smart_zoom_demo.py`
"""
from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "fonts-dejavu-core")
    .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
    .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_file("benchmarks/simple_engine_gold/gold_v1.json", remote_path="/root/gold_v1.json")
)
app = modal.App("cutsell-smart-zoom-demo")
VIDEO_KEY, GOLD_ID = "Yaskira/04.MP4", "Y04"
OUT_PREFIX = "private-qa/smart-zoom/20261116"
PRICES = {"in": 3.00, "write": 3.75, "read": 0.30, "out": 15.00}        # Sonnet 5.5, US$ per million
LABELS = {"hook": "gancho", "product": "producto", "price": "precio", "cta": "compra", "look": "mira esto",
          "benefit": "beneficio", "closeup": "de cerca", "point": "punto"}
MOVES = {"fast": "rapido", "slow": "lento", "reveal": "abre", "hold": "entra y sale en el siguiente punto"}


@app.function(image=image, secrets=[modal.Secret.from_name("cutsell-worker")], timeout=1200, cpu=2.0, memory=2048)
def demo(video: bytes) -> dict:
    import json
    import os
    import subprocess
    import tempfile
    from pathlib import Path

    os.environ["CUTSELL_SIMPLE_ENGINE_REMEMBER"] = "0"
    os.environ["REDIS_URL"] = ""                    # never touch production counters
    from cutsell_worker import render
    from cutsell_worker.broll_overlay import source_time_to_timeline
    from cutsell_worker.contracts import ProcessingRequest, SourceAsset
    from cutsell_worker.render_plan import build_render_plan
    from types import SimpleNamespace

    from cutsell_worker import simple_engine_adapter
    from cutsell_worker.simple_engine_adapter import process_with_simple_engine

    # A test video, not a customer's: no minutes are counted or checked (production's meter is never touched).
    simple_engine_adapter.check_processing_allowance = lambda **k: SimpleNamespace(allowed=True, reason="demo")

    gold = next(v for v in json.load(open("/root/gold_v1.json"))["videos"] if v["id"] == GOLD_ID)
    work = Path(tempfile.mkdtemp())
    src = work / "v.mp4"
    src.write_bytes(video)
    request = ProcessingRequest(project_id="demo", user_id="demo-smart-zoom", auto_zoom_enabled=True, language_hint="es",
                                sources=(SourceAsset(source_asset_id="s1", project_id="demo", user_id="demo-smart-zoom",
                                                     original_name="v.mp4", source_order=0,
                                                     duration_sec=float(gold["duration"]), uri=str(src)),))
    heard = {"language": "es", "duration": float(gold["duration"]), "words": gold["words"]}
    result = process_with_simple_engine(request, {"s1": str(src)}, transcribe=lambda *a, **k: heard)
    draft = result.draft
    diag = draft.diagnostics["simple_engine"]
    usage = diag.get("usage") or [u for s in diag.get("sources", []) for u in (s.get("usage") or [])]
    cost = sum((int(u.get("input_tokens") or 0) * PRICES["in"] + int(u.get("cache_creation_input_tokens") or 0) * PRICES["write"]
                + int(u.get("cache_read_input_tokens") or 0) * PRICES["read"] + int(u.get("output_tokens") or 0) * PRICES["out"]) / 1e6
               for u in usage if isinstance(u, dict))
    zooms = []
    log = [row for s in diag.get("sources", []) for row in (s.get("zoom_moments") or [])]
    for k in draft.zoom_keyframes:
        at = source_time_to_timeline(draft.selected, k.source_asset_id, k.source_time)
        row = next((r for r in log if r.get("zoom") == "added" and abs(float(r["s"]) - k.source_time) < 1.2), {})
        zooms.append({"t": None if at is None else round(at, 3), "d": k.duration, "kind": k.kind, "style": k.style,
                      "scale": k.scale, "x": k.x, "y": k.y, "word": row.get("word", ""), "target": row.get("target", "")})
    plan = build_render_plan(draft, {"s1": str(src)})
    plain = work / "plain.mp4"
    try:
        render.render_preview(plan, str(plain), width=464, height=832)       # the production renderer
    except Exception as exc:                                                 # noqa: BLE001
        return {"error": f"render: {exc.__class__.__name__}: {str(exc)[-300:]}", "cost_usd": round(cost, 4)}
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    tags = []
    for z in zooms:
        if z["t"] is None:
            continue
        text = f"ZOOM {LABELS.get(z['kind'], z['kind'])} · {MOVES.get(z['style'], z['style'])}"
        text += f" · {z['word']}" if z["word"] else ""
        text = "".join(ch for ch in text if ch not in ":'\\%,;[]")
        tags.append(f"drawtext=fontfile={font}:text='{text}':x=(w-tw)/2:y=24:fontsize=20:fontcolor=white:"
                    f"box=1:boxcolor=0x1769FF@0.85:boxborderw=8:enable='between(t,{z['t']:.2f},{z['t'] + z['d']:.2f})'")
    final = work / "final.mp4"
    done = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(plain), "-vf", ",".join(tags) or "null",
                           "-c:v", "libx264", "-crf", "22", "-preset", "veryfast", "-c:a", "copy",
                           "-movflags", "+faststart", str(final)], capture_output=True, text=True)
    if done.returncode != 0:
        return {"error": "labels: " + done.stderr[-300:], "cost_usd": round(cost, 4)}
    return {"mp4": final.read_bytes(), "zooms": zooms, "cost_usd": round(cost, 4),
            "cut_sec": round(sum(c.end - c.start for c in draft.selected), 1), "kept": len(draft.selected)}


def _bucket_env() -> dict:
    """The worker template's bucket keys (RunPod; values masked in the log)."""
    import os

    import requests

    response = requests.post("https://api.runpod.io/graphql", headers={"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"},
                             json={"query": "query { myself { podTemplates { name env { key value } } } }"}, timeout=(15, 60))
    templates = response.json()["data"]["myself"]["podTemplates"]
    env = {str(e["key"]): str(e["value"]) for t in templates if t.get("name") == "EditDNA-Worker-2" for e in t.get("env") or []}
    for name, value in env.items():
        if any(p in name.upper() for p in ("KEY", "TOKEN", "SECRET", "PASSWORD", "BUCKET")) or len(value) >= 12:
            print(f"::add-mask::{value}")
    return env


@app.local_entrypoint()
def main():
    import json

    import boto3

    env = _bucket_env()
    s3 = boto3.client("s3", region_name=env.get("AWS_REGION", "us-east-1"), aws_access_key_id=env["AWS_ACCESS_KEY_ID"],
                      aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"], aws_session_token=env.get("AWS_SESSION_TOKEN") or None)
    bucket = env["S3_BUCKET"]
    out = demo.remote(s3.get_object(Bucket=bucket, Key=VIDEO_KEY)["Body"].read())
    if out.get("error"):
        raise SystemExit(f"RESULT failed: {out['error'][:200]} (cost ~US${out.get('cost_usd')})")
    s3.put_object(Bucket=bucket, Key=f"{OUT_PREFIX}/y04-smart-zoom.mp4", Body=out.pop("mp4"), ContentType="video/mp4")
    s3.put_object(Bucket=bucket, Key=f"{OUT_PREFIX}/y04-zooms.json", Body=json.dumps(out, ensure_ascii=False, indent=1),
                  ContentType="application/json")
    kinds = {}
    for z in out["zooms"]:
        kinds[f"{z['kind']}/{z['style']}/{z['target']}"] = kinds.get(f"{z['kind']}/{z['style']}/{z['target']}", 0) + 1
    print(f"RESULT ok: {len(out['zooms'])} zooms {kinds}; cut {out['cut_sec']}s; cost ~US${out['cost_usd']}")
