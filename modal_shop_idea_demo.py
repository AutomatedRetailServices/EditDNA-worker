"""Shop board "My idea" on a REAL Sonnet call (PO 18 Nov 2026: "ok"). PAID, on request only, ~US$0.02 (one Sonnet look).

A test board is drawn here from product photos already in the PRIVATE bucket (title, brand line, tilted products on a
soft background) and sent through the same read_idea() the live endpoint uses. Then a copy is drawn from the map that
comes back, with the same photos in the slots, so the PO can compare picture vs copy side by side. The sheet and the
map are saved to the PRIVATE bucket only; this public log prints counts and the cost, nothing else.
Run (CI only): `modal run modal_shop_idea_demo.py`
"""
from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("fonts-dejavu-core")
    .pip_install_from_requirements("requirements.cutsell.api.txt")
    .add_local_file("cutsell_app/shop_idea_routes.py", remote_path="/root/shop_idea_routes.py")
)
app = modal.App("cutsell-shop-idea-demo")
IN_PREFIX = "private-qa/shape-compare/20261112/inputs/"
OUT_PREFIX = "private-qa/shop-idea/20261118"
W, H = 1080, 1920
SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
SANS_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _gradient(top, bottom):
    from PIL import Image

    img = Image.new("RGB", (W, H))
    px = img.load()
    for y in range(H):
        t = y / (H - 1)
        c = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
        for x in range(W):
            px[x, y] = c
    return img


def _paste(board, photo, box, turn):
    from PIL import Image, ImageOps

    x, y, w, h = (int(v) for v in box)
    piece = ImageOps.contain(photo.convert("RGBA"), (max(1, w), max(1, h)))
    if turn:
        piece = piece.rotate(-turn, expand=True, resample=Image.BICUBIC)
    board.paste(piece, (x + (w - piece.width) // 2, y + (h - piece.height) // 2), piece)


def _rgb(hex_):
    return tuple(int(hex_.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))


def _reference(photos):
    """The board she 'liked': what a TikTok shop board looks like."""
    from PIL import ImageDraw, ImageFont

    board = _gradient((246, 238, 226), (226, 210, 190))
    d = ImageDraw.Draw(board)
    d.text((W // 2, 170), "Fall Edit", font=ImageFont.truetype(SERIF, 130), fill=(58, 40, 30), anchor="mm")
    d.text((W // 2, 290), "MY COZY FAVORITES", font=ImageFont.truetype(SANS, 46), fill=(120, 92, 70), anchor="mm")
    places = [((90, 420, 560, 700), -6), ((560, 470, 440, 560), 5), ((110, 1130, 420, 470), 4), ((560, 1080, 430, 520), -4)]
    for photo, (box, turn) in zip(photos, places):
        _paste(board, photo, box, turn)
    d.text((W // 2, 1760), "@shop.with.me", font=ImageFont.truetype(SANS_B, 44), fill=(58, 40, 30), anchor="mm")
    return board


def _copy(style, photos):
    """Her board drawn from the map (the iPhone does the same with her own products)."""
    from PIL import ImageDraw, ImageFont

    board = _gradient(_rgb(style["background"]["top"]), _rgb(style["background"]["bottom"]))
    slots = sorted(style["slots"], key=lambda s: -s["front"])          # back to front
    for n, s in enumerate(slots):
        _paste(board, photos[n % len(photos)], (s["x"] * W, s["y"] * H, s["w"] * W, s["h"] * H), s["rotation"])
    d = ImageDraw.Draw(board)
    for t in style["texts"]:
        path = SERIF if t["font"] in ("serif", "script", "display", "handwritten") else (SANS_B if t["weight"] in ("bold", "black") else SANS)
        text = t["text"].upper() if t["uppercase"] else t["text"]
        size = max(14, int(t["h"] * H * 0.8))
        font = ImageFont.truetype(path, size)
        while size > 14 and d.textlength(text, font=font) > t["w"] * W * 1.05:
            size -= 2
            font = ImageFont.truetype(path, size)
        x = {"left": t["x"] * W, "center": (t["x"] + t["w"] / 2) * W, "right": (t["x"] + t["w"]) * W}[t["align"]]
        anchor = {"left": "lm", "center": "mm", "right": "rm"}[t["align"]]
        d.text((x, (t["y"] + t["h"] / 2) * H), text, font=font, fill=_rgb(t["color"]), anchor=anchor)
    return board


@app.function(image=image, secrets=[modal.Secret.from_name("cutsell-worker")], timeout=600, cpu=1.0, memory=1024)
def demo(raw_photos: list[bytes]) -> dict:
    import base64
    import io
    import sys

    from PIL import Image, ImageDraw, ImageFont

    sys.path.insert(0, "/root")
    from shop_idea_routes import read_idea

    photos = [Image.open(io.BytesIO(b)) for b in raw_photos]
    reference = _reference(photos)
    buf = io.BytesIO()
    reference.save(buf, "JPEG", quality=85)
    started = __import__("time").time()
    result = read_idea(base64.b64encode(buf.getvalue()).decode())
    seconds = round(__import__("time").time() - started, 1)
    if result["style"] is None:
        return {"error": result["error"], "cost_usd": result["cost_usd"], "seconds": seconds}
    copy = _copy(result["style"], photos)
    sheet = Image.new("RGB", (W * 2 + 60, H + 140), (255, 255, 255))
    sheet.paste(reference, (20, 120))
    sheet.paste(copy, (W + 40, 120))
    d = ImageDraw.Draw(sheet)
    f = ImageFont.truetype(SANS_B, 52)
    d.text((20 + W // 2, 60), "Her screenshot", font=f, fill=(11, 16, 32), anchor="mm")
    d.text((W + 40 + W // 2, 60), "Copy from the AI map", font=f, fill=(23, 105, 255), anchor="mm")
    out = io.BytesIO()
    sheet.save(out, "JPEG", quality=85)
    return {"sheet": out.getvalue(), "style": result["style"], "cost_usd": result["cost_usd"], "seconds": seconds}


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
    keys = sorted(o["Key"] for o in s3.list_objects_v2(Bucket=bucket, Prefix=IN_PREFIX).get("Contents", [])
                  if o["Key"].lower().endswith((".jpg", ".jpeg", ".png", ".webp")))[:4]
    if not keys:
        raise SystemExit("RESULT failed: no product photos found")
    out = demo.remote([s3.get_object(Bucket=bucket, Key=k)["Body"].read() for k in keys])
    if out.get("error"):
        raise SystemExit(f"RESULT failed: {out['error']} (cost ~US${out.get('cost_usd')}, {out.get('seconds')} s)")
    s3.put_object(Bucket=bucket, Key=f"{OUT_PREFIX}/idea-sheet.jpg", Body=out.pop("sheet"), ContentType="image/jpeg")
    s3.put_object(Bucket=bucket, Key=f"{OUT_PREFIX}/idea-map.json", Body=json.dumps(out, ensure_ascii=False, indent=1),
                  ContentType="application/json")
    style = out["style"]
    print(f"RESULT ok: {len(keys)} photos in; map has {len(style['slots'])} slots {[s['kind'] for s in style['slots']]}, "
          f"{len(style['texts'])} texts {[t['role'] + '/' + t['font'] for t in style['texts']]}; "
          f"{out['seconds']} s; cost ~US${out['cost_usd']}")
