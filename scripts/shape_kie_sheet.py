#!/usr/bin/env python3
"""Free (no paid calls): one side-by-side review sheet of the Shape it test photos -- original | Google direct
(12 Nov) | kie.ai (16 Nov) -- saved to the PRIVATE bucket only (private-qa/shape-compare/20261112/sheet.jpg)."""
from __future__ import annotations

import io
import sys

import boto3
from PIL import Image, ImageDraw

sys.path.insert(0, "scripts")
from shape_it_compare_ci import template_env  # noqa: E402

ITEMS = ("shirt", "pants", "shoes", "sweater")
CELL = 360

env = template_env()
s3 = boto3.client("s3", region_name=env.get("AWS_REGION", "us-east-1"), aws_access_key_id=env["AWS_ACCESS_KEY_ID"],
                  aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"], aws_session_token=env.get("AWS_SESSION_TOKEN") or None)
bucket, prefix = env["S3_BUCKET"], "private-qa/shape-compare/20261112"


def cell(key: str) -> Image.Image:
    img = Image.open(io.BytesIO(s3.get_object(Bucket=bucket, Key=key)["Body"].read())).convert("RGBA")
    img.thumbnail((CELL - 20, CELL - 20))
    tile = Image.new("RGB", (CELL, CELL), (245, 246, 250))
    tile.paste(img, ((CELL - img.width) // 2, (CELL - img.height) // 2), img)
    return tile


cols = [("Original", "in/{}.png"), ("Google directo · Nano Banana 2 Lite", "out/{}__nano-banana.png"),
        ("kie · Nano Banana (anterior)", "out-kie/{}-kie.png")]
for tag, title in (("nano-banana-2", "kie · Nano Banana 2"), ("nano-banana-pro", "kie · Nano Banana Pro"),
                   ("bytedance_seedream-v4-edit", "kie · Seedream 4"), ("qwen_image-edit", "kie · Qwen Edit")):
    cols.append((title, "out-kie-models/" + tag + "/{}.png"))


def exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


cols = [c for c in cols if any(exists(f"{prefix}/{c[1].format(i)}") for i in ITEMS)]
sheet = Image.new("RGB", (CELL * len(cols), 40 + CELL * len(ITEMS)), (255, 255, 255))
draw = ImageDraw.Draw(sheet)
for c, (title, _) in enumerate(cols):
    draw.text((c * CELL + 12, 12), title, fill=(11, 16, 32))
for r, item in enumerate(ITEMS):
    for c, (_, pattern) in enumerate(cols):
        key = f"{prefix}/{pattern.format(item)}"
        if exists(key):
            sheet.paste(cell(key), (c * CELL, 40 + r * CELL))
        else:
            draw.text((c * CELL + 140, 40 + r * CELL + CELL // 2), "(no probado)", fill=(107, 115, 133))
buf = io.BytesIO()
sheet.save(buf, "JPEG", quality=85)
s3.put_object(Bucket=bucket, Key=f"{prefix}/sheet.jpg", Body=buf.getvalue(), ContentType="image/jpeg")
print("sheet saved to the private bucket")
