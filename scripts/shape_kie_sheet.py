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


cols = ("in/{}.png", "out/{}__nano-banana.png", "out-kie/{}-kie.png")
sheet = Image.new("RGB", (CELL * 3, 40 + CELL * len(ITEMS)), (255, 255, 255))
draw = ImageDraw.Draw(sheet)
for c, title in enumerate(("Original", "Google direct (12 nov)", "kie.ai (16 nov)")):
    draw.text((c * CELL + 12, 12), title, fill=(11, 16, 32))
for r, item in enumerate(ITEMS):
    for c, pattern in enumerate(cols):
        sheet.paste(cell(f"{prefix}/{pattern.format(item)}"), (c * CELL, 40 + r * CELL))
buf = io.BytesIO()
sheet.save(buf, "JPEG", quality=85)
s3.put_object(Bucket=bucket, Key=f"{prefix}/sheet.jpg", Body=buf.getvalue(), ContentType="image/jpeg")
print("sheet saved to the private bucket")
