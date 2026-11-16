"""Shape it through kie.ai: the one real test the PO approved on 16 Nov 2026 ("dale").

PAID: one kie.ai task per photo (~US$0.02 each, 4 photos = ~US$0.08). Runs on Modal because the kie.ai key is only
in the Modal secret "cutsell-worker" (with the S3 keys). The 4 test photos are the ones of the 12 Nov comparison, in
the PRIVATE bucket (never in this public repo):
    s3://$S3_BUCKET/private-qa/shape-compare/20261112/in/<label>.png
Results go next to them:
    s3://$S3_BUCKET/private-qa/shape-compare/20261112/out-kie/<label>-kie.png
The bucket and its keys come from the worker template (RUNPOD_API_KEY), exactly as the 12 Nov comparison read them.
Nothing image-like is printed. Run (CI only): `modal run modal_shape_kie_probe.py`.
"""
from __future__ import annotations

import modal

ITEMS = (("shirt", "clothing"), ("pants", "clothing"), ("shoes", "shoes"), ("sweater", "clothing"))
RUN_ID = "20261112"
MAX_CALLS = 14
# PO 16 Nov: which kie.ai model keeps the product exactly? (photos per model; the expensive one on 2 photos only)
MODELS = (("nano-banana-2", 4), ("nano-banana-pro", 2), ("bytedance/seedream-v4-edit", 4), ("qwen/image-edit", 4))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("requests>=2.31", "pillow>=10", "numpy>=1.26", "boto3>=1.34")
    .add_local_file("cutsell_worker/kie_image.py", remote_path="/root/kie_image.py")
    .add_local_file("cutsell_app/shop_shape.py", remote_path="/root/shop_shape.py")
)
app = modal.App("cutsell-shape-kie-probe")


@app.function(image=image, secrets=[modal.Secret.from_name("cutsell-worker")], timeout=900, cpu=1.0, memory=1024)
def probe(items: list, models: list) -> list[dict]:
    """items: [(label, kind, png bytes)] -> rows (with "png" = the transparent result for the ok ones)."""
    import base64
    import os
    import time

    import kie_image
    import shop_shape

    if not os.environ.get("KIE_API_KEY", "").strip():
        return [{"error": "KIE_API_KEY is not in the Modal secret cutsell-worker"}]
    rows = []
    jobs = [(m, label, kind, png) for m, n in models for label, kind, png in items[:n]][:MAX_CALLS]
    for model, label, kind, png in jobs:
        row = {"model": model, "item": label, "kind": kind}
        try:
            garment = shop_shape.decode_image(base64.b64encode(png).decode("ascii"))
            started = time.monotonic()
            raw, info = kie_image.shape_png(garment.flattened_png(), shop_shape.build_prompt(kind, background=garment.key_name),
                                            model=model)
            row.update(ok=True, seconds=round(time.monotonic() - started, 1), cost_usd=info["cost_usd"],
                       model=info["model"], size=f"{garment.width}x{garment.height}",
                       png=shop_shape.finish_output(raw, garment))
        except kie_image.KieError as exc:
            row.update(ok=False, error=exc.code, detail=exc.detail[:160])
        except Exception as exc:                     # noqa: BLE001 -- report, never print image data
            row.update(ok=False, error=exc.__class__.__name__, detail=str(exc)[:160])
        rows.append(row)
    return rows


@app.local_entrypoint()
def main():
    """On the GitHub runner: photos from the private bucket (the worker template's keys, as the 12 Nov comparison),
    kie.ai on Modal, results back to the private bucket."""
    import os
    import sys

    import boto3

    sys.path.insert(0, "scripts")
    from shape_it_compare_ci import template_env

    env = template_env()
    s3 = boto3.client("s3", region_name=env.get("AWS_REGION", "us-east-1"),
                      aws_access_key_id=env["AWS_ACCESS_KEY_ID"], aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"],
                      aws_session_token=env.get("AWS_SESSION_TOKEN") or None)
    bucket, prefix = env["S3_BUCKET"], f"private-qa/shape-compare/{RUN_ID}"
    items = [(label, kind, s3.get_object(Bucket=bucket, Key=f"{prefix}/in/{label}.png")["Body"].read())
             for label, kind in ITEMS[:MAX_CALLS]]
    rows = probe.remote(items, [list(m) for m in MODELS])
    for r in rows:
        png = r.pop("png", None)
        if png:
            tag = r["model"].replace("/", "_")
            s3.put_object(Bucket=bucket, Key=f"{prefix}/out-kie-models/{tag}/{r['item']}.png", Body=png, ContentType="image/png")
            r["saved"] = True
        print("RESULT", r)
    ok = sum(1 for r in rows if r.get("ok"))
    print(f"RESULT {ok}/{len(rows)} ok; cost ~US${sum(float(r.get('cost_usd') or 0) for r in rows):.2f}")
