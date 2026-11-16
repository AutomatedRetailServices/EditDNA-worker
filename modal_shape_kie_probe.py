"""Shape it through kie.ai: the one real test the PO approved on 16 Nov 2026 ("dale").

PAID: one kie.ai task per photo (~US$0.02 each, 4 photos = ~US$0.08). Runs on Modal because the kie.ai key is only
in the Modal secret "cutsell-worker" (with the S3 keys). The 4 test photos are the ones of the 12 Nov comparison, in
the PRIVATE bucket (never in this public repo):
    s3://$S3_BUCKET/private-qa/shape-compare/20261112/in/<label>.png
Results go next to them:
    s3://$S3_BUCKET/private-qa/shape-compare/20261112/out-kie/<label>-kie.png
Nothing image-like is printed. Run (CI only): `modal run modal_shape_kie_probe.py`.
"""
from __future__ import annotations

import modal

ITEMS = (("shirt", "clothing"), ("pants", "clothing"), ("shoes", "shoes"), ("sweater", "clothing"))
RUN_ID = "20261112"
MAX_CALLS = 4

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("requests>=2.31", "pillow>=10", "numpy>=1.26", "boto3>=1.34")
    .add_local_file("cutsell_worker/kie_image.py", remote_path="/root/kie_image.py")
    .add_local_file("cutsell_app/shop_shape.py", remote_path="/root/shop_shape.py")
)
app = modal.App("cutsell-shape-kie-probe")


@app.function(image=image, secrets=[modal.Secret.from_name("cutsell-worker")], timeout=900, cpu=1.0, memory=1024)
def probe() -> list[dict]:
    import base64
    import os
    import time

    import boto3

    import kie_image
    import shop_shape

    if not os.environ.get("KIE_API_KEY", "").strip():
        return [{"error": "KIE_API_KEY is not in the Modal secret cutsell-worker"}]
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    bucket, prefix = os.environ["S3_BUCKET"], f"private-qa/shape-compare/{RUN_ID}"
    rows = []
    for label, kind in ITEMS[:MAX_CALLS]:
        row = {"item": label, "kind": kind}
        try:
            png = s3.get_object(Bucket=bucket, Key=f"{prefix}/in/{label}.png")["Body"].read()
            garment = shop_shape.decode_image(base64.b64encode(png).decode("ascii"))
            started = time.monotonic()
            raw, info = kie_image.shape_png(garment.flattened_png(), shop_shape.build_prompt(kind, background=garment.key_name))
            out = shop_shape.finish_output(raw, garment)
            s3.put_object(Bucket=bucket, Key=f"{prefix}/out-kie/{label}-kie.png", Body=out, ContentType="image/png")
            row.update(ok=True, seconds=round(time.monotonic() - started, 1), cost_usd=info["cost_usd"],
                       model=info["model"], size=f"{garment.width}x{garment.height}")
        except kie_image.KieError as exc:
            row.update(ok=False, error=exc.code, detail=exc.detail[:160])
        except Exception as exc:                     # noqa: BLE001 -- report, never print image data
            row.update(ok=False, error=exc.__class__.__name__, detail=str(exc)[:160])
        print(row)
        rows.append(row)
    return rows


@app.local_entrypoint()
def main():
    rows = probe.remote()
    ok = sum(1 for r in rows if r.get("ok"))
    print(f"RESULT {ok}/{len(rows)} ok; cost ~US${sum(float(r.get('cost_usd') or 0) for r in rows):.2f}")
    for r in rows:
        print("RESULT", r)
