#!/usr/bin/env python3
"""Shape it comparison inside GitHub Actions (PAID: one call per photo and provider, no retries).

The photos live in the PRIVATE worker bucket (never in this public repo). This script:
  1. reads the worker's environment from the canonical RunPod template (AWS keys, S3_BUCKET, provider keys)
     and masks every credential-like value in the log;
  2. downloads   s3://$S3_BUCKET/private-qa/shape-compare/<RUN_ID>/in/*.png
  3. runs scripts/shape_it_compare.py on them (outputs in RUNNER_TEMP, outside the repo);
  4. uploads the results to  s3://$S3_BUCKET/private-qa/shape-compare/<RUN_ID>/out/
Nothing image-like is printed or uploaded as a GitHub artifact.

Env: SHAPE_RUN_ID, SHAPE_PROVIDERS (e.g. "nano-banana"), SHAPE_ITEMS ("shirt:clothing,shoes:shoes"),
     RUNPOD_API_KEY, optional GEMINI_API_KEY / OPENAI_API_KEY overrides.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import boto3
import requests

ROOT = Path(__file__).resolve().parents[1]
PROVIDER_KEYS = ("GEMINI_API_KEY", "OPENAI_API_KEY", "CUTSELL_SEEDREAM_API_KEY")
MAX_CALLS = 8


def mask(value: str) -> None:
    if value:
        print("::add-mask::" + value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A"))


def template_env() -> dict:
    response = requests.get("https://rest.runpod.io/v1/templates",
                            headers={"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"}, timeout=(15, 60))
    if response.status_code != 200:
        raise SystemExit(f"template lookup HTTP {response.status_code}")
    matches = [t for t in response.json() if t.get("name") == "EditDNA-Worker-2"]
    if len(matches) != 1:
        raise SystemExit("expected one canonical worker template")
    env = {str(k): str(v) for k, v in matches[0]["env"].items()}
    for name, value in env.items():
        if any(p in name.upper() for p in ("KEY", "TOKEN", "SECRET", "PASSWORD", "BUCKET")) or len(value) >= 12:
            mask(value)
    return env


def main() -> int:
    run_id = os.environ["SHAPE_RUN_ID"]
    if not re.fullmatch(r"[0-9a-z-]{4,40}", run_id):
        raise SystemExit("bad SHAPE_RUN_ID")
    providers = os.environ.get("SHAPE_PROVIDERS", "nano-banana")
    items = [i.split(":") for i in os.environ["SHAPE_ITEMS"].split(",") if i]
    if len(items) * len(providers.split(",")) > MAX_CALLS:
        raise SystemExit(f"refusing: more than {MAX_CALLS} paid calls")

    env = template_env()
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "S3_BUCKET"):
        if not env.get(name):
            raise SystemExit(f"template is missing {name}")
    child = dict(os.environ)
    for name in PROVIDER_KEYS:
        if not child.get(name, "").strip() and env.get(name):
            child[name] = env[name]
    for name in PROVIDER_KEYS:
        print(f"{name}: {'present' if child.get(name, '').strip() else 'absent'}")

    s3 = boto3.client("s3", region_name=env.get("AWS_REGION", "us-east-1"),
                      aws_access_key_id=env["AWS_ACCESS_KEY_ID"],
                      aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"],
                      aws_session_token=env.get("AWS_SESSION_TOKEN") or None)
    bucket, prefix = env["S3_BUCKET"], f"private-qa/shape-compare/{run_id}"
    work = Path(os.environ["RUNNER_TEMP"]) / "shape-compare"
    (work / "in").mkdir(parents=True, exist_ok=True)
    args = []
    for label, kind in items:
        path = work / "in" / f"{label}.png"
        s3.download_file(bucket, f"{prefix}/in/{label}.png", str(path))
        args += ["--item", f"{label}:{kind}:{path}"]

    out = work / "out"
    code = subprocess.call([sys.executable, str(ROOT / "scripts/shape_it_compare.py"), *args,
                            "--out", str(out), "--providers", providers, "--yes-spend"], env=child, cwd=ROOT)
    uploaded = 0
    for f in sorted(out.glob("*.png")) if out.exists() else []:
        s3.upload_file(str(f), bucket, f"{prefix}/out/{f.name}", ExtraArgs={"ContentType": "image/png"})
        uploaded += 1
    print(f"uploaded {uploaded} result images to the private bucket")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
