"""
Build FULL take_judge training dataset from worker JSON outputs.

It uses ALL jobs (bloopers + good) that save JSONs under:

    s3://script2clipshop-video-automatedretailservices/editdna/outputs/dataset/bloopers/

For every clip in every job it writes ONE row in a JSONL file with:
    session_id, clip_id, slot, text, keep (0/1), label, llm_reason, source

- `keep` / `label` come from clip["meta"]["keep"]
- `source` is "good" if session_id starts with "good", otherwise "bloopers"

Finally, it uploads the dataset to:

    s3://<BUCKET>/editdna/training/take_judge_dataset.jsonl
"""

import os
import json
from typing import List, Dict, Any

import boto3


# === CONFIG ===

# S3 bucket where your outputs already are
BUCKET_NAME = os.getenv(
    "EDITDNA_DATASET_BUCKET",
    "script2clipshop-video-automatedretailservices",
)

# Prefix where ALL JSONs live (bloopers + good)
TAKEJUDGE_PREFIX = os.getenv(
    "EDITDNA_TAKEJUDGE_PREFIX",
    "editdna/outputs/dataset/bloopers/",
)

# Where we will save the final dataset file in S3
OUTPUT_KEY = os.getenv(
    "EDITDNA_TAKEJUDGE_DATASET_KEY",
    "editdna/training/take_judge_dataset.jsonl",
)


def list_json_objects(s3_client) -> List[str]:
    """
    List all JSON objects under TAKEJUDGE_PREFIX.
    Returns a list of object keys.
    """
    keys: List[str] = []
    continuation_token = None

    while True:
        kwargs = {
            "Bucket": BUCKET_NAME,
            "Prefix": TAKEJUDGE_PREFIX,
        }
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token

        resp = s3_client.list_objects_v2(**kwargs)
        contents = resp.get("Contents", [])
        for obj in contents:
            key = obj["Key"]
            if key.endswith(".json"):
                keys.append(key)

        if resp.get("IsTruncated"):
            continuation_token = resp.get("NextContinuationToken")
        else:
            break

    return keys


def load_json_from_s3(s3_client, key: str) -> Dict[str, Any]:
    """Download and parse one JSON file from S3."""
    resp = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
    body = resp["Body"].read()
    return json.loads(body)


def build_rows_from_job(job: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Turn one job JSON into many training rows (one per clip).
    """
    result = job.get("result", {}) or {}
    session_id = (result.get("session_id") or "") or ""
    clips = result.get("clips", []) or []

    # Infer source from session_id (GoodX vs BloppersX, etc.)
    sid_lower = str(session_id).lower()
    if sid_lower.startswith("good"):
        source = "good"
    elif sid_lower.startswith("bloop") or sid_lower.startswith("bloopers"):
        source = "bloopers"
    else:
        # fallback – not critical, but keeps info
        source = "unknown"

    rows: List[Dict[str, Any]] = []

    for clip in clips:
        text = (clip.get("text") or "").strip()
        if not text:
            continue

        slot = clip.get("slot")
        meta = clip.get("meta") or {}
        keep_flag = meta.get("keep")

        # Normalize label to 1 (keep) / 0 (discard)
        label = 1 if keep_flag else 0

        row = {
            "session_id": session_id,
            "clip_id": clip.get("id"),
            "slot": slot,
            "text": text,
            "keep": bool(keep_flag),
            "label": label,
            "llm_reason": clip.get("llm_reason", ""),
            "source": source,
        }
        rows.append(row)

    return rows


def main():
    s3 = boto3.client("s3")

    print(f"Using bucket: {BUCKET_NAME}")
    print(f"Listing JSONs under prefix: {TAKEJUDGE_PREFIX}")

    keys = list_json_objects(s3)
    print(f"Found {len(keys)} JSON files")

    all_rows: List[Dict[str, Any]] = []

    for key in keys:
        print(f"Processing {key} ...")
        job_json = load_json_from_s3(s3, key)
        rows = build_rows_from_job(job_json)
        all_rows.extend(rows)

    print(f"Total clips (rows) collected: {len(all_rows)}")

    # Write local JSONL
    local_path = "/tmp/take_judge_dataset.jsonl"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    with open(local_path, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote local dataset file: {local_path}")

    # Upload to S3
    with open(local_path, "rb") as f:
        s3.put_object(Bucket=BUCKET_NAME, Key=OUTPUT_KEY, Body=f.read())

    print(f"Uploaded dataset to s3://{BUCKET_NAME}/{OUTPUT_KEY}")
    print("DONE ✅")


if __name__ == "__main__":
    main()
