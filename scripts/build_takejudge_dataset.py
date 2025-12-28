"""
Build FULL take_judge training dataset from worker JSON outputs.

It uses ALL jobs (bloopers + good) that save JSONs under:

    s3://script2clipshop-video-automatedretailservices/editdna/outputs/dataset/bloopers/

For every clip in every job it writes ONE row in a JSONL file with:

    session_id, clip_id, slot, text, keep (0/1), label, llm_reason,
    source, take_judge_score, take_judge_verdict

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

    It supports both shapes:
    - {"result": {..., "clips": [...]}}
    - {"session_id": ..., "clips": [...]}
    """
    # Handle both shapes safely
    result = job.get("result", job)

    session_id = (result.get("session_id") or "").strip()
    clips = result.get("clips", []) or []

    rows: List[Dict[str, Any]] = []

    for clip in clips:
        text = (clip.get("text") or "").strip()
        if not text:
            continue

        meta = clip.get("meta") or {}

        # Try to get manual keep flag from meta
        keep_flag = meta.get("keep")

        # If no keep flag:
        # - If it's a "Good" session, assume keep = True
        # - If it's NOT a "Good" session, skip (unlabeled blooper)
        if keep_flag is None:
            if session_id.lower().startswith("good"):
                keep_flag = True
            else:
                continue

        label = 1 if keep_flag else 0
        source = "good" if session_id.lower().startswith("good") else "bloopers"

        row = {
            "session_id": session_id,
            "clip_id": clip.get("id"),
            "slot": clip.get("slot"),
            "text": text,
            "keep": bool(keep_flag),
            "label": label,
            "source": source,
            "llm_reason": clip.get("llm_reason", ""),
            "take_judge_score": meta.get("take_judge_score"),
            "take_judge_verdict": meta.get("take_judge_verdict"),
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
