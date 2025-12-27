"""
Build FULL TakeJudge training dataset from worker JSON outputs.

- Reads all JSON files under BOTH S3 prefixes:
    Bloopers: s3://script2clipshop-video-automatedretailservices/editdna/outputs/dataset/bloopers/
    Good:     s3://script2clipshop-video-automatedretailservices/editdna/outputs/dataset/good/

- For every clip in every job it writes ONE row in a JSONL file with:
    dataset ("bloopers" or "good"),
    session_id, clip_id, slot, text,
    keep (True/False), label (1 keep / 0 discard),
    llm_reason, and some scores (semantic/visual/take_judge).
"""

import os
import json
from typing import List, Dict, Any

import boto3


# === CONFIG ===

BUCKET_NAME = os.getenv(
    "EDITDNA_DATASET_BUCKET",
    "script2clipshop-video-automatedretailservices",
)

BLOOPERS_PREFIX = os.getenv(
    "EDITDNA_BLOOPERS_PREFIX",
    "editdna/outputs/dataset/bloopers/",
)

GOOD_PREFIX = os.getenv(
    "EDITDNA_GOOD_PREFIX",
    "editdna/outputs/dataset/good/",
)

OUTPUT_KEY = os.getenv(
    "EDITDNA_TAKEJUDGE_DATASET_KEY",
    "editdna/training/take_judge_dataset.jsonl",
)


def list_json_objects(s3_client, prefix: str) -> List[str]:
    """
    List all JSON objects under a given prefix.
    Returns a list of object keys.
    """
    keys: List[str] = []
    continuation_token = None

    while True:
        kwargs = {
            "Bucket": BUCKET_NAME,
            "Prefix": prefix,
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


def build_rows_from_job(job: Dict[str, Any], dataset_name: str) -> List[Dict[str, Any]]:
    """
    Turn one job JSON into many training rows (one per clip).
    """
    result = job.get("result", {})
    session_id = result.get("session_id")
    clips = result.get("clips", []) or []

    rows: List[Dict[str, Any]] = []

    for clip in clips:
        text = (clip.get("text") or "").strip()
        if not text:
            continue

        slot = clip.get("slot")
        meta = clip.get("meta") or {}
        keep_flag = meta.get("keep")
        label = 1 if keep_flag else 0  # 1 = keep, 0 = discard

        row = {
            "dataset": dataset_name,  # "bloopers" or "good"
            "session_id": session_id,
            "clip_id": clip.get("id"),
            "slot": slot,
            "text": text,
            "keep": bool(keep_flag),
            "label": label,
            "llm_reason": clip.get("llm_reason", ""),
            "score": clip.get("score"),
            "semantic_score": clip.get("semantic_score"),
            "visual_score": clip.get("visual_score"),
            "take_judge_score": meta.get("take_judge_score"),
            "take_judge_verdict": meta.get("take_judge_verdict"),
        }
        rows.append(row)

    return rows


def collect_dataset_for_prefix(s3_client, prefix: str, dataset_name: str) -> List[Dict[str, Any]]:
    keys = list_json_objects(s3_client, prefix)
    print(f"[{dataset_name}] Found {len(keys)} JSON files under {prefix}")

    all_rows: List[Dict[str, Any]] = []

    for key in keys:
        print(f"[{dataset_name}] Processing {key} ...")
        job_json = load_json_from_s3(s3_client, key)
        rows = build_rows_from_job(job_json, dataset_name)
        all_rows.extend(rows)

    print(f"[{dataset_name}] Total clips (rows): {len(all_rows)}")
    return all_rows


def main():
    s3 = boto3.client("s3")

    print(f"Using bucket: {BUCKET_NAME}")

    all_rows: List[Dict[str, Any]] = []

    # Bloopers dataset
    all_rows.extend(
        collect_dataset_for_prefix(s3, BLOOPERS_PREFIX, "bloopers")
    )

    # Good dataset
    all_rows.extend(
        collect_dataset_for_prefix(s3, GOOD_PREFIX, "good")
    )

    print(f"TOTAL rows (bloopers + good): {len(all_rows)}")

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
