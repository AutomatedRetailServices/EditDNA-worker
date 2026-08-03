"""Private, allowlisted S3 access for historical benchmarks."""

import os
from pathlib import Path, PurePosixPath
from typing import Iterable

import boto3
from botocore.exceptions import ClientError

DEFAULT_INPUT_PREFIXES = {
    "BENCHMARK_BLOOPERS_PREFIX": "Editdna bloopers videos/",
    "BENCHMARK_GOOD_VIDEOS_PREFIX": "Editdna good videos/",
    "BENCHMARK_TRAINING_PREFIX": "editdna/training/",
    "BENCHMARK_HISTORICAL_OUTPUTS_PREFIX": "editdna/outputs/",
}
OUTPUT_PREFIX = "editdna/benchmarks/"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm"}
DATA_EXTENSIONS = {".jsonl", ".json"}
MIN_OBJECT_BYTES = int(os.getenv("BENCHMARK_MIN_OBJECT_BYTES", "1024"))
MAX_OBJECT_BYTES = int(os.getenv("BENCHMARK_MAX_OBJECT_BYTES", str(8 * 1024**3)))


def configured_bucket() -> str:
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        raise RuntimeError("S3_BUCKET is required")
    return bucket


def client():
    return boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))


def _safe_key(key: str) -> str:
    path = PurePosixPath(key)
    if not key or key.startswith("/") or ".." in path.parts or "\\" in key:
        raise ValueError("unsafe S3 key")
    return key


def configured_input_prefixes() -> tuple[str, ...]:
    prefixes = []
    for environment_name, default in DEFAULT_INPUT_PREFIXES.items():
        prefix = os.getenv(environment_name, default)
        _safe_key(prefix)
        if not prefix or not prefix.endswith("/"):
            raise ValueError(f"{environment_name} must be a non-empty relative S3 prefix ending in '/'")
        prefixes.append(prefix)
    if len(prefixes) != len(set(prefixes)):
        raise ValueError("benchmark input prefixes must be unique")
    return tuple(prefixes)


def validate_input_prefix(prefix: str) -> str:
    _safe_key(prefix)
    if prefix not in configured_input_prefixes():
        raise ValueError("source prefix is not allowlisted")
    return prefix


def validate_dataset_key(key: str) -> str:
    _safe_key(key)
    if not key.startswith(configured_input_prefixes()[2]) or Path(key).suffix.lower() not in DATA_EXTENSIONS:
        raise ValueError("dataset key is not allowlisted")
    return key


def validate_output_key(key: str, job_id: str | None = None) -> str:
    _safe_key(key)
    required = f"{OUTPUT_PREFIX}{job_id}/" if job_id else OUTPUT_PREFIX
    if not key.startswith(required):
        raise ValueError("benchmark output key is not allowlisted")
    return key


def list_objects_inventory(s3, prefix: str, extensions: Iterable[str] = VIDEO_EXTENSIONS) -> tuple[list[dict], dict]:
    validate_input_prefix(prefix)
    allowed = {x.lower() for x in extensions}
    found, token, seen = [], None, 0
    while True:
        args = {"Bucket": configured_bucket(), "Prefix": prefix}
        if token:
            args["ContinuationToken"] = token
        response = s3.list_objects_v2(**args)
        for item in response.get("Contents", []):
            seen += 1
            key, size = item["Key"], int(item.get("Size", 0))
            name = PurePosixPath(key).name
            if (key.endswith("/") or name.startswith("._") or size < MIN_OBJECT_BYTES
                    or size > MAX_OBJECT_BYTES or Path(name).suffix.lower() not in allowed):
                continue
            found.append({"key": key, "size": size})
        if not response.get("IsTruncated"):
            break
        token = response["NextContinuationToken"]
    found = sorted(found, key=lambda value: value["key"].casefold())
    return found, {"filtered_s3_objects": seen - len(found), "eligible_s3_videos": len(found)}


def list_objects(s3, prefix: str, extensions: Iterable[str] = VIDEO_EXTENSIONS) -> list[dict]:
    return list_objects_inventory(s3, prefix, extensions)[0]


def read_object(s3, key: str, max_bytes: int = MAX_OBJECT_BYTES) -> bytes:
    validate_dataset_key(key)
    response = s3.get_object(Bucket=configured_bucket(), Key=key)
    if int(response.get("ContentLength", 0)) > max_bytes:
        raise ValueError("object exceeds maximum size")
    data = response["Body"].read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("object exceeds maximum size")
    return data


def read_output(s3, key: str, job_id: str, max_bytes: int = 16 * 1024 * 1024) -> bytes | None:
    validate_output_key(key, job_id)
    try:
        response = s3.get_object(Bucket=configured_bucket(), Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
            return None
        raise
    if int(response.get("ContentLength", 0)) > max_bytes:
        raise ValueError("checkpoint exceeds maximum size")
    data = response["Body"].read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("checkpoint exceeds maximum size")
    return data


def download_video(s3, key: str, destination: Path) -> None:
    _safe_key(key)
    if not any(key.startswith(prefix) for prefix in configured_input_prefixes()):
        raise ValueError("video key is not allowlisted")
    if Path(key).suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError("unsupported video extension")
    destination.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(configured_bucket(), key, str(destination))


def put_output(s3, key: str, body: bytes, job_id: str, content_type: str) -> None:
    validate_output_key(key, job_id)
    s3.put_object(Bucket=configured_bucket(), Key=key, Body=body, ContentType=content_type)


def presign_output(s3, key: str, job_id: str, expires: int = 3600) -> str:
    validate_output_key(key, job_id)
    if not 1 <= expires <= 3600:
        raise ValueError("expiry must be between 1 and 3600 seconds")
    return s3.generate_presigned_url("get_object", Params={"Bucket": configured_bucket(), "Key": key}, ExpiresIn=expires)
