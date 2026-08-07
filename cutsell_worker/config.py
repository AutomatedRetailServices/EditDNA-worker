"""Runtime configuration for the clean CutSell worker.

Only variable names live here. Secret values remain in Render/GitHub/RunPod.
"""
from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class RuntimeConfig:
    redis_url: str | None
    openai_api_key_present: bool
    aws_access_key_present: bool
    aws_secret_key_present: bool
    aws_region: str | None
    s3_bucket: str | None
    runpod_api_key_present: bool
    runpod_template_id: str | None
    asr_model: str
    semantic_model: str
    take_judge_model: str

    @property
    def storage_ready(self) -> bool:
        return bool(self.aws_access_key_present and self.aws_secret_key_present and self.aws_region and self.s3_bucket)

    @property
    def queue_ready(self) -> bool:
        return bool(self.redis_url)

    @property
    def semantic_ready(self) -> bool:
        return self.openai_api_key_present


def load_runtime_config(env: dict[str, str] | None = None) -> RuntimeConfig:
    values = env if env is not None else os.environ
    return RuntimeConfig(
        redis_url=values.get("REDIS_URL"),
        openai_api_key_present=bool(values.get("OPENAI_API_KEY")),
        aws_access_key_present=bool(values.get("AWS_ACCESS_KEY_ID")),
        aws_secret_key_present=bool(values.get("AWS_SECRET_ACCESS_KEY")),
        aws_region=values.get("AWS_REGION"),
        s3_bucket=values.get("S3_BUCKET"),
        runpod_api_key_present=bool(values.get("RUNPOD_API_KEY")),
        runpod_template_id=values.get("RUNPOD_TEMPLATE_ID"),
        asr_model=values.get("CUTSELL_ASR_MODEL", "medium"),
        semantic_model=values.get("CUTSELL_SEMANTIC_MODEL", "gpt-4o-mini"),
        take_judge_model=values.get("CUTSELL_TAKE_JUDGE_MODEL", "gpt-4o-mini"),
    )
