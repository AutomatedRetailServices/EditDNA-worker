"""Runtime configuration for the clean CutSell worker.

Only variable names live here. Secret values remain in Render/GitHub/RunPod.
"""
from __future__ import annotations

from dataclasses import dataclass
import os


def _env_bool(values: dict[str, str], key: str, default: bool = False) -> bool:
    raw = values.get(key)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


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
    visual_model: str
    take_judge_model: str
    clean_cut_judge_model: str
    clean_cut_judge_enabled: bool

    @property
    def storage_ready(self) -> bool:
        return bool(self.aws_access_key_present and self.aws_secret_key_present and self.aws_region and self.s3_bucket)

    @property
    def queue_ready(self) -> bool:
        return bool(self.redis_url)

    @property
    def semantic_ready(self) -> bool:
        return self.openai_api_key_present

    @property
    def visual_ready(self) -> bool:
        return self.openai_api_key_present

    @property
    def clean_cut_judge_ready(self) -> bool:
        return bool(self.clean_cut_judge_enabled and self.openai_api_key_present)


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
        # The inherited EditDNA OpenAI project currently exposes gpt-4o-mini.
        # All provider models remain independently overridable as access expands.
        semantic_model=values.get("CUTSELL_SEMANTIC_MODEL", "gpt-4o-mini"),
        visual_model=values.get("CUTSELL_VISUAL_MODEL", "gpt-4o-mini"),
        take_judge_model=values.get("CUTSELL_TAKE_JUDGE_MODEL", "gpt-4o-mini"),
        clean_cut_judge_model=values.get("CUTSELL_CLEAN_CUT_JUDGE_MODEL", "gpt-4o-mini"),
        # Experimental until golden real-video validation proves it is safe.
        clean_cut_judge_enabled=_env_bool(values, "CUTSELL_CLEAN_CUT_JUDGE", False),
    )
