"""Runtime configuration for the clean CutSell worker.

Only variable names live here. Secret values remain in Render/GitHub/RunPod.
"""
from __future__ import annotations

from dataclasses import dataclass
import os


RUNPOD_LOCAL_BACKEND = "runpod_local"


def _env_bool(values: dict[str, str], key: str, default: bool = False) -> bool:
    raw = values.get(key)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(values: dict[str, str], key: str, default: int) -> int:
    try:
        return int(values.get(key, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class RuntimeConfig:
    redis_url: str | None
    database_url: str | None
    sentry_dsn_present: bool
    openai_api_key_present: bool
    aws_access_key_present: bool
    aws_secret_key_present: bool
    aws_region: str | None
    s3_bucket: str | None
    runpod_api_key_present: bool
    runpod_template_id: str | None
    brain_backend: str
    asr_model: str
    semantic_model: str
    visual_model: str
    take_judge_model: str
    clean_cut_judge_model: str
    clean_cut_judge_enabled: bool
    max_source_minutes: int
    max_concurrent_jobs_per_user: int
    monthly_processing_minutes: int

    @property
    def storage_ready(self) -> bool:
        return bool(self.aws_access_key_present and self.aws_secret_key_present and self.aws_region and self.s3_bucket)

    @property
    def queue_ready(self) -> bool:
        return bool(self.redis_url)

    @property
    def commercial_db_ready(self) -> bool:
        return bool(self.database_url)

    @property
    def semantic_ready(self) -> bool:
        # A stored OpenAI key must never enable the brain implicitly. Mobile V1 is
        # RunPod-local; legacy external providers remain dormant on this backend.
        return bool(self.brain_backend != RUNPOD_LOCAL_BACKEND and self.openai_api_key_present)

    @property
    def visual_ready(self) -> bool:
        return bool(self.brain_backend != RUNPOD_LOCAL_BACKEND and self.openai_api_key_present)

    @property
    def clean_cut_judge_ready(self) -> bool:
        return bool(
            self.brain_backend != RUNPOD_LOCAL_BACKEND
            and self.clean_cut_judge_enabled
            and self.openai_api_key_present
        )


def load_runtime_config(env: dict[str, str] | None = None) -> RuntimeConfig:
    values = env if env is not None else os.environ
    return RuntimeConfig(
        redis_url=values.get("REDIS_URL"),
        database_url=values.get("DATABASE_URL"),
        sentry_dsn_present=bool(values.get("SENTRY_DSN")),
        openai_api_key_present=bool(values.get("OPENAI_API_KEY")),
        aws_access_key_present=bool(values.get("AWS_ACCESS_KEY_ID")),
        aws_secret_key_present=bool(values.get("AWS_SECRET_ACCESS_KEY")),
        aws_region=values.get("AWS_REGION"),
        s3_bucket=values.get("S3_BUCKET"),
        runpod_api_key_present=bool(values.get("RUNPOD_API_KEY")),
        runpod_template_id=values.get("RUNPOD_TEMPLATE_ID"),
        brain_backend=str(values.get("CUTSELL_BRAIN_BACKEND", RUNPOD_LOCAL_BACKEND)).strip().lower(),
        asr_model=values.get("CUTSELL_ASR_MODEL", "medium"),
        # Legacy model names remain readable for old metadata/config compatibility,
        # but they are not activated while brain_backend=runpod_local.
        semantic_model=values.get("CUTSELL_SEMANTIC_MODEL", "gpt-4o-mini"),
        visual_model=values.get("CUTSELL_VISUAL_MODEL", "gpt-4o-mini"),
        take_judge_model=values.get("CUTSELL_TAKE_JUDGE_MODEL", "gpt-4o-mini"),
        clean_cut_judge_model=values.get("CUTSELL_CLEAN_CUT_JUDGE_MODEL", "gpt-4o-mini"),
        clean_cut_judge_enabled=_env_bool(values, "CUTSELL_CLEAN_CUT_JUDGE", False),
        # 0 means no product-facing hard duration cap. A positive value can be enabled
        # later for a specific plan/infrastructure safety policy without changing code.
        max_source_minutes=max(0, _env_int(values, "CUTSELL_MAX_SOURCE_MINUTES", 0)),
        max_concurrent_jobs_per_user=max(1, _env_int(values, "CUTSELL_MAX_CONCURRENT_JOBS_PER_USER", 2)),
        monthly_processing_minutes=max(1, _env_int(values, "CUTSELL_MONTHLY_PROCESSING_MINUTES", 300)),
    )
