"""Lazy, centrally configured OpenAI SDK client construction."""

import os
from typing import Any, Optional

from .config import load_model_config


class OpenAIProviderError(RuntimeError):
    """Safe provider failure suitable for crossing the model boundary."""


class OpenAITimeoutError(OpenAIProviderError):
    """The provider operation exceeded its configured timeout."""


class OpenAIResponseValidationError(OpenAIProviderError):
    """The provider returned content that did not match the internal schema."""


def create_openai_client(api_key: Optional[str] = None) -> Any:
    """Construct a client only for an operation that is actually being executed."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise OpenAIProviderError("OpenAI operation is unavailable: API key is not configured")
    config = load_model_config().openai
    try:
        from openai import OpenAI
        return OpenAI(api_key=key, timeout=config.timeout_seconds, max_retries=config.max_retries)
    except OpenAIProviderError:
        raise
    except Exception as exc:
        raise OpenAIProviderError("OpenAI client initialization failed") from exc
