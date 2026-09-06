"""Commercial error/crash observability bootstrap for API + worker."""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("cutsell")
_initialized = False


def initialize_observability(*, service: str) -> dict[str, Any]:
    global _initialized
    dsn = str(os.getenv("SENTRY_DSN") or "").strip()
    environment = str(os.getenv("CUTSELL_ENVIRONMENT") or "staging").strip()
    release = str(os.getenv("CUTSELL_RELEASE") or "cutsell-unversioned").strip()
    if _initialized:
        return {"status": "already_initialized", "service": service}
    if not dsn:
        return {"status": "not_configured", "service": service}
    try:
        import sentry_sdk
        sentry_sdk.init(
            dsn=dsn,
            environment=environment,
            release=release,
            traces_sample_rate=float(os.getenv("CUTSELL_SENTRY_TRACES_SAMPLE_RATE") or "0.1"),
            send_default_pii=False,
        )
        sentry_sdk.set_tag("service", service)
        _initialized = True
        return {"status": "initialized", "service": service, "environment": environment}
    except Exception as exc:
        logger.warning("CutSell observability initialization degraded: %s", exc.__class__.__name__)
        return {"status": "degraded", "service": service, "reason": exc.__class__.__name__}


def capture_operational_event(name: str, **details: Any) -> None:
    """Structured event hook; logs always, forwards breadcrumb when Sentry is active."""
    logger.info("cutsell_event name=%s details=%s", name, details)
    if not _initialized:
        return
    try:
        import sentry_sdk
        sentry_sdk.add_breadcrumb(category="cutsell", message=name, data=details, level="info")
    except Exception:
        pass
