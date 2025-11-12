# tasks.py
import os
import json
from typing import List, Optional, Dict, Any
from worker.pipeline import run_pipeline

# RQ job entrypoint
def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expected payload:
    {
      "session_id": "funnel-test-1",
      "files": ["https://.../IMG_02.mov", "..."],  # at least one
      "portrait": true,
      "s3_prefix": "editdna/outputs/",
      "target_duration": 35.0,   # optional (seconds); None = no target cap
      "max_duration": None,      # optional hard cap; None = no cap
      "force_cta": False,        # optional
      "openai_model": "gpt-4o-mini",  # optional
    }
    """
    session_id: str = payload.get("session_id") or "session-unknown"
    files: List[str] = payload.get("files") or []
    portrait: bool = bool(payload.get("portrait", True))
    s3_prefix: Optional[str] = payload.get("s3_prefix") or "editdna/outputs/"
    target_duration = payload.get("target_duration", None)
    max_duration = payload.get("max_duration", None)
    force_cta = bool(payload.get("force_cta", False))
    openai_model = payload.get("openai_model", "gpt-4o-mini")

    out = run_pipeline(
        session_id=session_id,
        file_urls=files,                 # REQUIRED keyword
        portrait=portrait,
        s3_prefix=s3_prefix,
        target_duration=target_duration, # None → keep all good clauses
        max_duration=max_duration,       # None → no hard cap
        llm_always_on=True,              # “full brain” as requested
        force_cta=force_cta,
        openai_model=openai_model,
    )
    return out

# simple manual test (optional)
if __name__ == "__main__":
    demo = {
        "session_id": "local-test",
        "files": [os.getenv("TEST_VIDEO_URL", "")],
        "portrait": True,
        "s3_prefix": "editdna/outputs/",
        "target_duration": None,
        "max_duration": None,
        "force_cta": False,
    }
    print(json.dumps(job_render(demo), indent=2))
