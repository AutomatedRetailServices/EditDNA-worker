# web/routes_render.py
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, AnyHttpUrl
from typing import Optional, Dict, Any
from tasks import job_render

router = APIRouter()

class RenderRequest(BaseModel):
    # REQUIRED: S3 (or any HTTP) video URL
    input_url: AnyHttpUrl
    # Optional knobs
    frame_stride: Optional[int] = 12
    max_frames_for_llm: Optional[int] = 4
    session_id: Optional[str] = "funnel-test-1"
    target_duration_sec: Optional[int] = None  # ignored by fluid pipeline but allowed

@router.post("/render")
async def render(payload: RenderRequest = Body(...)) -> Dict[str, Any]:
    """
    JSON-only endpoint. Send:
      {
        "input_url": "https://.../video.mp4",
        "frame_stride": 12,
        "max_frames_for_llm": 4,
        "session_id": "funnel-test-1"
      }
    """
    # Hand straight to the worker pipeline
    return job_render(payload.model_dump(exclude_none=True))
