"""Plan my week: preferences, the week (built with rules from Ready videos), moving / removing a video, marking it
posted, and her own caption (never overwritten by the AI)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from cutsell_worker import post_plan
from cutsell_worker.project_store import list_projects

router = APIRouter(prefix="/v1/plan", tags=["plan"])


def _projects(user_id: str) -> list[dict]:
    return list_projects(user_id=user_id, limit=500)


def _owned(user_id: str, project_id: str) -> None:
    if project_id not in {str(p.get("project_id")) for p in _projects(user_id)}:
        raise HTTPException(status_code=404, detail="video not found")


def _guard(call):
    try:
        return call()
    except (TypeError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc).strip("'")) from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


def _view(user_id: str) -> dict:
    return post_plan.view(user_id, _projects(user_id))


def _caption_missing(user_id: str, view: dict) -> None:
    """Planned videos without a caption get one written (once) by the worker."""
    missing = [s["project_id"] for s in ((view.get("week") or {}).get("slots") or []) if not s.get("caption")]
    if missing:
        try:
            from cutsell_worker.queueing import enqueue_captions
            enqueue_captions({"user_id": user_id, "project_ids": missing})
        except Exception:
            pass


@router.get("")
def get_plan(user_id: str):
    return _guard(lambda: _view(user_id))


class Reminder(BaseModel):
    label: str = Field(default="Reminder", max_length=30)
    time: str = Field(pattern=r"^\d{1,2}:\d{2}$")
    on: bool = True


class PrefsRequest(BaseModel):
    user_id: str
    per_day: int | None = Field(default=None, ge=1, le=post_plan.MAX_PER_DAY)
    days: list[int] | None = None
    reminders: list[Reminder] | None = Field(default=None, max_length=6)
    tz: str | None = None
    # Goal v2 (flexible): her posting times (one video each, at most 8), how many videos, from which day, reminders
    times: list[str] | None = Field(default=None, min_length=1, max_length=post_plan.MAX_TIMES)
    count: int | None = Field(default=None, ge=1, le=1000)
    start: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    remind: bool | None = None


@router.put("/prefs")
def put_prefs(payload: PrefsRequest):
    prefs = {k: v for k, v in payload.model_dump().items() if k != "user_id" and v is not None}
    _guard(lambda: post_plan.set_prefs(payload.user_id, prefs))
    return _view(payload.user_id)


class UserRequest(BaseModel):
    user_id: str


@router.post("/week/build")
def build_week(payload: UserRequest):
    _guard(lambda: post_plan.build_week(payload.user_id, _projects(payload.user_id)))
    view = _view(payload.user_id)
    _caption_missing(payload.user_id, view)
    return view


class MoveRequest(BaseModel):
    user_id: str
    project_id: str
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    time: str = Field(pattern=r"^\d{1,2}:\d{2}$")


@router.post("/slots/move")
def move_slot(payload: MoveRequest):
    _owned(payload.user_id, payload.project_id)
    _guard(lambda: post_plan.move(payload.user_id, payload.project_id, day=payload.date, time=payload.time))
    return _view(payload.user_id)


class VideoRequest(BaseModel):
    user_id: str
    project_id: str


@router.post("/slots/remove")
def remove_slot(payload: VideoRequest):
    _owned(payload.user_id, payload.project_id)
    _guard(lambda: post_plan.remove(payload.user_id, payload.project_id))
    return _view(payload.user_id)


class PostedRequest(BaseModel):
    user_id: str
    project_id: str
    posted: bool = True


@router.post("/posted")
def mark_posted(payload: PostedRequest):
    _owned(payload.user_id, payload.project_id)
    _guard(lambda: post_plan.mark_posted(payload.user_id, payload.project_id, posted=payload.posted))
    return _view(payload.user_id)


class CaptionRequest(BaseModel):
    user_id: str
    project_id: str
    caption: str = Field(min_length=1, max_length=2200)


@router.put("/caption")
def put_caption(payload: CaptionRequest):
    """Her own caption: kept as it is; the AI never writes over it."""
    _owned(payload.user_id, payload.project_id)
    _guard(lambda: post_plan.set_caption(payload.user_id, payload.project_id, payload.caption, source="user"))
    return {"project_id": payload.project_id, "caption": payload.caption.strip(), "caption_source": "user"}


class PersonalRequest(BaseModel):
    user_id: str
    project_id: str
    personal: bool = True


@router.post("/personal")
def mark_personal(payload: PersonalRequest):
    """A personal video is never put in a week (and leaves the current week); `personal: false` brings it back."""
    _owned(payload.user_id, payload.project_id)
    _guard(lambda: post_plan.set_personal(payload.user_id, [payload.project_id], personal=payload.personal))
    if payload.personal:
        _guard(lambda: post_plan.remove(payload.user_id, payload.project_id))
    return _view(payload.user_id)
