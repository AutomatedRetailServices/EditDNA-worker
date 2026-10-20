"""Mix: the app uploads every clip of a Mix (voice first) into one project, then starts ONE job for the whole Mix.

POST /v1/mix/plan     the combinations for these section sizes (the "How many videos?" preview), no storage
POST /v1/mix/jobs     save the Mix, make one project per video (in its product, tagged MIX + code), start the job
GET  /v1/mix          the creator's recent Mixes (Edits · Processing shows each as one group)
GET  /v1/mix/{id}     one Mix: each video's code and state, failed clips
POST /v1/mix/{id}/retry   run the job again for the failed clips only
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from cutsell_worker import mix_engine as engine
from cutsell_worker import mix_store as store
from cutsell_worker import product_catalog as catalog
from cutsell_worker.project_store import create_project, get_project, update_project
from cutsell_worker.uploads import validate_product_source_uri

router = APIRouter(prefix="/v1/mix", tags=["mix"])


class PlanRequest(BaseModel):
    user_id: str
    sections: list[str] = Field(min_length=1, max_length=engine.MAX_SECTIONS)
    counts: list[int] = Field(min_length=1, max_length=engine.MAX_SECTIONS)
    how_many: int | None = Field(default=None, ge=1, le=engine.MAX_COMBOS)


@router.post("/plan")
def plan(payload: PlanRequest):
    if len(payload.sections) != len(payload.counts):
        raise HTTPException(status_code=422, detail="one count per section")
    try:
        combos = engine.plan(payload.counts, payload.how_many)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    codes = engine.section_codes(payload.sections)
    return {"total": engine.total_combos(payload.counts), "codes": codes,
            "videos": [{"n": i + 1, "picks": c, "code": engine.combo_code(c, codes),
                        "names": [f"{name} {p + 1}" for name, p in zip(payload.sections, c)]}
                       for i, c in enumerate(combos)]}


class MixClip(BaseModel):
    uri: str
    audio_uri: str | None = None
    original_name: str = "clip.mov"
    duration_sec: float = Field(default=0.0, ge=0.0, le=600.0)


class MixSection(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    clips: list[MixClip] = Field(min_length=1, max_length=engine.MAX_CLIPS_PER_SECTION)


class MixJobRequest(BaseModel):
    user_id: str
    project_id: str                       # the project the clips were uploaded to (becomes the Mix's id)
    product_id: str
    sections: list[MixSection] = Field(min_length=1, max_length=engine.MAX_SECTIONS)
    how_many: int | None = Field(default=None, ge=1, le=engine.MAX_COMBOS)


@router.post("/jobs", status_code=202)
def start(payload: MixJobRequest):
    user_id, mix_id = payload.user_id, payload.project_id
    try:
        get_project(user_id=user_id, project_id=mix_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="project not found") from None
    if store.exists(user_id, mix_id):
        raise HTTPException(status_code=409, detail="this Mix was already started")
    for section in payload.sections:
        for clip in section.clips:
            for uri in filter(None, (clip.uri, clip.audio_uri)):
                try:
                    validate_product_source_uri(uri, project_id=mix_id, user_id=user_id)
                except ValueError as exc:
                    raise HTTPException(status_code=422, detail=str(exc)) from None
    counts = [len(s.clips) for s in payload.sections]
    try:
        combos = engine.plan(counts, payload.how_many)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    data = catalog.load(user_id)
    product = data["products"].get(payload.product_id)
    if not product:
        raise HTTPException(status_code=404, detail="product not found")
    brand = (data["brands"].get(product.get("brand_id") or "") or {}).get("name")
    from cutsell_worker.usage_limits import check_processing_allowance
    allowance = check_processing_allowance(user_id=user_id,
                                           durations_sec=[c.duration_sec for s in payload.sections for c in s.clips])
    if not allowance.allowed:
        raise HTTPException(status_code=402, detail=f"processing denied: {allowance.reason}")

    codes = engine.section_codes([s.name for s in payload.sections])
    sections, clips = [], {}
    for si, section in enumerate(payload.sections):
        items = []
        for ci, clip in enumerate(section.clips):
            clip_id = f"s{si}c{ci}"
            items.append({"clip_id": clip_id, "uri": clip.uri, "audio_uri": clip.audio_uri,
                          "original_name": clip.original_name, "duration_sec": clip.duration_sec})
            clips[clip_id] = {"state": "waiting", "section": section.name, "label": f"{codes[si]}{ci + 1}"}
        sections.append({"name": section.name.strip(), "code": codes[si], "clips": items})
    videos = []
    for n, picks in enumerate(combos, start=1):
        code = engine.combo_code(picks, codes)
        project = create_project(user_id=user_id, title=f"{product['name']} · {code}")
        update_project(user_id=user_id, project_id=project["project_id"], state="processing")
        videos.append({"n": n, "picks": picks, "code": code, "project_id": project["project_id"], "state": "processing"})
    update_project(user_id=user_id, project_id=mix_id, state=catalog.MIX_CONTAINER_STATE,
                   title=f"Mix · {product['name']}")
    catalog.place_mix(user_id, mix_id, {v["project_id"]: v["code"] for v in videos}, product_id=payload.product_id)
    record = {"schema": "cutsell.mix.v1", "mix_id": mix_id, "user_id": user_id, "product_id": payload.product_id,
              "product": product["name"], "brand": brand, "sections": sections, "combos": videos, "clips": clips,
              "state": "processing", "created_at": store.now(), "usage": {}, "timing": {}}
    store.save_new(record)
    job = _enqueue(user_id, mix_id)
    return {**store.summary(record), "job_id": job.job_id}


def _enqueue(user_id: str, mix_id: str):
    from cutsell_worker.queueing import enqueue_mix
    try:
        return enqueue_mix({"user_id": user_id, "mix_id": mix_id})
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


@router.get("")
def recent(user_id: str, limit: int = 10):
    return {"mixes": [store.summary(r) for r in store.recent(user_id, limit=max(1, min(limit, 50)))]}


@router.get("/{mix_id}")
def one(mix_id: str, user_id: str):
    try:
        return store.summary(store.load(user_id, mix_id))
    except KeyError:
        raise HTTPException(status_code=404, detail="mix not found") from None


class RetryRequest(BaseModel):
    user_id: str
    clip_ids: list[str] | None = None


@router.post("/{mix_id}/retry", status_code=202)
def retry(mix_id: str, payload: RetryRequest):
    """Only the failed clips are done again (or the ones named); every clip and video already made is kept."""
    try:
        record = store.load(payload.user_id, mix_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="mix not found") from None
    wanted = set(payload.clip_ids or [cid for cid, c in record["clips"].items() if c.get("state") == "failed"])
    if not wanted:
        return {**store.summary(record), "job_id": None}

    def reset(d):
        for cid in wanted & set(d["clips"]):
            d["clips"][cid] = {k: v for k, v in d["clips"][cid].items() if k in ("section", "label")}
            d["clips"][cid]["state"] = "waiting"
        for c in d["combos"]:
            if c.get("state") in ("waiting_clip", "failed"):
                c["state"] = "processing"
        d["state"] = "processing"
    record = store.update(payload.user_id, mix_id, reset)
    job = _enqueue(payload.user_id, mix_id)
    return {**store.summary(record), "job_id": job.job_id}


def is_mix(user_id: str, project_id: str) -> bool:
    try:
        return store.exists(user_id, project_id)
    except Exception:
        return False
