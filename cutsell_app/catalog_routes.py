"""Edits · Products: the creator's brands and products, the videos in each, and moving videos between them.

Virtual folders: a video (project) is linked to a product id; nothing is copied in storage. A video the creator moved
by hand is never moved again by the AI (cutsell_worker.product_catalog)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field

from cutsell_worker import product_catalog as catalog
from cutsell_worker.project_store import list_projects

router = APIRouter(prefix="/v1/catalog", tags=["catalog"])


def _projects(user_id: str) -> list[dict]:
    return list_projects(user_id=user_id, limit=100)


def _guard(call):
    try:
        return call()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'")) from None
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


@router.get("")
def get_catalog(user_id: str):
    """Brands -> products with counts (ready / drafts / processing / videos), the videos that need a product (with
    the AI's suggestion), and every video's product id (for the Edits filters)."""
    def build():
        projects = _projects(user_id)
        out = catalog.overview(user_id, projects)
        data = catalog.load(user_id)
        out["videos"] = {pid: {"product_id": v.get("product_id"), "source": v.get("source")}
                         for pid, v in data["videos"].items()}
        return out
    return _guard(build)


@router.get("/products/{product_id}")
def get_product(product_id: str, user_id: str):
    return _guard(lambda: catalog.product_videos(user_id, product_id, _projects(user_id)))


@router.get("/products/{product_id}/thumbnail")
def get_product_thumbnail(product_id: str, user_id: str):
    image = _guard(lambda: catalog.thumbnail(user_id, product_id))
    if not image:
        raise HTTPException(status_code=404, detail="no thumbnail")
    return Response(content=image, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


class AssignRequest(BaseModel):
    user_id: str
    project_ids: list[str] = Field(min_length=1, max_length=200)
    product_id: str | None = None
    brand: str | None = None
    product: str | None = None
    category: str | None = None


@router.post("/assign")
def assign_videos(payload: AssignRequest):
    """Put one or several videos in a product: an existing product_id, or brand + product (created or reused)."""
    owned = {str(p.get("project_id")) for p in _guard(lambda: _projects(payload.user_id))}
    if any(pid not in owned for pid in payload.project_ids):
        raise HTTPException(status_code=404, detail="video not found")
    return _guard(lambda: catalog.assign(payload.user_id, payload.project_ids, product_id=payload.product_id,
                                         brand=payload.brand, product=payload.product, category=payload.category))


class CreateRequest(BaseModel):
    user_id: str
    brand: str = Field(min_length=1, max_length=80)
    product: str | None = Field(default=None, max_length=80)
    category: str | None = Field(default=None, max_length=40)


@router.post("/products")
def create_product(payload: CreateRequest):
    return _guard(lambda: catalog.create(payload.user_id, brand=payload.brand, product=payload.product,
                                         category=payload.category))


class RenameRequest(BaseModel):
    user_id: str
    kind: str = Field(pattern="^(brand|product)$")
    id: str
    name: str = Field(min_length=1, max_length=80)


@router.post("/rename")
def rename_item(payload: RenameRequest):
    return _guard(lambda: catalog.rename(payload.user_id, kind=payload.kind, item_id=payload.id, name=payload.name))
