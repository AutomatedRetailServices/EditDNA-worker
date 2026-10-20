"""Brands and products per creator, and which product each video (project) is about.

Virtual folders only: a project is linked to a product id; no file is copied or moved in storage.
One small JSON document per creator in Redis (optimistic WATCH/MULTI writes):

    {"brands":   {brand_id: {"name", "norm", "created_at"}},
     "products": {product_id: {"brand_id", "name", "norm", "category", "thumb_jpeg_b64", "created_at"}},
     "videos":   {project_id: {"product_id" | None, "source": "ai"|"user"|"album", "suggestion": {...} | None,
                               "updated_at"}}}

Rules:
- Names are matched through `normalize` ("Cerave" = "CeraVe", "CeraVe®" = "cerave", extra spaces ignored), so the
  same brand or product is never created twice.
- What the creator chooses by hand (source "user") always wins: the AI never moves that video again; a later AI
  answer is kept only as a suggestion.
- A video the AI could not place with confidence >= MIN_CONFIDENCE stays without a product ("Uncategorized") with
  the AI's best guess kept as its suggestion ("Suggested from the video").
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .project_store import _redis_client, _scope

MIN_CONFIDENCE = 0.7
_MARKS = re.compile(r"[®™©℠]")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize(name: str | None) -> str:
    """'  CeraVe® ' -> 'cerave'; 'Miss  Dior' -> 'miss dior'. Accents are kept as their base letter."""
    text = unicodedata.normalize("NFKD", _MARKS.sub("", str(name or "")))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = re.sub(r"[^\w\s&+'-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_name(name: str | None) -> str:
    """The display name: trademark marks and extra spaces removed, the creator's capitalization kept."""
    return re.sub(r"\s+", " ", _MARKS.sub("", str(name or ""))).strip()[:80]


def catalog_key(user_id: str) -> str:
    return f"cutsell:v1:catalog:{_scope(user_id)}"


def _empty() -> dict[str, Any]:
    return {"brands": {}, "products": {}, "videos": {}}


def load(user_id: str, *, client=None) -> dict[str, Any]:
    raw = _redis_client(client).get(catalog_key(user_id))
    if not raw:
        return _empty()
    data = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    for part in ("brands", "products", "videos"):
        data.setdefault(part, {})
    return data


def _update(user_id: str, change, *, client=None):
    """Read-modify-write with WATCH, retried on a concurrent write. `change(data)` edits data and returns a value."""
    target = _redis_client(client)
    key = catalog_key(user_id)
    from redis.exceptions import WatchError
    for _ in range(20):
        with target.pipeline() as pipe:
            try:
                pipe.watch(key)
                raw = pipe.get(key)
                data = json.loads(raw.decode() if isinstance(raw, bytes) else raw) if raw else _empty()
                for part in ("brands", "products", "videos"):
                    data.setdefault(part, {})
                value = change(data)
                pipe.multi()
                pipe.set(key, json.dumps(data, ensure_ascii=False))
                pipe.execute()
                return value
            except WatchError:
                continue
    raise RuntimeError("catalog is busy, try again")


def _id(prefix: str, *parts: str) -> str:
    return prefix + "_" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def _ensure_brand(data: dict, name: str) -> str | None:
    norm = normalize(name)
    if not norm:
        return None
    for brand_id, brand in data["brands"].items():
        if brand.get("norm") == norm:
            return brand_id
    brand_id = _id("brd", norm, uuid4().hex)
    data["brands"][brand_id] = {"name": clean_name(name), "norm": norm, "created_at": _now()}
    return brand_id


def _ensure_product(data: dict, brand_id: str | None, name: str, category: str | None = None) -> str | None:
    norm = normalize(name)
    if not norm:
        return None
    for product_id, product in data["products"].items():
        if product.get("brand_id") == brand_id and product.get("norm") == norm:
            if category and not product.get("category"):
                product["category"] = clean_name(category)
            return product_id
    product_id = _id("prd", str(brand_id), norm, uuid4().hex)
    data["products"][product_id] = {"brand_id": brand_id, "name": clean_name(name), "norm": norm,
                                    "category": clean_name(category) if category else None, "created_at": _now()}
    return product_id


def known_names(data: dict) -> list[dict]:
    """What the AI is shown so it reuses the creator's own names: [{"brand", "product"}]."""
    out = []
    for product in data["products"].values():
        brand = data["brands"].get(product.get("brand_id") or "", {})
        out.append({"brand": brand.get("name"), "product": product.get("name")})
    for brand_id, brand in data["brands"].items():
        if not any(p.get("brand_id") == brand_id for p in data["products"].values()):
            out.append({"brand": brand.get("name"), "product": None})
    return out


# ------------------------------------------------------------------------------------------------ AI writes


def record_ai_result(user_id: str, project_id: str, result: dict, *, client=None) -> dict:
    """Store the AI's answer for one video. Confident (>= MIN_CONFIDENCE, brand and product named) -> the video is
    placed in that product, unless the creator already chose one by hand. Otherwise it stays Uncategorized with the
    answer kept as a suggestion. Returns the video's entry."""
    suggestion = {k: result.get(k) for k in ("brand", "product", "category", "confidence", "evidence", "used_photo",
                                             "needs_photo")}
    suggestion["log"] = result.get("log")

    def change(data):
        entry = data["videos"].get(project_id) or {"product_id": None, "source": None}
        entry["suggestion"] = suggestion
        entry["updated_at"] = _now()
        confident = (float(result.get("confidence") or 0) >= MIN_CONFIDENCE
                     and normalize(result.get("brand")) and normalize(result.get("product")))
        if entry.get("source") == "user":
            pass                                        # the creator's choice always wins
        elif confident:
            brand_id = _ensure_brand(data, result["brand"])
            product_id = _ensure_product(data, brand_id, result["product"], result.get("category"))
            entry["product_id"] = product_id
            entry["source"] = result.get("source") or "ai"
            thumb = result.get("thumb_jpeg_b64")
            if thumb and product_id and not data["products"][product_id].get("thumb_jpeg_b64"):
                data["products"][product_id]["thumb_jpeg_b64"] = thumb
        data["videos"][project_id] = entry
        return dict(entry)

    return _update(user_id, change, client=client)


# ------------------------------------------------------------------------------------------------ creator writes


def assign(user_id: str, project_ids: list[str], *, product_id: str | None = None, brand: str | None = None,
           product: str | None = None, category: str | None = None, client=None) -> dict:
    """The creator puts one or several videos in a product (an existing id, or a brand + product name that is
    created or reused). These videos are then never moved by the AI."""
    if not project_ids:
        raise ValueError("at least one video is required")

    def change(data):
        target = product_id
        if target is None:
            brand_id = _ensure_brand(data, brand or "")
            target = _ensure_product(data, brand_id, product or "", category)
            if target is None:
                raise ValueError("a product name is required")
        elif target not in data["products"]:
            raise KeyError("product not found")
        for project_id in project_ids:
            entry = data["videos"].get(project_id) or {}
            entry.update({"product_id": target, "source": "user", "updated_at": _now()})
            data["videos"][project_id] = entry
        return {"product_id": target, "moved": len(project_ids)}

    return _update(user_id, change, client=client)


def create(user_id: str, *, brand: str, product: str | None = None, category: str | None = None, client=None) -> dict:
    def change(data):
        brand_id = _ensure_brand(data, brand)
        if brand_id is None:
            raise ValueError("a brand name is required")
        product_id = _ensure_product(data, brand_id, product, category) if product else None
        return {"brand_id": brand_id, "product_id": product_id}

    return _update(user_id, change, client=client)


def rename(user_id: str, *, kind: str, item_id: str, name: str, client=None) -> dict:
    """Rename a brand or a product. If another brand (or product of the same brand) already has that name, the two
    are merged into the existing one (its videos move with it)."""
    if not normalize(name):
        raise ValueError("a name is required")

    def change(data):
        if kind == "brand":
            if item_id not in data["brands"]:
                raise KeyError("brand not found")
            norm = normalize(name)
            other = next((b for b, v in data["brands"].items() if v.get("norm") == norm and b != item_id), None)
            if other:
                for product in data["products"].values():
                    if product.get("brand_id") == item_id:
                        product["brand_id"] = other
                del data["brands"][item_id]
                _merge_duplicate_products(data, other)
                return {"brand_id": other, "merged": True}
            data["brands"][item_id].update({"name": clean_name(name), "norm": norm})
            return {"brand_id": item_id, "merged": False}
        if kind == "product":
            if item_id not in data["products"]:
                raise KeyError("product not found")
            norm = normalize(name)
            brand_id = data["products"][item_id].get("brand_id")
            other = next((p for p, v in data["products"].items()
                          if v.get("norm") == norm and v.get("brand_id") == brand_id and p != item_id), None)
            if other:
                for entry in data["videos"].values():
                    if entry.get("product_id") == item_id:
                        entry["product_id"] = other
                del data["products"][item_id]
                return {"product_id": other, "merged": True}
            data["products"][item_id].update({"name": clean_name(name), "norm": norm})
            return {"product_id": item_id, "merged": False}
        raise ValueError("kind must be brand or product")

    return _update(user_id, change, client=client)


def _merge_duplicate_products(data: dict, brand_id: str) -> None:
    seen: dict[str, str] = {}
    for product_id in list(data["products"]):
        product = data["products"][product_id]
        if product.get("brand_id") != brand_id:
            continue
        keep = seen.get(product["norm"])
        if keep is None:
            seen[product["norm"]] = product_id
            continue
        for entry in data["videos"].values():
            if entry.get("product_id") == product_id:
                entry["product_id"] = keep
        del data["products"][product_id]


# ------------------------------------------------------------------------------------------------ reads


def state_bucket(state: str | None) -> str:
    """Project state -> the app's bucket."""
    state = str(state or "")
    if state == "draft_ready":
        return "ready"
    # the same states the app shows under Processing (Project.processingStates), plus "queued"
    if state in ("queued", "processing", "uploaded", "preparing", "transcribing", "analyzing", "composing"):
        return "processing"
    return "drafts"


def overview(user_id: str, projects: list[dict], *, client=None) -> dict:
    """Brands -> products with video counts (ready / drafts / processing), and the videos that need a product.
    `projects` = the creator's projects (project_id, state, title) from the project library."""
    data = load(user_id, client=client)
    alive = {str(p.get("project_id")): p for p in projects}
    counts: dict[str, dict[str, int]] = {}
    uncategorized = []
    for project_id, project in alive.items():
        entry = data["videos"].get(project_id) or {}
        product_id = entry.get("product_id")
        if product_id and product_id in data["products"]:
            bucket = counts.setdefault(product_id, {"ready": 0, "drafts": 0, "processing": 0, "videos": 0})
            bucket[state_bucket(project.get("state"))] += 1
            bucket["videos"] += 1
        elif state_bucket(project.get("state")) != "processing":
            uncategorized.append({"project_id": project_id, "title": project.get("title"),
                                  "suggestion": entry.get("suggestion")})
    brands = []
    for brand_id, brand in sorted(data["brands"].items(), key=lambda kv: kv[1].get("name", "").lower()):
        products = []
        for product_id, product in sorted(data["products"].items(), key=lambda kv: kv[1].get("name", "").lower()):
            if product.get("brand_id") != brand_id:
                continue
            c = counts.get(product_id, {"ready": 0, "drafts": 0, "processing": 0, "videos": 0})
            products.append({"product_id": product_id, "name": product["name"], "category": product.get("category"),
                             "has_thumbnail": bool(product.get("thumb_jpeg_b64")), **c})
        if products:
            brands.append({"brand_id": brand_id, "name": brand["name"], "products": products,
                           "videos": sum(p["videos"] for p in products)})
    return {"brands": brands, "uncategorized": uncategorized}


def product_videos(user_id: str, product_id: str, projects: list[dict], *, client=None) -> dict:
    data = load(user_id, client=client)
    product = data["products"].get(product_id)
    if not product:
        raise KeyError("product not found")
    brand = data["brands"].get(product.get("brand_id") or "", {})
    videos = []
    for project in projects:
        project_id = str(project.get("project_id"))
        if (data["videos"].get(project_id) or {}).get("product_id") == product_id:
            videos.append({"project_id": project_id, "title": project.get("title"), "state": project.get("state"),
                           "bucket": state_bucket(project.get("state"))})
    return {"product_id": product_id, "name": product["name"], "category": product.get("category"),
            "brand_id": product.get("brand_id"), "brand": brand.get("name"), "videos": videos}


def thumbnail(user_id: str, product_id: str, *, client=None) -> bytes | None:
    import base64
    product = load(user_id, client=client)["products"].get(product_id) or {}
    raw = product.get("thumb_jpeg_b64")
    return base64.b64decode(raw) if raw else None


def video_entry(user_id: str, project_id: str, *, client=None) -> dict | None:
    return load(user_id, client=client)["videos"].get(project_id)
