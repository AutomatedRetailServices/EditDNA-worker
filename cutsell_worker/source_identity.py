"""Deterministic source/clip identity helpers."""
from __future__ import annotations

import hashlib


def stable_source_id(project_id: str, source_order: int, original_name: str) -> str:
    raw = f"{project_id}|{source_order}|{original_name}".encode("utf-8")
    return "src_" + hashlib.sha256(raw).hexdigest()[:20]


def stable_clip_id(source_asset_id: str, start: float, end: float, text: str) -> str:
    raw = f"{source_asset_id}|{start:.3f}|{end:.3f}|{text.strip()}".encode("utf-8")
    return "clip_" + hashlib.sha256(raw).hexdigest()[:20]


def assert_same_source(source_asset_id: str, *candidate_source_ids: str) -> None:
    if any(value != source_asset_id for value in candidate_source_ids):
        raise ValueError("cross-source merge is forbidden")
