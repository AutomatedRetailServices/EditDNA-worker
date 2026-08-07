"""Versioned public API contracts for the EditDNA/CutSell V1 product.

These are transport contracts only. Persistence/authentication are deliberately
separate so the worker never becomes an ad-hoc user database.
"""
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

API_VERSION = "v1"
ProjectFlow = Literal["script", "raw_improv"]
ProjectStatus = Literal["draft", "uploading", "processing", "ready", "rendering", "failed", "archived"]
AssetStatus = Literal["pending", "uploading", "uploaded", "processing", "ready", "failed"]


class ProjectCreate(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    flow: ProjectFlow
    language: Literal["en", "es"] = "en"


class ProjectSummary(BaseModel):
    project_id: str
    title: str
    flow: ProjectFlow
    language: Literal["en", "es"]
    status: ProjectStatus
    created_at: str
    updated_at: str


class UploadPartRequest(BaseModel):
    part_number: int = Field(ge=1, le=10000)


class MultipartUploadCreate(BaseModel):
    project_id: str
    filename: str = Field(min_length=1, max_length=255)
    content_type: str = Field(min_length=1, max_length=120)
    size_bytes: int = Field(gt=0)


class MultipartUploadSession(BaseModel):
    asset_id: str
    upload_id: str
    object_key: str
    status: AssetStatus = "uploading"


class CompletedPart(BaseModel):
    part_number: int = Field(ge=1, le=10000)
    etag: str = Field(min_length=1, max_length=300)


class MultipartUploadComplete(BaseModel):
    asset_id: str
    upload_id: str
    parts: List[CompletedPart] = Field(min_length=1)


class ProcessingRequest(BaseModel):
    project_id: str
    asset_ids: List[str] = Field(min_length=1)
    mode: Literal["clean", "human", "blooper"] = "human"
    idempotency_key: str = Field(min_length=8, max_length=200)
    use_semantic_v2: bool = True
    use_take_judge_v2: bool = True


class DraftSwapRequest(BaseModel):
    selected_clip_id: str
    replacement_clip_id: str


class DraftRemoveRequest(BaseModel):
    clip_id: str


class DraftRestoreRequest(BaseModel):
    clip_id: str
    position: Optional[int] = Field(default=None, ge=0)


class DraftReorderRequest(BaseModel):
    ordered_clip_ids: List[str] = Field(min_length=1)


class CaptionEdit(BaseModel):
    clip_id: str
    text: str = Field(max_length=2000)


class CaptionPatchRequest(BaseModel):
    edits: List[CaptionEdit] = Field(min_length=1)


class RenderRequestV1(BaseModel):
    project_id: str
    draft_version: int = Field(ge=1)
    kind: Literal["preview", "final"] = "preview"
    captions: bool = True
    idempotency_key: str = Field(min_length=8, max_length=200)


class ScriptCard(BaseModel):
    card_id: str
    slot: Literal["HOOK", "PROBLEM", "FEATURES", "BENEFITS", "PROOF", "STORY", "CTA", "OTHER"]
    text: str
    recording_note: Optional[str] = None


class FlowAScriptRequest(BaseModel):
    project_id: str
    product_url: Optional[str] = None
    product_name: Optional[str] = None
    product_context: Optional[str] = None
    language: Literal["en", "es"] = "en"
    style: Literal["talking_head", "testimonial", "storytelling", "voiceover", "faceless", "skit"] = "talking_head"
    user_script: Optional[str] = None
