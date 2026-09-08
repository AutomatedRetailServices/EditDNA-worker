"""D-136 (D-127 Phase 2) -- concrete OpenAI-backed `MultimodalBestTakeArbiter`.

docs/CUTSELL_DECISIONS.md D-136, docs/CUTSELL_MULTIMODAL_FALLBACK_ARBITER_
FORENSIC_D127.md Sections 17/21 (Phase 2). Bounded, OFFLINE-EVAL-ONLY
concrete implementation of the `MultimodalBestTakeArbiter` Protocol
(`multimodal_besttake_arbiter.py`) -- mirrors `visual_openai.
OpenAIVisualProvider`'s exact provider pattern (same `OpenAI().responses.
create(...)` call shape, same `input_text`/`input_image` content blocks,
same `client_factory` injection point every `*_openai.py` provider in this
repo already uses for tests). The Protocol interface itself
(`arbitrate(request) -> response`) is UNCHANGED -- no concrete
incompatibility was found that would require touching it.

THIS CLASS IS NEVER IMPORTED BY THE LIVE PIPELINE. `pipeline.py` imports
only `detect_class_b_trigger`/`fallback_trigger_diagnostics` from
`multimodal_besttake_fallback.py`; nothing in `pipeline.py`, `flow_b.py`,
or any production call path imports this module, `multimodal_besttake_
arbiter.py`, or `NullMultimodalBestTakeArbiter` -- confirmed structurally
by `tests/test_cutsell_d136_multimodal_besttake_openai_phase2.py::
test_no_live_pipeline_import_or_instantiation`.

AUDIO PERCEPTION: NOT AVAILABLE. This provider sends TEXT (transcript,
timing, CASE B aggregate evidence, semantic/DeliveryScorer summaries) and
sampled IMAGE frames only -- exactly like the existing `OpenAIVisualProvider`
it mirrors. No audio bytes are ever constructed, read, encoded, or sent to
the provider anywhere in this module. This task never claims the arbiter
"hears" a take, only that it "sees the sampled frames and reads the
transcript/metadata."
"""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .multimodal_besttake_arbiter import (
    MultimodalBestTakeRequest,
    MultimodalBestTakeResponse,
)
from .openai_json import parse_json_object

#: D-136 PROMPT CONTRACT (directive-required 8 reasoning steps, general --
#: never Video00/pimples-specific, never a Human Gold/Cut.ai reference leak).
INSTRUCTION = (
    "You are a bounded BestTake arbiter comparing a small number of candidate "
    "takes of the same short marketing/UGC video segment. For each candidate you "
    "are given its transcript, timing, structured metadata, and one or more "
    "sampled visual frames spanning the take (not the full video). Reason as "
    "follows, in order: "
    "(1) every supplied candidate has ALREADY passed a meaning-sufficiency check "
    "upstream -- do not re-judge whether its spoken content is adequate, only "
    "compare delivery usability; "
    "(2) compare the candidates' delivery/performance quality, not their content; "
    "(3) consider any visible fumble, reset, or camera-disengagement moment; "
    "(4) consider performance continuity across the whole take, not one frame; "
    "(5) do not reward maximum motion or energy for its own sake; "
    "(6) consider how editable each candidate is -- a take with a clean core and a "
    "trimmable entry or exit issue is not the same as a take that is unusable "
    "throughout; "
    "(7) do not penalize a removable ENTRY- or EXIT-only defect as a whole-take "
    "failure -- prefer GOOD_TAKE_TRIM_ENTRY or GOOD_TAKE_TRIM_EXIT for that shape; "
    "(8) choose exactly one supplied candidate id as BEST_TAKE, or EQUIVALENT if "
    "genuinely tied, or UNCERTAIN if the evidence does not support a confident "
    "choice. Never name a candidate id that was not supplied to you. "
    'Return JSON only, no prose: {"outcome":"BEST_TAKE|EQUIVALENT|'
    'GOOD_TAKE_TRIM_ENTRY|GOOD_TAKE_TRIM_EXIT|UNCERTAIN",'
    '"best_take_candidate_id":"<id or null>","confidence":0.0,'
    '"reason":"<one short sentence>"}.'
)


@dataclass
class OpenAIMultimodalBestTakeArbiter:
    """Concrete, OFFLINE-EVAL-ONLY `MultimodalBestTakeArbiter` implementation.

    `model` defaults to a vision-capable OpenAI model (matching `visual_
    openai.OpenAIVisualProvider`'s own default). `client_factory` mirrors
    every other `*_openai.py` provider in this repo for test injection --
    the eval harness's own bounded live run leaves it `None` (real
    `openai.OpenAI()`, reading `OPENAI_API_KEY` from the environment);
    every targeted test always injects a fake client.

    After a successful `.arbitrate(...)` call, `self.last_call_metadata`
    holds a small dict (`latency_ms`, `request_candidate_count`,
    `frame_count`, `usage`) for the eval harness's own cost/latency
    observability -- kept OUTSIDE the `MultimodalBestTakeResponse` return
    value so the Protocol/dataclass interface stays byte-for-byte
    unchanged (per this task's "do not modify the interface contract
    unless a concrete incompatibility is proven" instruction)."""

    model: str = "gpt-4o-mini"
    client_factory: Callable[[], object] | None = None
    last_call_metadata: dict | None = None

    def _client(self):
        if self.client_factory is not None:
            return self.client_factory()
        from openai import OpenAI
        return OpenAI()

    @staticmethod
    def _image_url(path: str) -> str:
        raw = Path(path).read_bytes()
        return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")

    def _content_for(self, request: MultimodalBestTakeRequest) -> list[dict]:
        content: list[dict] = [{"type": "input_text", "text": INSTRUCTION}]
        content.append({
            "type": "input_text",
            "text": json.dumps({
                "family_id": request.family_id,
                "proposition_context": request.proposition_context,
                "candidate_ids": [f.candidate_id for f in request.finalists],
            }, ensure_ascii=False),
        })
        for finalist in request.finalists:
            content.append({
                "type": "input_text",
                "text": json.dumps({
                    "candidate_id": finalist.candidate_id,
                    "transcript": finalist.transcript_text,
                    "source_asset_id": finalist.source_asset_id,
                    "source_start_sec": finalist.source_start,
                    "source_end_sec": finalist.source_end,
                    # D-136: already-computed upstream fact, never re-judged.
                    "meaning_sufficient": finalist.meaning_sufficient,
                    "semantic_label": finalist.semantic_label,
                    "semantic_confidence": finalist.semantic_confidence,
                    "deliveryscore_summary": finalist.deliveryscore_summary,
                    "boundary_editability_note": finalist.boundary_editability_note,
                    # D-122 CASE B factual aggregates only -- never a score.
                    "case_b_delivery_event_count": finalist.case_b_evidence.delivery_event_count,
                    "case_b_delivery_event_duration_total": finalist.case_b_evidence.delivery_event_duration_total,
                    "case_b_count_by_kind": dict(finalist.case_b_evidence.count_by_kind),
                    "case_b_duration_by_kind": dict(finalist.case_b_evidence.duration_by_kind),
                    "case_b_event_density": finalist.case_b_evidence.event_density,
                    "sampled_frame_count": len(finalist.sampled_frame_references),
                }, ensure_ascii=False),
            })
            for frame_path in finalist.sampled_frame_references:
                content.append({
                    "type": "input_image",
                    "image_url": self._image_url(frame_path),
                    "detail": "low",
                })
        return content

    def arbitrate(self, request: MultimodalBestTakeRequest) -> MultimodalBestTakeResponse:
        content = self._content_for(request)
        client = self._client()
        started = time.monotonic()
        response = client.responses.create(model=self.model, input=[{"role": "user", "content": content}])
        latency_ms = round((time.monotonic() - started) * 1000, 1)

        usage = getattr(response, "usage", None)
        if usage is not None and not isinstance(usage, dict):
            usage = getattr(usage, "model_dump", lambda: None)() or getattr(usage, "__dict__", None)
        self.last_call_metadata = {
            "latency_ms": latency_ms,
            "request_candidate_count": len(request.finalists),
            "frame_count": sum(len(f.sampled_frame_references) for f in request.finalists),
            "usage": usage,
        }

        data = parse_json_object(response.output_text)
        outcome = str(data.get("outcome") or "")
        raw_candidate_id = data.get("best_take_candidate_id")
        confidence_value = data.get("confidence")
        reason = str(data.get("reason") or "")

        return MultimodalBestTakeResponse(
            family_id=request.family_id,
            outcome=outcome,
            best_take_candidate_id=(str(raw_candidate_id) if raw_candidate_id else None),
            confidence=float(confidence_value) if confidence_value is not None else 0.0,
            reason=reason,
            provider="openai",
            model=self.model,
            requested=True,
            available=True,
        )
