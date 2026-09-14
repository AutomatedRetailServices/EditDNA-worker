import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

import cutsell_app.feedback_routes as routes
import cutsell_app.main as api
import cutsell_worker.feedback as feedback


class FakeS3:
    def __init__(self):
        self.objects = []
    def put_object(self, **kwargs):
        self.objects.append(kwargs)
        return {"ETag": "x"}


def _draft():
    return {
        "schema_version": "cutsell.v1",
        "project_id": "p1",
        "strategy": "demo_product_led",
        "selected": [{
            "clip_id": "c1",
            "source_asset_id": "s1",
            "start": 0.0,
            "end": 1.0,
            "semantic_role": "HOOK",
            "take_group_id": "g1",
        }],
        "alternates": [{
            "clip_id": "c2",
            "source_asset_id": "s1",
            "start": 1.2,
            "end": 2.2,
            "semantic_role": "HOOK",
            "take_group_id": "g1",
        }],
        "discarded": [],
        "diagnostics": {
            "take_judge_status_counts": {"applied": 1},
            "take_judge_fallback_reasons": {},
        },
    }


def test_feedback_event_keeps_evaluation_context_without_raw_media(monkeypatch):
    monkeypatch.setattr(
        feedback,
        "load_runtime_config",
        lambda: SimpleNamespace(
            asr_model="medium",
            semantic_model="gpt-4o-mini",
            visual_model="gpt-4o-mini",
            take_judge_model="gpt-4o-mini",
            s3_bucket="bucket",
            aws_region="us-east-1",
        ),
    )
    event = feedback.build_feedback_event(
        user_id="u1",
        project_id="p1",
        rating="bad",
        draft=_draft(),
        reason="wrong take",
        clip_id="c1",
        time_sec=0.4,
        processing_metrics={"elapsed_ms": 1234},
    )
    assert event["rating"] == "bad"
    assert event["marker"] == {"clip_id": "c1", "time_sec": 0.4}
    assert event["selected"][0]["source_asset_id"] == "s1"
    assert event["models"]["take_judge"] == "gpt-4o-mini"
    assert "uri" not in event["selected"][0]


def test_feedback_rejects_marker_from_another_draft():
    try:
        feedback.build_feedback_event(
            user_id="u1", project_id="p1", rating="bad", draft=_draft(), clip_id="other"
        )
    except ValueError as exc:
        assert "not part of this draft" in str(exc)
    else:
        raise AssertionError("foreign clip marker must be rejected")


def test_feedback_storage_is_scoped_immutable_json(monkeypatch):
    monkeypatch.setattr(
        feedback,
        "load_runtime_config",
        lambda: SimpleNamespace(
            asr_model="medium", semantic_model="gpt-4o-mini", visual_model="gpt-4o-mini",
            take_judge_model="gpt-4o-mini", s3_bucket="bucket", aws_region="us-east-1"
        ),
    )
    event = feedback.build_feedback_event(user_id="u1", project_id="p1", rating="good", draft=_draft())
    s3 = FakeS3()
    result = feedback.store_feedback_event(event, user_id="u1", project_id="p1", client=s3)
    assert result["stored"] is True
    assert result["feedback_uri"].startswith("s3://bucket/cutsell/feedback/")
    obj = s3.objects[0]
    assert obj["ContentType"] == "application/json"
    assert obj["ServerSideEncryption"] == "AES256"
    stored = json.loads(obj["Body"].decode("utf-8"))
    assert stored["feedback_id"] == event["feedback_id"]


def test_feedback_api_accepts_good_edit(monkeypatch):
    monkeypatch.setattr(
        routes,
        "build_feedback_event",
        lambda **kwargs: {"feedback_id": "fb_1", "project_id": kwargs["project_id"]},
    )
    monkeypatch.setattr(
        routes,
        "store_feedback_event",
        lambda event, **kwargs: {"feedback_id": event["feedback_id"], "stored": True},
    )
    response = TestClient(api.app).post("/v1/projects/p1/feedback", json={
        "user_id": "u1",
        "rating": "good",
        "draft": _draft(),
    })
    assert response.status_code == 200
    assert response.json() == {"feedback_id": "fb_1", "stored": True}
