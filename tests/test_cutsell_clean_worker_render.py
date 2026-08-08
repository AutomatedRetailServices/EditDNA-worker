from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION, SemanticRole
from cutsell_worker.render_plan import build_render_plan


def _draft():
    return DraftTimeline(
        schema_version=SCHEMA_VERSION,
        project_id="p1",
        strategy=EditStrategy.MIXED,
        selected=(
            DraftClip(
                clip_id="c1",
                source_asset_id="s1",
                source_order=0,
                start=1.0,
                end=2.5,
                text="hello",
                caption_text="hello",
                semantic_role=SemanticRole.HOOK,
            ),
            DraftClip(
                clip_id="c2",
                source_asset_id="s2",
                source_order=1,
                start=3.0,
                end=4.0,
                text="world",
                caption_text="world",
            ),
        ),
        alternates=(),
        discarded=(),
    )


def test_render_plan_keeps_each_clip_bound_to_its_source():
    plan = build_render_plan(_draft(), {"s1": "/tmp/one.mov", "s2": "/tmp/two.mov"})
    assert [(item.clip_id, item.source_path) for item in plan] == [
        ("c1", "/tmp/one.mov"),
        ("c2", "/tmp/two.mov"),
    ]


def test_render_plan_rejects_missing_source_path():
    try:
        build_render_plan(_draft(), {"s1": "/tmp/one.mov"})
    except ValueError as exc:
        assert "missing source path" in str(exc)
    else:
        raise AssertionError("expected missing source path rejection")
