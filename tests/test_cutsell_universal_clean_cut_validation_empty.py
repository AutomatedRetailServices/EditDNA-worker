from types import SimpleNamespace

from cutsell_worker.universal_clean_cut_validation import _render_validation_preview


def test_empty_draft_skips_preview_without_crashing():
    draft = SimpleNamespace(selected=())

    preview_path, reason, qc_result = _render_validation_preview(
        draft,
        {},
        preview_output="preview.mp4",
        preview_captions=False,
    )

    assert preview_path is None
    assert reason == "empty_draft"
    assert qc_result is None


def test_no_preview_request_needs_no_empty_draft_warning():
    draft = SimpleNamespace(selected=())

    preview_path, reason, qc_result = _render_validation_preview(
        draft,
        {},
        preview_output=None,
        preview_captions=False,
    )

    assert preview_path is None
    assert reason is None
    assert qc_result is None


def test_freeze_blocked_draft_skips_render_without_calling_render_qc():
    # D-035: "if semantic validation fails: no render." A freeze-blocked
    # draft never reaches Boundary/Render upstream (universal_clean_cut.py
    # skips freeze/Boundary entirely) -- this harness must not render it
    # either, even if it happens to still carry `selected` clips.
    draft = SimpleNamespace(selected=(object(),))

    preview_path, reason, qc_result = _render_validation_preview(
        draft,
        {},
        preview_output="preview.mp4",
        preview_captions=False,
        freeze_blocked=True,
    )

    assert preview_path is None
    assert reason == "freeze_blocked_no_render"
    assert qc_result is None
