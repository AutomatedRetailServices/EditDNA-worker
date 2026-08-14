from types import SimpleNamespace

from cutsell_worker.universal_clean_cut_validation import _render_validation_preview


def test_empty_draft_skips_preview_without_crashing():
    draft = SimpleNamespace(selected=())

    preview_path, reason = _render_validation_preview(
        draft,
        {},
        preview_output="preview.mp4",
        preview_captions=False,
    )

    assert preview_path is None
    assert reason == "empty_draft"


def test_no_preview_request_needs_no_empty_draft_warning():
    draft = SimpleNamespace(selected=())

    preview_path, reason = _render_validation_preview(
        draft,
        {},
        preview_output=None,
        preview_captions=False,
    )

    assert preview_path is None
    assert reason is None
