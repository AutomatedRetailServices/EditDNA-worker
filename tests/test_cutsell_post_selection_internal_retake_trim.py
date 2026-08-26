from cutsell_worker.contracts import DraftClip, SemanticRole, Word
from cutsell_worker.post_selection_internal_retake_trim import trim_selected_internal_retakes


def _words(tokens):
    out = []
    t = 0.0
    for token in tokens:
        out.append(Word(token, t, t + 0.2))
        t += 0.3
    return tuple(out)


def _clip(tokens):
    words = _words(tokens)
    return DraftClip(
        clip_id="clip-a",
        source_asset_id="src",
        source_order=0,
        start=0.0,
        end=words[-1].end + 0.2,
        text=" ".join(tokens),
        caption_text=" ".join(tokens),
        words=words,
        semantic_role=SemanticRole.STORY,
        selected=True,
    )


def _diagnostics(restart_time):
    return {
        "whole_video_context": {
            "sources": [{
                "source_asset_id": "src",
                "events": [
                    {"kind": "hand_motion_reset_candidate", "start": restart_time - 0.1, "end": restart_time + 0.1, "confidence": 0.97},
                    {"kind": "body_reset_candidate", "start": restart_time - 0.1, "end": restart_time + 0.1, "confidence": 0.96},
                    {"kind": "facial_expression_shift_candidate", "start": restart_time - 0.1, "end": restart_time + 0.1, "confidence": 0.88},
                ],
            }]
        }
    }


def test_covered_internal_attempt_yields_to_later_clean_retake():
    tokens = [
        "intro", "único", "antes",
        "me", "hicieron", "una", "prueba", "de", "tiroides",
        "me", "hicieron", "una", "prueba", "de", "tiroides",
        "y", "salió", "todo", "normal", "después",
    ]
    clip = _clip(tokens)
    restart_time = clip.words[9].start
    selected, audit = trim_selected_internal_retakes((clip,), _diagnostics(restart_time))

    assert len(audit) == 1
    assert audit[0]["reason"] == "earlier_internal_attempt_covered_by_later_clean_retake"
    assert len(selected) == 2
    assert [w.text for w in selected[0].words] == ["intro", "único", "antes"]
    assert [w.text for w in selected[1].words[:6]] == ["me", "hicieron", "una", "prueba", "de", "tiroides"]


def test_unique_numeric_fact_in_earlier_attempt_fails_open():
    tokens = [
        "intro", "antes",
        "me", "hicieron", "una", "prueba", "de", "tiroides", "3",
        "me", "hicieron", "una", "prueba", "de", "tiroides",
        "y", "salió", "normal", "después",
    ]
    clip = _clip(tokens)
    restart_time = clip.words[9].start
    selected, audit = trim_selected_internal_retakes((clip,), _diagnostics(restart_time))

    assert audit == ()
    assert len(selected) == 1
    assert selected[0].clip_id == "clip-a"


def test_unique_negation_in_earlier_attempt_fails_open():
    tokens = [
        "intro", "antes",
        "no", "me", "hicieron", "una", "prueba", "de", "tiroides",
        "me", "hicieron", "una", "prueba", "de", "tiroides",
        "y", "salió", "normal", "después",
    ]
    clip = _clip(tokens)
    restart_time = clip.words[9].start
    selected, audit = trim_selected_internal_retakes((clip,), _diagnostics(restart_time))

    assert audit == ()
    assert len(selected) == 1
