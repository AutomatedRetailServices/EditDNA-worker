from cutsell_worker.contracts import DraftClip, SemanticRole, Word
from cutsell_worker.final_draft_retry_integrity import (
    promote_group_peer_over_failed_selected_chain,
)
from cutsell_worker.selected_failed_bridge_integrity import (
    suppress_selected_attempt_before_failed_bridge_v2,
)
from cutsell_worker.temporal_word_boundary_integrity import _preserve_current_word_end


def _clip(cid, start, end, text, *, group=None, selected=True):
    return DraftClip(
        clip_id=cid,
        source_asset_id="src",
        source_order=0,
        start=float(start),
        end=float(end),
        text=text,
        caption_text=text,
        semantic_role=SemanticRole.OTHER,
        take_group_id=group,
        selected=selected,
    )


def _diag(*decisions, events=()):
    return {
        "hybrid_editorial_chunks": [
            {
                "decisions": [
                    {"clip_id": cid, "label": label, "confidence": conf}
                    for cid, label, conf in decisions
                ]
            }
        ],
        "whole_video_context": {
            "sources": [
                {
                    "source_asset_id": "src",
                    "events": list(events),
                }
            ]
        },
    }


def test_round7_failed_selected_prefix_and_winner_suffix_promote_prior_group_peer():
    prior = _clip(
        "prior",
        295.33,
        313.13,
        (
            "Esta es mi experiencia. Soy la única en mi familia que tiene este tipo de cáncer. "
            "Por eso no creo y está comprobado científicamente que los cánceres son hereditarios. "
            "Más bien, solo un 5-10% son de carácter hereditario. Mayormente son nuestras "
            "elecciones de vida. Así que cuida."
        ),
        group="closing_group",
        selected=False,
    )
    failed_prefix = _clip(
        "failed_prefix",
        319.37,
        334.25,
        (
            "Soy la primera en mi familia con este tipo de cáncer. Nadie en mi familia tiene un "
            "carcinoma papilar en la tiroides ni sufre de la tiroides. Así que estoy convencida "
            "y la ciencia lo avala que solo un 5 -10 % de los"
        ),
        group="prefix_group",
    )
    continuation = _clip(
        "continuation",
        335.89,
        346.53,
        "cánceres son hereditarios. Soy la única en mi familia que tiene este tipo de cáncer.",
        group="closing_group",
    )
    diag = _diag(
        ("failed_prefix", "failed", 0.90),
        ("continuation", "winner", 0.95),
        ("prior", "winner", 0.95),
    )

    selected, alternates, discarded, audit = promote_group_peer_over_failed_selected_chain(
        (failed_prefix, continuation),
        (prior,),
        (),
        diag,
    )

    assert tuple(x.clip_id for x in selected) == ("prior",)
    assert selected[0].selected is True
    assert alternates == ()
    assert {x.clip_id for x in discarded} == {"failed_prefix", "continuation"}
    assert audit[0]["reason"] == "promote_prior_group_peer_over_failed_selected_retry_chain"


def test_failed_prefix_group_promotion_fails_open_without_open_ending():
    prior = _clip("prior", 10, 20, "una historia completa diferente.", group="g", selected=False)
    prefix = _clip("prefix", 25, 32, "esta frase ya terminó correctamente.", group="x")
    continuation = _clip("cont", 33, 38, "otra frase distinta y completa.", group="g")
    diag = _diag(("prefix", "failed", 0.95), ("cont", "winner", 0.95))

    selected, alternates, discarded, audit = promote_group_peer_over_failed_selected_chain(
        (prefix, continuation), (prior,), (), diag
    )

    assert tuple(x.clip_id for x in selected) == ("prefix", "cont")
    assert tuple(x.clip_id for x in alternates) == ("prior",)
    assert discarded == ()
    assert audit == ()


def test_round7_failed_repeated_opening_bridge_removes_earlier_selected_attempt():
    earlier = _clip(
        "earlier",
        108.90,
        112.34,
        "Ahí fue cuando me mandaron a hacer sonografías de tiroides y otros.",
    )
    bridge = _clip(
        "bridge",
        115.12,
        118.94,
        "Ahí fue cuando me mandaron",
        selected=False,
    )
    later = _clip(
        "later",
        121.27,
        124.57,
        "a hacer sonografía de tiroides y otras sonografías.",
    )
    events = (
        {"kind": "hand_motion_reset_candidate", "start": 113.88, "end": 113.94, "confidence": 0.89},
        {"kind": "hand_motion_reset_candidate", "start": 114.14, "end": 114.21, "confidence": 1.00},
        {"kind": "hand_motion_reset_candidate", "start": 116.14, "end": 116.21, "confidence": 1.00},
    )
    diag = _diag(("bridge", "failed", 0.92), events=events)

    selected, discarded, audit = suppress_selected_attempt_before_failed_bridge_v2(
        (earlier, later), (bridge,), diag
    )

    assert tuple(x.clip_id for x in selected) == ("later",)
    assert {x.clip_id for x in discarded} == {"bridge", "earlier"}
    assert audit[0]["reason"] == "selected_attempt_yields_across_failed_repeated_opening_bridge"
    assert audit[0]["opening_repeat_words"] >= 4
    assert audit[0]["reset_event_count"] >= 2


def test_failed_bridge_does_not_remove_neighboring_unique_story_without_retry_overlap():
    earlier = _clip("earlier", 10, 15, "mi historia sobre mi trabajo en cruceros.")
    bridge = _clip("bridge", 17, 19, "ahí fue cuando me mandaron", selected=False)
    later = _clip("later", 21, 26, "me hicieron una sonografía de tiroides.")
    events = (
        {"kind": "hand_motion_reset_candidate", "start": 16, "end": 16.1, "confidence": 1.0},
        {"kind": "hand_motion_reset_candidate", "start": 20, "end": 20.1, "confidence": 1.0},
    )
    diag = _diag(("bridge", "failed", 0.95), events=events)

    selected, discarded, audit = suppress_selected_attempt_before_failed_bridge_v2(
        (earlier, later), (bridge,), diag
    )

    assert tuple(x.clip_id for x in selected) == ("earlier", "later")
    assert tuple(x.clip_id for x in discarded) == ("bridge",)
    assert audit == ()


def test_round7_trim_boundary_inside_final_word_preserves_that_word():
    words = (
        Word("hidrátate", 359.70, 360.20, 0.97),
        Word("y", 360.25, 360.40, 0.97),
        Word("haz", 360.45, 360.80, 0.97),
        Word("ejercicio", 361.02, 361.48, 0.97),
    )

    boundary, snapped = _preserve_current_word_end(words, 356.22, 361.40)

    assert snapped is True
    assert boundary == 361.48


def test_end_boundary_between_words_is_not_moved():
    words = (
        Word("haz", 10.0, 10.4, 0.97),
        Word("ejercicio", 10.8, 11.4, 0.97),
    )

    boundary, snapped = _preserve_current_word_end(words, 9.0, 10.6)

    assert snapped is False
    assert boundary == 10.6
