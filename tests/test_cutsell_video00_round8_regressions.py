from cutsell_worker.contracts import DraftClip, SemanticRole
from cutsell_worker.round8_retry_reconciliation import (
    restore_clean_retake_after_failed_discard_chain,
    suppress_orphan_failed_open_prefix,
)


def _clip(clip_id, start, end, text, *, selected=True, group=None):
    return DraftClip(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=float(start),
        end=float(end),
        text=text,
        caption_text=text,
        semantic_role=SemanticRole.STORY,
        take_group_id=group,
        selected=selected,
    )


def _diagnostics(decisions, events=()):
    return {
        "hybrid_editorial_chunks": [
            {
                "decisions": [
                    {"clip_id": clip_id, "label": label, "confidence": confidence}
                    for clip_id, label, confidence in decisions
                ]
            }
        ],
        "whole_video_context": {
            "sources": [
                {
                    "source_asset_id": "src",
                    "events": [
                        {
                            "kind": kind,
                            "start": float(start),
                            "end": float(end),
                            "confidence": float(confidence),
                        }
                        for start, end, kind, confidence in events
                    ],
                }
            ]
        },
    }


def test_round8_clean_120_retake_is_restored_after_failed_discard_chain():
    """Exact Round 8 failure: 108 and 115 were discarded but clean 120 was discarded too."""
    context = _clip(
        "context",
        95.58,
        107.48,
        "Al terminar mi contrato cambié de ginecóloga y le pedí todos los test. Ahí me mandó a hacer sonografías.",
    )
    broken = _clip(
        "broken",
        108.56,
        112.34,
        "Ahí fue cuando me mandaron a hacer sonografías de tiroides y otros.",
        selected=False,
    )
    bridge = _clip(
        "bridge",
        115.14,
        117.78,
        "Ahí fue cuando me mandaron",
        selected=False,
    )
    clean = _clip(
        "clean",
        120.11,
        124.15,
        "a hacer sonografías de tiroides y otras sonografías.",
        selected=False,
    )
    next_clip = _clip(
        "next",
        128.16,
        134.22,
        "En la sonografía de tiroides apareció un nódulo sospechoso de tres centímetros.",
    )
    diagnostics = _diagnostics(
        (
            ("broken", "alternate", 0.80),
            ("bridge", "failed", 0.90),
            ("clean", "alternate", 0.78),
        ),
        (
            (112.40, 113.00, "hand_motion_reset_candidate", 0.90),
            (118.00, 119.00, "body_reset_candidate", 0.88),
        ),
    )

    selected, discarded, audit = restore_clean_retake_after_failed_discard_chain(
        (context, next_clip),
        (broken, bridge, clean),
        diagnostics,
    )

    assert tuple(clip.clip_id for clip in selected) == ("context", "clean", "next")
    assert tuple(clip.clip_id for clip in discarded) == ("broken", "bridge")
    assert audit[0]["reason"] == "restore_clean_retake_after_failed_discard_chain"
    assert audit[0]["restored_clip_id"] == "clean"
    assert audit[0]["failed_attempt_clip_id"] == "broken"
    assert audit[0]["failed_bridge_clip_id"] == "bridge"


def test_round8_clean_retake_is_not_restored_when_candidate_itself_is_failed():
    broken = _clip("broken", 10.0, 14.0, "Aquí fue cuando repetí esta frase completa", selected=False)
    bridge = _clip("bridge", 15.0, 17.0, "Aquí fue cuando repetí", selected=False)
    candidate = _clip("candidate", 18.0, 22.0, "repetí esta frase completa", selected=False)
    diagnostics = _diagnostics(
        (
            ("bridge", "failed", 0.92),
            ("candidate", "alternate", 0.78),
            ("candidate", "failed", 0.90),
        ),
        (
            (14.2, 14.7, "hand_motion_reset_candidate", 0.90),
            (17.1, 17.5, "body_reset_candidate", 0.90),
        ),
    )

    selected, discarded, audit = restore_clean_retake_after_failed_discard_chain(
        (),
        (broken, bridge, candidate),
        diagnostics,
    )

    assert selected == ()
    assert tuple(clip.clip_id for clip in discarded) == ("broken", "bridge", "candidate")
    assert audit == ()


def test_round8_orphan_319_open_prefix_is_removed_when_335_continuation_already_lost():
    """Exact Round 8 failure: 295 won and 335 lost, but open failed 319 still survived."""
    prior = _clip(
        "prior",
        295.36,
        314.60,
        (
            "Esta es mi experiencia. Soy la única en mi familia que tiene este tipo de cáncer. "
            "Está comprobado científicamente que solo un 5-10% son de carácter hereditario. "
            "Mayormente son nuestras elecciones de vida, así que cuídate."
        ),
    )
    open_prefix = _clip(
        "open",
        319.38,
        334.24,
        (
            "Soy la primera en mi familia con este tipo de cáncer. Nadie en mi familia tiene un carcinoma papilar "
            "en la tiroides ni sufre de la tiroides. Así que estoy convencida y la ciencia lo avala que solo un 5 -10 % de los"
        ),
    )
    continuation = _clip(
        "continuation",
        335.88,
        346.54,
        "cánceres son hereditarios. Soy la única que tiene este tipo de cáncer.",
        selected=False,
    )
    cta = _clip(
        "cta",
        356.83,
        361.55,
        "Por eso cuídate, alimentate bien, hidrátate y haz ejercicio.",
    )
    diagnostics = _diagnostics((("open", "failed", 0.90),))

    selected, discarded, audit = suppress_orphan_failed_open_prefix(
        (prior, open_prefix, cta),
        (continuation,),
        diagnostics,
    )

    assert tuple(clip.clip_id for clip in selected) == ("prior", "cta")
    assert tuple(clip.clip_id for clip in discarded) == ("continuation", "open")
    assert audit[0]["reason"] == "orphan_failed_open_prefix_yields_to_prior_complete_delivery"
    assert audit[0]["prior_winner_clip_id"] == "prior"
    assert audit[0]["discarded_continuation_clip_id"] == "continuation"


def test_open_failed_prefix_is_not_removed_when_discarded_continuation_is_unrelated():
    prior = _clip("prior", 10.0, 18.0, "Esta historia anterior trata de la tiroides y termina correctamente.")
    prefix = _clip("prefix", 20.0, 26.0, "Ahora quiero hablar de mi trabajo y de los")
    unrelated = _clip("unrelated", 27.0, 31.0, "viajes familiares que hice el año pasado.", selected=False)
    diagnostics = _diagnostics((("prefix", "failed", 0.92),))

    selected, discarded, audit = suppress_orphan_failed_open_prefix(
        (prior, prefix),
        (unrelated,),
        diagnostics,
    )

    assert tuple(clip.clip_id for clip in selected) == ("prior", "prefix")
    assert tuple(clip.clip_id for clip in discarded) == ("unrelated",)
    assert audit == ()
