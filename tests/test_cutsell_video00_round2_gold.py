from cutsell_worker.contracts import CandidateTake, MediaSignals, Word
from cutsell_worker.delivery_edge_trim import trim_delivery_edge_slack
from cutsell_worker.internal_retake_winner import prefer_internal_clean_retakes
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


def _words(tokens, start=0.0, step=0.22):
    cursor = float(start)
    out = []
    for token in tokens:
        out.append(Word(token, cursor, cursor + step, 0.97))
        cursor += step
    return tuple(out)


def _context(*events):
    return WholeVideoContext(
        sources=(
            SourceVideoContext(
                source_asset_id="src",
                summary="raw talking head with retries",
                dominant_style="talking_head",
                creator_intent="tell personal story naturally",
                events=tuple(events),
            ),
        ),
        status=ProviderStatus("test", True, True, "applied"),
    )


def test_video00_micro_visual_reset_after_completed_sentence_is_trimmed():
    words = _words(("esta", "frase", "ya", "termino"), start=0.0, step=0.2)
    take = CandidateTake(
        clip_id="sentence",
        source_asset_id="src",
        source_order=0,
        start=0.0,
        end=0.98,
        text="esta frase ya termino",
        words=words,
        signals=MediaSignals("src", 0.0, 0.98),
        complete_idea=True,
    )
    # Last spoken word ends at 0.80. The remaining 180 ms is a strong visible body
    # reset/mueca: human Gold says this should not survive in a talking-head cut.
    context = _context(
        TemporalEvent("src", 0.80, 0.96, "body_reset_candidate", 0.95, "creator visibly resets after sentence"),
    )

    trimmed, diagnostics = trim_delivery_edge_slack((take,), context)

    assert len(trimmed) == 1
    assert trimmed[0].end == words[-1].end
    assert diagnostics
    assert diagnostics[0]["actions"][0]["talking_head_micro_edge"] is True


def test_video00_broken_internal_attempt_yields_to_later_clean_retake():
    tokens = (
        "nunca", "se", "nos", "ocurrio", "hacer", "chequeo", "tiroides", "sonografia",
        "porque", "siempre", "eh", "no",  # broken first attempt/fumble
        "nunca", "se", "nos", "ocurrio", "hacer", "chequeo", "tiroides", "sonografia",
        "porque", "siempre", "salio", "perfectamente",
    )
    words = _words(tokens, start=10.0, step=0.24)
    take = CandidateTake(
        clip_id="merged_retry",
        source_asset_id="src",
        source_order=0,
        start=10.0,
        end=words[-1].end + 0.1,
        text=" ".join(tokens),
        words=words,
        signals=MediaSignals("src", 10.0, words[-1].end + 0.1),
        complete_idea=True,
    )
    # Retry setup sits between the broken delivery and the clean redo.
    retry_start = words[12].start - 0.08
    context = _context(
        TemporalEvent("src", retry_start, words[12].start + 0.06, "retry_setup", 0.94, "creator restarts same idea"),
    )

    resolved, diagnostics = prefer_internal_clean_retakes((take,), context)

    assert len(resolved) == 1
    assert resolved[0].start == words[12].start
    assert resolved[0].text.startswith("nunca se nos ocurrio hacer chequeo")
    assert "eh no nunca" not in resolved[0].text
    assert diagnostics[0]["reason"] == "internal_broken_attempt_yields_to_clean_retake"


def test_video00_internal_retake_guard_fails_open_without_retry_evidence():
    tokens = (
        "esta", "es", "una", "historia", "sobre", "mi", "familia", "y", "mi", "salud",
        "despues", "cuento", "otra", "parte", "distinta", "de", "la", "historia",
    )
    words = _words(tokens, start=30.0, step=0.22)
    take = CandidateTake(
        clip_id="story",
        source_asset_id="src",
        source_order=0,
        start=30.0,
        end=words[-1].end,
        text=" ".join(tokens),
        words=words,
        signals=MediaSignals("src", 30.0, words[-1].end),
        complete_idea=True,
    )

    resolved, diagnostics = prefer_internal_clean_retakes((take,), _context())

    assert resolved == (take,)
    assert diagnostics == ()
