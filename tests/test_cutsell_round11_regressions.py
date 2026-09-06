from cutsell_worker.contracts import CandidateTake, DraftClip, SemanticRole, Word
from cutsell_worker.failed_prefix_completion_rescue import rescue_failed_completion_prefixes
from cutsell_worker.round11_semantic_retry_cleanup import suppress_failed_open_attempt_before_later_winner


def _word(text, start, end):
    return Word(text=text, start=float(start), end=float(end), confidence=0.98)


def _candidate(clip_id, start, end, text, words, *, complete=True):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=float(start),
        end=float(end),
        text=text,
        words=tuple(words),
        complete_idea=complete,
    )


def _draft_clip(clip_id, start, end, text, *, selected=True):
    return DraftClip(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=float(start),
        end=float(end),
        text=text,
        caption_text=text,
        semantic_role=SemanticRole.STORY,
        selected=selected,
    )


def test_round11_video03_rescues_cutanea_at_failed_085_with_explicit_collision():
    previous = _candidate(
        "winner",
        26.35,
        42.39,
        "esta crema es mágica tiene unos componentes que de verdad te protegen te reparan",
        (_word("te", 41.8, 41.95), _word("reparan", 41.96, 42.39)),
    )
    failed_words = (
        _word("la", 44.33, 44.45),
        _word("barrera", 44.46, 44.88),
        _word("cutánea", 44.89, 45.30),
        _word("te", 45.31, 45.42),
        _word("la", 45.43, 45.52),
        _word("te", 45.53, 45.64),
        _word("hace", 45.65, 45.92),
        _word("como", 45.93, 46.18),
    )
    failed = _candidate(
        "failed",
        44.33,
        47.67,
        "la barrera cutánea te la te hace como",
        failed_words,
        complete=False,
    )

    kept, audit = rescue_failed_completion_prefixes(
        (previous,),
        (failed,),
        (("failed", "failed", 0.85),),
    )

    rescued = [take for take in kept if take.clip_id.endswith("_completion_prefix")]
    assert len(rescued) == 1
    assert rescued[0].text == "la barrera cutánea"
    assert rescued[0].end == 45.30
    assert audit[0]["semantic_confidence"] == 0.85


def test_completion_rescue_still_fails_open_at_085_without_collision():
    previous = _candidate("winner", 10, 12, "te protegen te reparan", (_word("reparan", 11.5, 12),))
    failed = _candidate(
        "failed",
        13,
        16,
        "la barrera cutánea queda muy bien",
        (
            _word("la", 13.0, 13.1),
            _word("barrera", 13.1, 13.5),
            _word("cutánea", 13.5, 13.9),
            _word("queda", 14.0, 14.3),
            _word("muy", 14.3, 14.5),
            _word("bien", 14.5, 14.8),
        ),
        complete=False,
    )

    kept, audit = rescue_failed_completion_prefixes((previous,), (failed,), (("failed", "failed", 0.85),))

    assert kept == (previous,)
    assert audit == ()


def test_round11_stomach_failed_open_attempt_yields_to_later_winner():
    failed = _draft_clip(
        "stomach_failed",
        236.21,
        245.03,
        "Tuve problemas estomacales a un tiempo en donde se me hizo una endoscopía y me diagnosticaron con...",
    )
    winner = _draft_clip(
        "stomach_winner",
        258.57,
        269.35,
        "no. Tuve problemas de digestión en donde me hicieron una endoscopía y dijeron que tenía gastritis. Nada severo pero tenía gastritis y me mandaron tres meses con pastillas.",
    )
    diagnostics = {
        "hybrid_editorial_chunks": [
            {"decisions": [
                {"clip_id": "stomach_failed", "label": "failed", "confidence": 0.85},
                {"clip_id": "stomach_winner", "label": "winner", "confidence": 0.95},
            ]}
        ]
    }

    selected, discarded, audit = suppress_failed_open_attempt_before_later_winner(
        (failed, winner), (), diagnostics
    )

    assert [clip.clip_id for clip in selected] == ["stomach_winner"]
    assert [clip.clip_id for clip in discarded] == ["stomach_failed"]
    assert audit[0]["reason"] == "failed_open_attempt_superseded_by_later_semantic_winner"


def test_semantic_retry_cleanup_fails_open_for_distinct_neighboring_idea():
    failed = _draft_clip("failed", 10, 14, "Tuve problemas estomacales y me diagnosticaron con...")
    winner = _draft_clip("winner", 20, 26, "Después cambié de trabajo y comencé una rutina completamente diferente.")
    diagnostics = {
        "hybrid_editorial_chunks": [
            {"decisions": [
                {"clip_id": "failed", "label": "failed", "confidence": 0.90},
                {"clip_id": "winner", "label": "winner", "confidence": 0.95},
            ]}
        ]
    }

    selected, discarded, audit = suppress_failed_open_attempt_before_later_winner(
        (failed, winner), (), diagnostics
    )

    assert selected == (failed, winner)
    assert discarded == ()
    assert audit == ()
