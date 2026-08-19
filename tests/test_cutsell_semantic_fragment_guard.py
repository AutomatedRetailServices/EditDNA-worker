from cutsell_worker.contracts import CandidateTake
from cutsell_worker.hybrid_editorial import EditorialDecision, EditorialJudgeResult
from cutsell_worker.hybrid_session_cleanup import apply_hybrid_session_cleanup


def take(index: int, text: str, duration: float) -> CandidateTake:
    start = float(index * 4)
    return CandidateTake(
        clip_id=f"clip-{index}",
        source_asset_id="src",
        source_order=0,
        start=start,
        end=start + duration,
        text=text,
    )


class MappingJudge:
    def __init__(self, labels):
        self.labels = labels

    def judge(self, session):
        return EditorialJudgeResult(
            decisions=tuple(
                EditorialDecision(candidate.clip_id, *self.labels[candidate.clip_id], "test")
                for candidate in session.candidates
            ),
            provider="fake",
            model="flash-lite",
            requested=True,
            available=True,
            estimated_input_tokens=100,
            estimated_output_tokens=50,
        )


def test_failed_open_micro_fragment_deletes_without_visual_corroboration():
    item = take(0, "worried if", 1.50)
    result = apply_hybrid_session_cleanup(
        (item,), None, MappingJudge({"clip-0": ("failed", 0.80)})
    )
    assert result.kept == ()
    assert result.deleted == (item,)
    guard = result.diagnostics[-1]["semantic_fragment_guard"]
    assert guard[0]["reason"] == "semantic_failed_micro_fragment"


def test_two_word_failed_false_start_deletes_at_point_eight():
    item = take(0, "you're tired", 1.28)
    result = apply_hybrid_session_cleanup(
        (item,), None, MappingJudge({"clip-0": ("failed", 0.80)})
    )
    assert result.deleted == (item,)
    assert result.diagnostics[-1]["semantic_fragment_guard"][0]["reason"] == "semantic_failed_micro_fragment"


def test_failed_repetition_pathology_is_structural_corroboration():
    item = take(0, "non gmo non gmo non gmo gluten free and vegan", 6.0)
    result = apply_hybrid_session_cleanup(
        (item,), None, MappingJudge({"clip-0": ("failed", 0.85)})
    )
    assert result.deleted == (item,)
    assert result.diagnostics[-1]["semantic_fragment_guard"][0]["reason"] == "semantic_failed_repetition_pathology"


def test_bts_filler_micro_debris_deletes_without_visual_signal():
    item = take(0, "you know", 1.05)
    result = apply_hybrid_session_cleanup(
        (item,), None, MappingJudge({"clip-0": ("bts", 0.90)})
    )
    assert result.deleted == (item,)
    assert result.diagnostics[-1]["semantic_fragment_guard"][0]["reason"] == "semantic_bts_micro_debris"


def test_failed_open_comma_fragment_deletes_at_point_eight():
    item = take(0, "I give me the money,", 1.18)
    result = apply_hybrid_session_cleanup(
        (item,), None, MappingJudge({"clip-0": ("failed", 0.80)})
    )
    assert result.deleted == (item,)
    assert result.diagnostics[-1]["semantic_fragment_guard"][0]["reason"] == "semantic_failed_micro_fragment"


def test_failed_longer_spanish_open_tail_deletes_at_point_eight():
    item = take(0, "la barrera cutánea te la te hace como", 3.34)
    result = apply_hybrid_session_cleanup(
        (item,), None, MappingJudge({"clip-0": ("failed", 0.80)})
    )
    assert result.deleted == (item,)
    assert result.diagnostics[-1]["semantic_fragment_guard"][0]["reason"] == "semantic_failed_open_fragment"


def test_valid_short_hook_is_preserved_when_semantics_say_keep():
    item = take(0, "Shop now", 1.0)
    result = apply_hybrid_session_cleanup(
        (item,), None, MappingJudge({"clip-0": ("keep", 0.95)})
    )
    assert result.kept == (item,)
    assert result.deleted == ()


def test_low_confidence_failed_long_unique_story_still_fails_open():
    item = take(
        0,
        "This is the complete story about why I changed doctors and what happened next",
        8.0,
    )
    result = apply_hybrid_session_cleanup(
        (item,), None, MappingJudge({"clip-0": ("failed", 0.80)})
    )
    assert result.kept == (item,)
    assert result.deleted == ()
