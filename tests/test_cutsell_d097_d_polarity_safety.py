"""D-097 Priority D -- semantic polarity / micro-fragment safety.

Run 34008386434 rendered a clause with inverted meaning: an emphatic pause
after a bare "No" put the particle in its own ASR speech unit, the strict
one-word bridge refused it, a cleanup deleted it as micro debris. The
earliest authority (segmentation / AttemptReconstructor input) now rejoins a
bare polarity particle to the clause it negates across a normal pause; the
semantic fragment guard refuses every brevity-only deletion of a
polarity-bearing micro fragment. Multilingual, no literal video phrase; the
fixtures below are generic.
"""
import sys

import pytest

from cutsell_worker.contracts import CandidateTake, SourceAsset, TranscriptSegment, Word
from cutsell_worker.hybrid_editorial import EditorialDecision, EditorialJudgeResult
from cutsell_worker.polarity_safety import carries_polarity, is_bare_polarity_unit
from cutsell_worker.semantic_fragment_guard import PROTECTED_POLARITY_FRAGMENT, remove_semantic_fragment_debris
from cutsell_worker.take_segmentation import _looks_complete_idea, _repair_boundary_fragments, segment_takes


def _take(clip_id, start, end, text, source="src-1"):
    tokens = text.split()
    step = max(0.05, (end - start) / max(1, len(tokens)))
    words = tuple(Word(token, start + i * step, min(end, start + i * step + step * 0.8)) for i, token in enumerate(tokens))
    return CandidateTake(
        clip_id=clip_id, source_asset_id=source, source_order=0, start=start, end=end, text=text,
        words=words, complete_idea=_looks_complete_idea(text, end - start),
    )


# --- shared vocabulary --------------------------------------------------------

@pytest.mark.parametrize("text", ["No", "no", "No.", "no, no", "pero no", "Nunca", "not", "and not", "Don't", "jamás"])
def test_bare_polarity_units(text):
    assert is_bare_polarity_unit(text)


@pytest.mark.parametrize("text", ["No lo sé.", "not really sure", "nothing", "nadie vino", "Sí", "okay", ""])
def test_units_with_other_content_are_not_bare(text):
    assert not is_bare_polarity_unit(text)


def test_carries_polarity_is_the_wider_test():
    assert carries_polarity("no quiero")
    assert carries_polarity("I can't")
    assert not carries_polarity("quiero sonar claro")


# --- segmentation rejoin (earliest authority) -----------------------------------

def test_bare_negation_rejoins_following_clause_across_an_emphatic_pause():
    particle = _take("p", 10.00, 10.80, "No")
    clause = _take("c", 11.90, 15.20, "quiero sonar a conspiración con esto")
    rejoins = []
    repaired = _repair_boundary_fragments((particle, clause), polarity_rejoins=rejoins)
    assert len(repaired) == 1
    assert repaired[0].text == "No quiero sonar a conspiración con esto"
    assert repaired[0].start == 10.00 and repaired[0].end == 15.20
    assert len(repaired[0].words) == len(particle.words) + len(clause.words)
    assert rejoins and rejoins[0]["particle_text"] == "No" and rejoins[0]["gap_sec"] == pytest.approx(1.1)


def test_english_contraction_particle_rejoins_too():
    particle = _take("p", 4.00, 4.50, "Don't")
    clause = _take("c", 5.30, 8.00, "use it on broken skin, ever.")
    repaired = _repair_boundary_fragments((particle, clause))
    assert len(repaired) == 1
    assert repaired[0].text.startswith("Don't use it")


def test_no_rejoin_across_a_real_section_boundary():
    particle = _take("p", 10.00, 10.80, "No")
    clause = _take("c", 13.50, 17.00, "el siguiente punto es la rutina de noche")
    repaired = _repair_boundary_fragments((particle, clause))
    assert len(repaired) == 2  # 2.7 s gap > 2.0 s ceiling: a standalone answer stays standalone


def test_no_rejoin_when_the_particle_belongs_to_another_source():
    particle = _take("p", 10.00, 10.80, "No", source="src-A")
    clause = _take("c", 11.00, 14.00, "quiero sonar a conspiración", source="src-B")
    assert len(_repair_boundary_fragments((particle, clause))) == 2


def test_repeated_emphatic_particle_folds_then_rejoins():
    first = _take("p1", 10.00, 10.40, "No")
    second = _take("p2", 10.90, 11.30, "no")
    clause = _take("c", 12.00, 15.00, "es hereditario en la mayoría de casos")
    repaired = _repair_boundary_fragments((first, second, clause))
    assert len(repaired) == 1
    assert repaired[0].text == "No no es hereditario en la mayoría de casos"


def test_a_non_polarity_one_word_marker_is_still_strict():
    marker = _take("m", 10.00, 10.60, "Bueno")
    clause = _take("c", 11.50, 14.50, "vamos a hablar de la rutina de noche")
    assert len(_repair_boundary_fragments((marker, clause))) == 2  # unchanged pre-D-097 contract


def test_a_short_complete_answer_with_content_is_not_a_bare_particle():
    answer = _take("a", 10.00, 11.20, "No lo sé.")
    clause = _take("c", 11.60, 14.50, "pero creo que vale la pena probarlo")
    repaired = _repair_boundary_fragments((answer, clause))
    assert len(repaired) == 2


def test_segment_takes_publishes_polarity_rejoin_diagnostics():
    words = (
        Word("No", 10.0, 10.6),
        Word("quiero", 11.7, 12.0), Word("sonar", 12.0, 12.4), Word("a", 12.4, 12.5),
        Word("conspiración", 12.5, 13.2), Word("con", 13.2, 13.4), Word("esto", 13.4, 13.9),
    )
    segment = TranscriptSegment(source_asset_id="src-1", start=10.0, end=13.9, text=" ".join(w.text for w in words), words=words)
    diagnostics: dict = {}
    source = SourceAsset(
        source_asset_id="src-1", project_id="p", user_id="u", original_name="a.mp4",
        source_order=0, duration_sec=60.0, uri="s3://x/y.mp4",
    )
    takes = segment_takes((segment,), (source,), diagnostics=diagnostics)
    assert len(takes) == 1
    assert takes[0].text.startswith("No quiero")
    assert diagnostics["polarity_rejoins"][0]["particle_start"] == 10.0
    assert diagnostics["polarity_rejoins"][0]["clause_start"] == 11.7


# --- fragment guard protection (safety net) ------------------------------------

def test_guard_never_deletes_a_bare_negation_as_micro_debris():
    particle = _take("p", 10.00, 10.80, "No")
    survivors, removed, diagnostics = remove_semantic_fragment_debris((particle,), (("p", "failed", 0.95),))
    assert survivors == (particle,) and removed == ()
    assert diagnostics[0]["reason"] == PROTECTED_POLARITY_FRAGMENT
    assert diagnostics[0]["refused_reason"] == "semantic_failed_micro_fragment"


def test_guard_never_deletes_a_polarity_bearing_bts_micro_fragment():
    particle = _take("p", 10.00, 11.00, "not that")
    survivors, removed, _ = remove_semantic_fragment_debris((particle,), (("p", "bts", 0.95),))
    assert survivors == (particle,) and removed == ()


def test_guard_still_deletes_a_polarity_free_micro_false_start():
    fragment = _take("f", 10.00, 11.00, "worried if")
    survivors, removed, diagnostics = remove_semantic_fragment_debris((fragment,), (("f", "failed", 0.80),))
    assert survivors == () and removed == (fragment,)
    assert diagnostics[0]["reason"] == "semantic_failed_micro_fragment"


def test_guard_still_judges_a_longer_failed_fragment_on_its_own_evidence():
    # A polarity-bearing but clearly open, longer failed fragment keeps the
    # ordinary open-fragment contract: protection is for MICRO fragments only.
    fragment = _take("f", 10.00, 13.00, "no porque yo creo que la")
    survivors, removed, diagnostics = remove_semantic_fragment_debris((fragment,), (("f", "failed", 0.90),))
    assert removed == (fragment,)
    assert diagnostics[0]["reason"] in {"semantic_failed_short_fragment", "semantic_failed_open_fragment"}


# --- end-to-end: cleanup cannot invert meaning --------------------------------

class _MappingJudge:
    def __init__(self, labels):
        self.labels = labels

    def judge(self, session):
        return EditorialJudgeResult(
            decisions=tuple(EditorialDecision(c.clip_id, *self.labels[c.clip_id], "test") for c in session.candidates),
            provider="fake", model="flash-lite", requested=True, available=True,
            estimated_input_tokens=100, estimated_output_tokens=50,
        )


def test_hybrid_cleanup_with_guard_cannot_invert_meaning(monkeypatch):
    from cutsell_worker import hybrid_session_cleanup
    from cutsell_worker.semantic_fragment_guard import install_semantic_fragment_guard

    pure = hybrid_session_cleanup.apply_hybrid_session_cleanup
    install_semantic_fragment_guard()
    wrapped = hybrid_session_cleanup.apply_hybrid_session_cleanup
    monkeypatch.setattr(hybrid_session_cleanup, "apply_hybrid_session_cleanup", pure)

    particle = _take("p", 10.00, 10.80, "No")
    clause = _take("c", 13.50, 18.00, "quiero sonar a conspiración con esto que digo.")
    result = wrapped((particle, clause), None, _MappingJudge({"p": ("failed", 0.95), "c": ("winner", 0.95)}))
    kept_ids = {take.clip_id for take in result.kept}
    assert "c" in kept_ids
    assert "p" in kept_ids, "the negation must survive: deleting it inverts the clause"
    protection = [d for d in result.diagnostics if isinstance(d, dict) and d.get("protected_polarity_fragments")]
    assert protection and protection[0]["protected_polarity_fragments"][0]["clip_id"] == "p"
