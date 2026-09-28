from dataclasses import replace
from types import SimpleNamespace
import pytest

from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, JobState, ProcessingResult, Word
from cutsell_worker.v2_source_selection import canonicalize_candidates
from cutsell_worker.v2_recording_tail import trim_recording_tail
from cutsell_worker.unified_selection_reasoner import UnifiedSelectionDecision


def test_candidates_and_boundary_reuse_one_source_word_stream():
    words = (Word("full", 0, 1), Word("phrase", 1, 2))
    clip = DraftClip("c", "src", 0, .2, 1.8, "different words", "", words=())
    draft = DraftTimeline("cutsell.v1", "p", EditStrategy.STORYTELLING, (clip,), (), ())
    result = ProcessingResult("cutsell.v1", "p", JobState.DRAFT_READY, draft, {})
    class ASR:
        calls = 0
        def transcribe(self, *a, **k):
            self.calls += 1
            return (SimpleNamespace(words=words),)
    asr = ASR()
    out, frozen = canonicalize_candidates(result, {"src": "unused"}, asr)
    assert out.draft.selected[0].text == "full phrase"
    assert out.draft.selected[0].words == frozen.transcribe("unused", source_asset_id="src")[0].words
    assert asr.calls == 1
    assert out.draft.selected[0].start == .2 and out.draft.selected[0].end == 1.8


def test_missing_canonical_alignment_identifies_geometry_without_raw_speech():
    clip = DraftClip('candidate-1', 'src', 0, 5.0, 6.0, 'private dialogue', '')
    draft = DraftTimeline('cutsell.v1', 'p', EditStrategy.STORYTELLING, (clip,), (), ())
    result = ProcessingResult('cutsell.v1', 'p', JobState.DRAFT_READY, draft, {})
    class ASR:
        def transcribe(self, *a, **k):
            return (SimpleNamespace(words=(Word('elsewhere', 0.0, 1.0),)),)
    with pytest.raises(ValueError, match='clip_id=candidate-1.*interval=5.000..6.000.*source_word_count=1') as exc:
        canonicalize_candidates(result, {'src': 'unused'}, ASR())
    assert 'private dialogue' not in str(exc.value)


def test_shared_source_word_survives_once_across_adjacent_selected_candidates():
    from cutsell_worker.v2_source_selection import reconcile_canonical_word_seams
    from cutsell_worker.final_boundary_authority import enforce_complete_idea_boundaries
    words = (Word("full", 0, 1), Word("phrase", 1, 2), Word("continues", 2, 3))
    a = DraftClip("a", "src", 0, 0, 1.5, "full phrase", "")
    b = DraftClip("b", "src", 0, 1.5, 3, "phrase continues", "")
    draft = DraftTimeline("cutsell.v1", "p", EditStrategy.STORYTELLING, (a, b), (), (),
                          {"editorial_engine_v2_request": {"require_audiovisual_evidence": True}})
    result = ProcessingResult("cutsell.v1", "p", JobState.DRAFT_READY, draft, {})
    class ASR:
        def transcribe(self, *a, **kw): return (SimpleNamespace(words=words),)
    result, asr = canonicalize_candidates(result, {"src": "unused"}, ASR())
    result = reconcile_canonical_word_seams(enforce_complete_idea_boundaries(
        result, {"src": "unused"}, asr_provider=asr), asr.words_by_source)
    assert [(c.start, c.end) for c in result.draft.selected] == [(0, 2), (2, 3)]
    assert " ".join(c.text for c in result.draft.selected) == "full phrase continues"


def restart_case():
    text = "tienes energía en el día y aparte tus músculos tus mus"
    tokens = text.split()
    words = tuple(Word(t, i*.3, (i+1)*.3) for i, t in enumerate(tokens))
    clip = DraftClip("head", "src", 0, 0, words[-1].end, text, text, words=words)
    target = DraftClip("next", "src", 0, 5, 9, "y aparte tus músculos van a estar más fuertes", "")
    decision = UnifiedSelectionDecision("head", "select", "independent", .95, 0,
        "independent_story_coverage", 0, 6, .99, "abandoned_restart", "next")
    return clip, target, decision, {"v2_native_selection_modalities": ["video", "audio", "source_words"]}


def test_shared_seam_repair_never_drops_unrelated_partial_outer_words():
    from cutsell_worker.v2_source_selection import reconcile_canonical_word_seams
    words = tuple(Word(t, i, i+1) for i,t in enumerate(("Alpha", "Bridge", "Beta", "Tail")))
    left = DraftClip("a", "src", 0, .5, 1.5, "Alpha Bridge", "", words=words[:2])
    right = DraftClip("b", "src", 0, 1.5, 3.5, "Bridge Beta Tail", "", words=words[1:])
    draft = DraftTimeline("cutsell.v1", "p", EditStrategy.STORYTELLING, (left,right), (), ())
    result = ProcessingResult("cutsell.v1", "p", JobState.DRAFT_READY, draft, {})
    assert reconcile_canonical_word_seams(result, {"src": words}).draft.selected == draft.selected


def test_explicit_native_abandoned_suffix_requires_complete_selected_coverage():
    clip, target, decision, diag = restart_case()
    out, row = trim_recording_tail(clip, decision, diag, selected_clips=(target,))
    assert out.text == "tienes energía en el día"
    assert row["action"] == "trim"


@pytest.mark.parametrize("defect", ["no_video", "not_selected", "other_source", "missing_fact", "uncertain"])
def test_abandoned_suffix_refused_without_corroboration(defect):
    clip, target, decision, diag = restart_case()
    targets = (target,)
    if defect == "no_video": diag = {}
    if defect == "not_selected": targets = ()
    if defect == "other_source": targets = (replace(target, source_asset_id="other"),)
    if defect == "missing_fact": targets = (replace(target, text="una explicación distinta"),)
    if defect == "uncertain": decision = replace(decision, trailing_recording_confidence=.85)
    assert trim_recording_tail(clip, decision, diag, selected_clips=targets)[0] == clip


def test_complete_accented_word_not_removed_as_an_aborted_prefix():
    clip, target, decision, diag = restart_case()
    tokens = "lo explico ahora tomo cafeína y también tomo café".split()
    words = tuple(Word(t, i*.3, (i+1)*.3) for i,t in enumerate(tokens))
    clip = replace(clip, text=" ".join(tokens), words=words, end=words[-1].end)
    target = replace(target, text="y también tomo cafeína")
    decision = replace(decision, trailing_recording_word_count=4)
    assert trim_recording_tail(clip, decision, diag, selected_clips=(target,))[0] == clip
