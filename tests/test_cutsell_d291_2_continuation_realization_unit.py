"""D-291.2 -- a sentence-continuation chain is ONE realization for the
canonical model (Ledger / Resolver / authoritative application), not only
for the family competition.

RAW #125 (run 35921819172): IdeaClusterer recorded `sentence_continuation`
between the acne head ("Por temporada me salió un acné en la espalda con la
que yo resolvía con", 185.24-189.84 s) and its tail ("resorcina.",
191.14-191.74 s); BestTake selected the chain; but the Semantic Ledger
registered head and tail as two realizations of one idea, the Realization
Resolver read the head's pre-fold `failed` window label (>= 0.85 ->
unusable), kept the tail alone as the idea's winner, discarded the sentence
and waived its claims (`waived_failed_realization_source`). The rendered
selection therefore carried a stranded "resorcina." and lost the acne take.

Fix: `continuation_chain.unify_chain_realizations` (called by the pipeline
right where the chain is known) gives every tail its head's
`realization_id` (D-050A's own "a physical split preserves realization
identity" invariant) and `DraftClip.parent_realization_id` records the
join; the Ledger registers a multi-clip realization with the complete
sentence, claims from that sentence, the joined span and completeness
re-graded on the joined text. No id, phrase, timestamp, threshold or
later restoration: the unit is evaluated, kept or discarded exactly as
any single realization is.

The texts/spans are RAW #125's recorded ones; the window labels are the
recorded RAW #122 chunk-3 labels for the same clips (RAW #125's own labels
are in a result JSON this container cannot reach) -- every judge/arbiter
answer here is a LABELLED FAKE.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from cutsell_worker import universal_clean_cut as universal
from cutsell_worker.continuation_chain import unify_chain_realizations
from cutsell_worker.contracts import (
    SCHEMA_VERSION, CandidateTake, JobState, ProcessingRequest, ProcessingResult, SourceAsset, TranscriptSegment, Word,
)
from cutsell_worker.hybrid_editorial import EditorialDecision, EditorialJudgeResult
from cutsell_worker.pipeline import build_flow_b_draft
from cutsell_worker.realization_resolver import (
    EFFECTIVE_IMPORTANCE_WAIVED_FAILED_REALIZATION_SOURCE,
    apply_authoritative_realization_resolution,
    resolve_realizations_shadow,
)
from cutsell_worker.semantic_ledger import build_semantic_ledger_shadow
from tests.test_cutsell_d289_contained_realization_closure import RecordedAnswersArbiter

F0_TEXT = "Por temporada me salía en la Por temporada me salía acné en la espalda."
H_TEXT = "Por temporada me salió un acné en la espalda con la que yo resolvía con"
T_TEXT = "resorcina."
X_TEXT = "También me salían espinillas. Era como un rush, una alergia."
Y_TEXT = "Se me caía mucho el pelo cuando me lavaba el pelo perdía mucho pelo."
CHAIN_TEXT = f"{H_TEXT} {T_TEXT}"


def _take(cid, start, end, text, *, complete=True):
    return CandidateTake(cid, "src", 0, start, end, text, complete_idea=complete)


def _takes():
    return (
        _take("F0", 171.34, 181.34, F0_TEXT),
        _take("H", 185.24, 189.84, H_TEXT, complete=False),
        _take("T", 191.14, 191.74, T_TEXT),
        _take("X", 192.44, 198.12, X_TEXT),
        _take("Y", 226.74, 233.18, Y_TEXT),
    )


# RAW #122 chunk-3 labels for the same clips (LABELLED FAKE for RAW #125).
LABELS = {"F0": ("failed", 0.98), "H": ("failed", 0.9), "T": ("alternate", 0.7), "X": ("winner", 0.9), "Y": ("winner", 0.95)}


class FakeJudge:
    def __init__(self, labels):
        self.labels = dict(labels)

    def judge(self, session):
        return EditorialJudgeResult(
            tuple(EditorialDecision(c.clip_id, *self.labels.get(c.clip_id, ("keep", 0.6)), "fake") for c in session.candidates),
            "fake", "fake-model", True, True, 200, 40,
        )


def _arbiter():
    # the recorded same-idea verdict that puts the earlier attempt in the family (plus its component probe)
    return RecordedAnswersArbiter({
        (F0_TEXT, H_TEXT): (True, 0.95, "same idea: back acne, two deliveries"),
        (F0_TEXT, CHAIN_TEXT): (True, 0.9, "component probe: same idea"),
    })


def _request():
    return ProcessingRequest(project_id="p", user_id="u", sources=(SourceAsset(
        source_asset_id="src", project_id="p", user_id="u", original_name="raw.mp4", source_order=0,
        duration_sec=400.0, uri="s3://b/raw.mp4",
    ),))


def _pipeline(takes=None, labels=LABELS, arbiter=None):
    return build_flow_b_draft(_request(), takes or _takes(), editorial_judge=FakeJudge(labels),
                              semantic_equivalence_arbiter=arbiter or _arbiter())


def _authority(draft):
    ledger = build_semantic_ledger_shadow(draft)
    report = resolve_realizations_shadow(ledger)
    applied = apply_authoritative_realization_resolution(draft, ledger, report)
    return ledger, report, applied


def _ids(clips):
    return [c.clip_id for c in clips]


# --- A. reproduction: the recorded shape, before and after ---

def test_raw125_shape_before_the_fix_the_resolver_keeps_the_tail_alone_and_waives_the_sentence():
    # The pre-D-291.2 canonical model, reconstructed deterministically: the
    # tail carries its OWN realization id (what the pipeline minted before).
    result = _pipeline()
    draft = result.draft
    assert (draft.diagnostics["distinct_idea_grouping_safety"]["continuation_chains"]) == [["H", "T"]]
    assert _ids(draft.selected) == ["H", "T", "X", "Y"] and _ids(draft.alternates) == ["F0"]
    split = replace(draft, selected=tuple(
        replace(c, realization_id="real_tail_own", parent_realization_id=None) if c.clip_id == "T" else c
        for c in draft.selected
    ))
    ledger, report, applied = _authority(split)
    idea = next(r for r in ledger.realizations().values() if "H" in r.clip_ids).semantic_idea_id
    resolution = report.idea_resolutions[idea]
    assert resolution.winner_realization_id == "real_tail_own"
    assert set(resolution.discarded_realization_ids) == {
        r.realization_id for r in ledger.realizations().values() if r.clip_ids in (("H",), ("F0",))
    }
    assert resolution.evidence["critical_groups_waived_from_failed_realizations"]
    assert applied.status == "SEMANTICALLY_RESOLVED"
    assert _ids(applied.draft.selected) == ["T", "X", "Y"] and set(_ids(applied.draft.discarded)) == {"H", "F0"}


def test_raw125_shape_after_the_fix_the_chain_is_one_realization_kept_as_a_unit():
    result = _pipeline()
    draft = result.draft
    head = next(c for c in draft.selected if c.clip_id == "H")
    tail = next(c for c in draft.selected if c.clip_id == "T")
    assert tail.realization_id == head.realization_id
    assert tail.parent_realization_id == head.realization_id and head.parent_realization_id is None
    ledger, report, applied = _authority(draft)
    unit = ledger.realizations()[head.realization_id]
    assert unit.clip_ids == ("H", "T") and unit.text == CHAIN_TEXT
    assert unit.complete_idea is True and unit.start == 185.24 and unit.end == 191.74
    assert (unit.semantic_label, unit.semantic_label_confidence) == ("failed", 0.9)  # the head's pre-fold window label, recorded, not hidden
    resolution = report.idea_resolutions[unit.semantic_idea_id]
    assert resolution.winner_realization_id == head.realization_id
    assert resolution.discarded_realization_ids == ()
    # the failed label on BOTH candidates cancels (WHEN UNCERTAIN, KEEP); nothing is waived
    assert resolution.evidence["unusable_realization_ids"] == []
    assert resolution.evidence["critical_groups_waived_from_failed_realizations"] == []
    assert applied.status == "SEMANTICALLY_RESOLVED"
    assert _ids(applied.draft.selected) == ["H", "T", "X", "Y"]
    assert "T" not in _ids(applied.draft.discarded)


# --- B. controls ---

def test_control_independent_ideas_stay_separate_realizations():
    takes = (_take("A", 1.0, 5.0, "Tuve problemas de digestión y me hicieron una endoscopía."),
             _take("B", 6.0, 10.0, "Se me caía mucho el pelo cuando me lavaba el pelo."))
    draft = _pipeline(takes, {"A": ("winner", 0.9), "B": ("winner", 0.9)}, RecordedAnswersArbiter({})).draft
    assert draft.diagnostics["distinct_idea_grouping_safety"].get("continuation_chains", []) == []
    ledger, report, applied = _authority(draft)
    ids = {r.realization_id for r in ledger.realizations().values()}
    assert len(ids) == 2 and all(c.parent_realization_id is None for c in draft.selected)
    assert _ids(applied.draft.selected) == ["A", "B"]


def test_control_a_genuinely_failed_attempt_is_still_discarded_beside_the_chain():
    result = _pipeline()
    ledger, report, applied = _authority(result.draft)
    f0 = next(r for r in ledger.realizations().values() if r.clip_ids == ("F0",))
    assert (f0.semantic_label, f0.state) == ("failed", "alternate")
    assert "F0" in _ids(applied.draft.alternates)
    from cutsell_worker.final_story_coherence_validation import fold_alternates_into_discarded
    folded = fold_alternates_into_discarded(applied.draft)
    assert "F0" in _ids(folded.discarded) and _ids(folded.selected) == ["H", "T", "X", "Y"]


def _clean_retake_takes():
    clean = _take("C", 195.0, 201.0, "Por temporada me salió un acné en la espalda que resolvía con resorcina.")
    return (_take("H", 185.24, 189.84, H_TEXT, complete=False), _take("T", 191.14, 191.74, T_TEXT), clean,
            _take("Y", 226.74, 233.18, Y_TEXT)), clean


def _clean_retake_arbiter(clean):
    return RecordedAnswersArbiter({(H_TEXT, clean.text): (True, 0.95, "same sentence, clean retake"),
                                   (CHAIN_TEXT, clean.text): (True, 0.95, "component probe: same idea")})


def test_control_a_failed_head_with_a_clean_later_retake_is_removed_with_its_tail_before_grouping():
    # `failed` 0.9 on the head plus a later overlapping complete retake is the
    # existing hybrid "semantic_failed_plus_later_overlapping_complete_retake"
    # basis: the attempt is removed early and the bare tail never survives
    # alone beside the clean take. A genuinely failed attempt is still discarded.
    takes, clean = _clean_retake_takes()
    labels = {"H": ("failed", 0.9), "T": ("alternate", 0.7), "C": ("winner", 0.95), "Y": ("winner", 0.95)}
    draft = _pipeline(takes, labels, _clean_retake_arbiter(clean)).draft
    assert _ids(draft.selected) == ["C", "Y"]
    assert set(_ids(draft.discarded)) == {"H", "T"}
    ledger, report, applied = _authority(draft)
    assert _ids(applied.draft.selected) == ["C", "Y"] and "T" not in _ids(applied.draft.selected)


def test_control_a_chain_competing_with_a_clean_later_take_moves_as_a_unit_never_the_tail_alone():
    # No early removal (head merely "alternate"): the chain forms and competes
    # against the clean later take. Whatever the Resolver decides for the
    # idea, head and tail land in the SAME bucket -- the unit property this
    # decision adds. (Observed and recorded, not endorsed: in this synthetic
    # harness every DeliveryScore is 1.0, BestTake's "winner" label for C
    # agrees with the local winner so the Ledger records no
    # SEMANTIC_WINNER_OVERRIDE for it, and the Resolver's pre-existing tie
    # rule then falls through score and richness to realization-id order --
    # a separate finding, see D-291.2 "pending limits"; no rule was added or
    # changed here to force either outcome.)
    takes, clean = _clean_retake_takes()
    labels = {"H": ("alternate", 0.7), "T": ("alternate", 0.7), "C": ("winner", 0.95), "Y": ("winner", 0.95)}
    draft = _pipeline(takes, labels, _clean_retake_arbiter(clean)).draft
    assert draft.diagnostics["distinct_idea_grouping_safety"]["continuation_chains"] == [["H", "T"]]
    assert "C" in _ids(draft.selected) and "H" not in _ids(draft.selected) and "T" not in _ids(draft.selected)
    ledger, report, applied = _authority(draft)
    chain = next(r for r in ledger.realizations().values() if r.clip_ids == ("H", "T"))
    assert chain.text == CHAIN_TEXT and chain.state == "alternate"
    idea = report.idea_resolutions[chain.semantic_idea_id]
    assert idea.decision_status == "RESOLVED_WINNER"
    assert set(idea.candidate_realization_ids) == {chain.realization_id, next(r.realization_id for r in ledger.realizations().values() if r.clip_ids == ("C",))}

    def bucket(cid):
        for name in ("selected", "alternates", "discarded"):
            if cid in _ids(getattr(applied.draft, name)):
                return name
        return None

    assert bucket("H") is not None and bucket("H") == bucket("T")
    if bucket("H") == "selected":
        assert _ids(applied.draft.selected).index("T") == _ids(applied.draft.selected).index("H") + 1
    assert bucket("Y") == "selected"


def test_unify_chain_realizations_contract():
    head = replace(_take("H", 1.0, 2.0, H_TEXT, complete=False), realization_id="real_head")
    tail = replace(_take("T", 2.5, 3.0, T_TEXT), realization_id="real_tail")
    other = replace(_take("O", 5.0, 6.0, Y_TEXT), realization_id="real_other")
    out, parents = unify_chain_realizations((head, tail, other), {"H": ("T",)})
    assert [t.realization_id for t in out] == ["real_head", "real_head", "real_other"]
    assert parents == {"T": "real_head"}
    # a head that never received an identity leaves its chain untouched (nothing unified on partial identity)
    out2, parents2 = unify_chain_realizations((replace(head, realization_id=None), tail), {"H": ("T",)})
    assert [t.realization_id for t in out2] == [None, "real_tail"] and parents2 == {}
    # no chains: byte-identical
    out3, parents3 = unify_chain_realizations((head, tail), {})
    assert out3 == (head, tail) and parents3 == {}


# --- C. the whole path to Freeze and the pre-Freeze boundary authority ---

def _synthetic_words(text, start, end):
    toks = text.split()
    slot = (end - start) / len(toks)
    return tuple(Word(text=t, start=round(start + i * slot, 3), end=round(start + i * slot + slot * 0.9, 3)) for i, t in enumerate(toks))


class _FakeASR:
    """SYNTHETIC word timings spread over each recorded clip span (RAW #125's
    real timed ASR is in the unreachable result JSON)."""

    def __init__(self, takes):
        self.words = tuple(sorted((w for t in takes for w in _synthetic_words(t.text, t.start, t.end)), key=lambda w: w.start))

    def transcribe(self, path, *, source_asset_id, language_hint=None):
        return (TranscriptSegment(source_asset_id=source_asset_id, start=self.words[0].start, end=self.words[-1].end,
                                  text=" ".join(w.text for w in self.words), words=self.words),)


def test_full_path_to_freeze_keeps_the_acne_sentence_and_reaches_final_boundary_authority(monkeypatch):
    takes = _takes()
    pipeline_result = _pipeline(takes)
    monkeypatch.setattr(universal, "process_local_sources", lambda request, local_paths, **kw: pipeline_result)
    monkeypatch.setattr(universal, "polish_human_boundaries_v5", lambda result, paths: result)
    result = universal.process_universal_clean_cut_sources(
        object(), {"src": "/nonexistent.mp4"}, asr_provider=_FakeASR(takes), selection_reasoner=None,
    )
    diag = result.draft.diagnostics
    assert result.stage_status["freeze_blocked_pending_coherence_review"] is False
    assert diag["selection_boundary_contract"]["status"] == "verified"
    final = _ids(result.draft.selected)
    assert "H" in final and "T" in final and final.index("T") == final.index("H") + 1
    assert "F0" not in final
    # the pre-Freeze boundary authority ran on this selection (RAW #125 never reached it for the CTA question)
    rows = diag["final_boundary_authority"]
    assert {r.get("clip_id") for r in rows if r.get("clip_id")} >= {"H", "T"}
    assert "final_boundary_reopened_closing_trim_count" in diag
    assert result.stage_status.get("final_edit_reviewer") == "PASS"
