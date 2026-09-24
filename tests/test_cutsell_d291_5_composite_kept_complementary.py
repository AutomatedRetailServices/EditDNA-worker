"""D-291.5 -- the composite Best Take judgment ("two complementary complete
deliveries replace a monolithic retry") must not depend on whether Hybrid
first DELETED the complementary pieces.

RAW #125 (video00-modal-35921819172-1): the long skin take M (198.88-211.02 s,
window label winner 0.95) was suppressed because the short earlier delivery
A1 (192.44-198.12 s) and the later delivery L (213.34-222.98 s) had been
deleted as cross-group retries, restored for their unique tails, and then
composed by `hybrid_composite_best_take` (peer coverage 0.6667). RAW #126
(run 35931561397, ladder): the same A1 and L were never deleted, so the
composite authority never saw them; M survived into grouping, paired with
A1 (the top-priority pair), won the family and A1 was lost while M played.
Both references keep A1 and L and drop M.

Fix: `_kept_complementary_rows` offers KEPT complete complementary
deliveries to the SAME composite judgment under the SAME association
criterion the complementary guard applies to a deletion. Texts here are
RAW #125's recorded ASR texts (QA fixtures); labels are RAW #125's
recorded family-window labels; every judge/arbiter answer is a LABELLED
FAKE.
"""
from __future__ import annotations

import pytest

from cutsell_worker.contracts import CandidateTake, ProcessingRequest, SourceAsset
from cutsell_worker.hybrid_composite_best_take import _choose_composite_replacements, _kept_complementary_rows
from cutsell_worker.hybrid_editorial import EditorialDecision, EditorialJudgeResult
from cutsell_worker.pipeline import build_flow_b_draft
from tests.test_cutsell_d289_contained_realization_closure import RecordedAnswersArbiter

A1_TEXT = "También me salían espinillas. Era como un rush, una alergia."
M_TEXT = ("También me salían espinillas en esta parte de aquí detrás de la oreja y todo el cuello que yo pensaba que era "
          "alergia pero era como espinillas de personas con problemas hormonales.")
L_TEXT = ("Otro síntoma era que me salían espinillas como si fuera una alergia de esta parte aquí detrás de la oreja y en "
          "el cuello. Me salía por temporadas.")
HAIR_TEXT = "Se me caía mucho el pelo cuando me lavaba el pelo perdía mucho pelo y pensaba que era por el agua del barco."


def take(cid, start, end, text, *, complete=True):
    return CandidateTake(cid, "src", 0, start, end, text, complete_idea=complete)


A1 = take("A1", 192.44, 198.12, A1_TEXT)
M = take("M", 198.88, 211.02, M_TEXT)
L = take("L", 213.34, 222.98, L_TEXT)
HAIR = take("HAIR", 226.74, 233.18, HAIR_TEXT)
# RAW #125 recorded family-window labels for these clips
LABELS = {"M": ("winner", 0.95), "A1": ("alternate", 0.75), "L": ("alternate", 0.8), "HAIR": ("winner", 0.95)}


def _suppressed(kept, semantic, rows):
    suppress, split, out = _choose_composite_replacements(tuple(kept), dict(semantic), rows)
    return suppress, split, out


# --- A. reproduction: RAW #126's shape (nothing deleted) before and after ---

def test_raw126_shape_before_no_restored_rows_means_the_monolith_is_never_judged():
    suppress, split, rows = _suppressed((A1, M, L, HAIR), LABELS, [])
    assert suppress == set() and split == set() and rows == []


def test_raw126_shape_after_kept_complementary_deliveries_replace_the_monolith():
    cand = _kept_complementary_rows((A1, M, L, HAIR), LABELS)
    assert {r["clip_id"] for r in cand} == {"A1", "L"}
    assert all(r["peer_clip_id"] == "M" and r["reason"] == "kept_complete_complementary_delivery_for_composite_best_take" for r in cand)
    suppress, split, rows = _suppressed((A1, M, L, HAIR), LABELS, cand)
    assert suppress == {"M"} and split == {"A1", "L"}
    row = rows[0]
    assert row["suppressed_peer_clip_id"] == "M" and row["composite_clip_ids"] == ["A1", "L"]
    assert row["peer_content_coverage"] >= 0.60 and row["peer_semantic_label"] == "winner"
    # RAW #125's recorded composite: coverage 0.6667, 10 shared tokens
    assert row["peer_content_coverage"] == pytest.approx(0.6667, abs=0.01) and row["shared_peer_content_tokens"] == 10


# --- B. controls ---

def test_control_same_topic_complementary_pieces_without_a_monolith_are_never_suppressed():
    labels = {"A1": ("alternate", 0.75), "L": ("winner", 0.9), "HAIR": ("winner", 0.95)}
    cand = _kept_complementary_rows((A1, L, HAIR), labels)
    # at most one piece points to L; a composite needs two pieces for one peer
    suppress, split, rows = _suppressed((A1, L, HAIR), labels, cand)
    assert suppress == set() and rows == []


def test_control_two_true_retakes_stay_for_the_family_competition():
    labels = {"M": ("winner", 0.95), "L": ("alternate", 0.8)}
    cand = _kept_complementary_rows((M, L), labels)
    suppress, split, rows = _suppressed((M, L), labels, cand)
    assert suppress == set() and rows == []  # one candidate only: BestTake decides M vs L


def test_control_an_incomplete_piece_or_a_failed_piece_never_composes():
    cand = _kept_complementary_rows((take("A1", 192.44, 198.12, A1_TEXT, complete=False), M, L), LABELS)
    assert {r["clip_id"] for r in cand} == {"L"}
    assert _suppressed((A1, M, L), LABELS, cand)[0] == set()
    failed = {**LABELS, "L": ("failed", 0.9)}
    cand2 = _kept_complementary_rows((A1, M, L), failed)
    # D-291.5.1: with L unusable, A1 is the only piece and it PRECEDES M --
    # an earlier attempt alone never makes the later monolith replaceable.
    assert cand2 == []
    assert _suppressed((A1, M, L), failed, cand2)[0] == set()


# --- B2. D-291.5.1 controls from the RAW #126 replay (gynecologist family) ---

X_TEXT = ("Al terminar mi contrato hablé con mi ginecóloga y le pedí todos los test que ella pudiera imaginarse o que me "
          "pudiera indicar.")
Y_TEXT = "Al terminar mi contrato le pedía a mi ginecóloga"
Z_TEXT = ("al terminar mi contrato cambié de ginecóloga y le pedí que me hiciera un test de todo lo que ella se pudiera "
          "imaginar y me pudiese indicar.")
X = take("X", 82.82, 90.60, X_TEXT)
Y = take("Y", 91.20, 94.34, Y_TEXT)
Z = take("Z", 95.52, 104.32, Z_TEXT)


def test_control_true_retakes_with_different_words_before_the_final_delivery_never_compose():
    """RAW #126 replay on D-291.5 (before D-291.5.1): X + Y, two earlier
    attempts of the same idea with different wording, replaced the final
    complete delivery Z. Both precede Z: the later complete retake dominates."""
    labels = {"X": ("alternate", 0.85), "Y": ("failed", 0.95), "Z": ("winner", 0.98)}
    cand = _kept_complementary_rows((X, Y, Z), labels)
    assert cand == []
    assert _suppressed((X, Y, Z), labels, cand)[0] == set()
    # even with Y usable and positively labelled, both pieces still precede Z
    labels2 = {"X": ("alternate", 0.85), "Y": ("keep", 0.8), "Z": ("winner", 0.98)}
    cand2 = _kept_complementary_rows((X, Y, Z), labels2)
    assert cand2 == [] and _suppressed((X, Y, Z), labels2, cand2)[0] == set()


def test_control_unlabelled_kept_pieces_carry_no_evidence_and_never_compose():
    """The RAW #126 replay window that fired had NO label for X or Y in its
    semantic map ('' / 0.0): a kept take without a judge label is not a
    proven usable delivery."""
    labels = {"M": ("winner", 0.95), "HAIR": ("winner", 0.95)}  # A1, L unlabelled
    cand = _kept_complementary_rows((A1, M, L, HAIR), labels)
    assert cand == []
    assert _suppressed((A1, M, L, HAIR), labels, cand)[0] == set()


def test_a_monolith_is_replaceable_only_when_a_complementary_piece_follows_it():
    # RAW #125/#126 skin shape: A1 before M, L after M -> both rows survive
    cand = _kept_complementary_rows((A1, M, L, HAIR), LABELS)
    assert {r["clip_id"]: r["candidate_after_peer"] for r in cand} == {"A1": False, "L": True}
    # the same two pieces both moved BEFORE the monolith -> earlier attempts, no composite
    a_early = take("A1", 170.0, 175.68, A1_TEXT)
    l_early = take("L", 176.5, 186.14, L_TEXT)
    m_late = take("M", 198.88, 211.02, M_TEXT)
    cand2 = _kept_complementary_rows((a_early, l_early, m_late, HAIR), LABELS)
    assert cand2 == []
    assert _suppressed((a_early, l_early, m_late, HAIR), LABELS, cand2)[0] == set()


def test_control_numbers_and_negations_of_the_monolith_must_survive_in_the_pair():
    m_num = take("M", 198.88, 211.02, M_TEXT.replace("problemas hormonales.", "problemas hormonales en un 5% de los casos."))
    cand = _kept_complementary_rows((A1, m_num, L), LABELS)
    assert {r["clip_id"] for r in cand} == {"A1", "L"}
    assert _suppressed((A1, m_num, L), LABELS, cand)[0] == set()  # the 5% is only in the monolith
    m_neg = take("M", 198.88, 211.02, M_TEXT.replace("que yo pensaba que era alergia", "que no era alergia"))
    cand2 = _kept_complementary_rows((A1, m_neg, L), LABELS)
    assert _suppressed((A1, m_neg, L), LABELS, cand2)[0] == set()  # the negation is only in the monolith


def test_control_inconclusive_or_weak_provider_labels_never_suppress():
    for peer_label in (("uncertain", 0.95), ("alternate", 0.95), ("winner", 0.7), ("keep", 0.6)):
        labels = {**LABELS, "M": peer_label}
        cand = _kept_complementary_rows((A1, M, L), labels)
        assert cand == [] and _suppressed((A1, M, L), labels, cand)[0] == set()


def test_control_pieces_that_do_not_cover_the_monolith_never_suppress():
    other = take("O", 213.34, 222.98, "Otro síntoma era que me dolía la cabeza todas las mañanas al despertar.")
    labels = {**LABELS, "O": ("alternate", 0.8)}
    cand = _kept_complementary_rows((A1, M, other), labels)
    assert {r["clip_id"] for r in cand} <= {"A1"}
    assert _suppressed((A1, M, other), labels, cand)[0] == set()


def test_control_a_piece_larger_than_its_peer_is_not_a_composite_piece():
    # the "peer" must be the monolith: a shorter winner never gets replaced by a longer piece plus another
    labels = {"A1": ("winner", 0.95), "M": ("alternate", 0.8), "L": ("alternate", 0.8)}
    cand = _kept_complementary_rows((A1, M, L), labels)
    assert all(r["peer_clip_id"] != "A1" for r in cand)


# --- C. end to end through the pipeline (composite chain installed by apply_composite_resolution) ---

class FakeJudge:
    def __init__(self, labels):
        self.labels = dict(labels)

    def judge(self, session):
        return EditorialJudgeResult(
            tuple(EditorialDecision(c.clip_id, *self.labels.get(c.clip_id, ("keep", 0.6)), "fake") for c in session.candidates),
            "fake", "fake-model", True, True, 200, 40,
        )


def _request():
    return ProcessingRequest(project_id="p", user_id="u", sources=(SourceAsset(
        source_asset_id="src", project_id="p", user_id="u", original_name="raw.mp4", source_order=0,
        duration_sec=400.0, uri="s3://b/raw.mp4",
    ),))


def test_end_to_end_the_monolith_is_replaced_before_grouping_and_the_short_delivery_is_never_lost():
    intro = take("I", 150.68, 158.14, "Síntomas que no me parecían sospechosos pero que ahora que lo analizo si eran sospechosos.")
    labels = {**LABELS, "I": ("winner", 0.95)}
    result = build_flow_b_draft(_request(), (intro, A1, M, L, HAIR), editorial_judge=FakeJudge(labels),
                                semantic_equivalence_arbiter=RecordedAnswersArbiter({}))
    draft = result.draft
    selected = [c.clip_id for c in draft.selected]
    assert selected == ["I", "A1", "L", "HAIR"], selected
    assert "M" in [c.clip_id for c in draft.discarded]
    groups = draft.diagnostics["take_group_members"]
    assert all(not ({"A1", "M"} <= set(g)) for g in groups)
    composite = [row for chunk in draft.diagnostics["hybrid_editorial_chunks"]
                 for row in ((chunk.get("hybrid_composite_best_take") or {}).get("composite_replacements") or [])]
    assert composite and composite[0]["suppressed_peer_clip_id"] == "M" and composite[0]["composite_clip_ids"] == ["A1", "L"]
