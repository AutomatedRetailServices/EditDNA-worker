"""D-097.8 -- two gaps RAW 34043967265 (head 23306a4) exposed on the FIRST
deliverable MP4 that carried both the clean gynaecologist retry and the
diagnosis sentence.

R9 -- the arbiter's per-request pair budget (14 of 55 candidate pairs) was
spent on ADJACENT pairs of different sentences (nodule -> biopsy result,
"cuídate" -> "no, no, no") because temporal proximity carried a full unit
of weight in `_pair_priority_score` while content overlap tops out at 1.0;
the abandoned stomach attempt <-> clean gastritis delivery pair (shared
opening, "endoscopía", "donde", 14 s apart) was never asked, the two
abandoned attempts formed their own family and its incomplete winner
("... me diagnosticaron con...") played for 7.96 s. Proximity is now a
tie-break (weight 0.25); overlap and restart/continuation evidence lead.

R10 -- the video ENDED on "No, no, no, no, no." (Hybrid `bts` 0.95,
`dense_physical_reset:5`, `visual_fumble:0.85`): D-081 defers every
semantic delete to the authoritative resolution boundary, but a SINGLETON
never reaches Best Take's contest, so the corroborated-bts basis was never
applied by anyone. A lone `bts` realization at or above the usable floor
with deterministic local corroboration is now a no-usable-realization
outcome (lost by decision, recorded like a dropped family). A lone
`failed` delivery stays kept exactly as D-097.B ruled; labels alone never
drop anything.
"""
from __future__ import annotations

from cutsell_worker.contracts import CandidateTake
from cutsell_worker.final_story_coherence_validation import _no_usable_realization_clip_ids
from cutsell_worker.pipeline import _semantic_best_take
from cutsell_worker.take_grouping_provider import _pair_priority_score, _rank_candidate_pairs
from cutsell_worker.take_judge import RankedTake


def _take(clip_id, start, end, text, *, complete=True):
    return CandidateTake(
        clip_id=clip_id, source_asset_id="src", source_order=0, start=start, end=end, text=text, complete_idea=complete,
    )


# ------------------------------------------------------------------ R9 ranking

NODULE = _take("nodule", 128.0, 134.2, "En la sonografía de tiroides apareció un nódulo sospechoso de tres centímetros que se mandó a biopsia.")
BIOPSY = _take("biopsy", 135.0, 138.6, "La biopsia confirmó que era un cáncer papilar de tiroides.")
STOMACH_ABANDONED = _take("stomach_a", 236.2, 244.2, "Tuve problemas estomacales a un tiempo en donde se me hizo una endoscopía y me diagnosticaron con")
STOMACH_CLEAN = _take("stomach_c", 258.9, 268.5, "Tuve problemas de digestión en donde me hicieron una endoscopía y dijeron que tenía gastritis, nada severo, y me mandaron tres meses con pastillas.")


def test_a_far_retry_with_shared_content_outranks_an_adjacent_pair_of_different_sentences():
    adjacent = _pair_priority_score(NODULE, BIOPSY, gap_sec=0.8)
    retry = _pair_priority_score(STOMACH_ABANDONED, STOMACH_CLEAN, gap_sec=14.7)
    assert retry > adjacent


def test_rank_puts_the_retry_pair_inside_a_small_budget_ahead_of_sequential_neighbours():
    takes = {t.clip_id: t for t in (NODULE, BIOPSY, STOMACH_ABANDONED, STOMACH_CLEAN)}
    raw = ((0, 1, "nodule", "biopsy"), (2, 3, "stomach_a", "stomach_c"))
    ranked = _rank_candidate_pairs(raw, takes)
    assert ranked[0] == (2, 3, "stomach_a", "stomach_c")


def test_proximity_still_breaks_ties_between_equally_overlapping_pairs():
    a = _take("a", 0.0, 2.0, "we launched the new product line today")
    near = _take("near", 3.0, 5.0, "today we finally launched our new product line")
    far = _take("far", 40.0, 42.0, "today we finally launched our new product line")
    assert _pair_priority_score(a, near, gap_sec=1.0) > _pair_priority_score(a, far, gap_sec=38.0)


# --------------------------------------------------------- R10 bts singleton

BTS = _take("bts", 364.4, 366.9, "No, no, no, no, no.")


def _ranked(clip_id, score=0.5):
    return [RankedTake(clip_id=clip_id, score=score, reason="watch_listen_baseline")]


def test_a_corroborated_bts_singleton_is_a_no_usable_realization():
    selected, preferred, reason = _semantic_best_take(
        (BTS,), {"bts": ("bts", 0.95)}, "bts", _ranked("bts"),
        semantic_delete_recommended={"bts": True}, deterministic_unusable={"bts": True},
    )
    assert selected is None and preferred is None and reason == "single_bts_unusable"


def test_a_bts_label_alone_never_drops_a_singleton():
    selected, _p, reason = _semantic_best_take(
        (BTS,), {"bts": ("bts", 0.95)}, "bts", _ranked("bts"),
        semantic_delete_recommended={"bts": True}, deterministic_unusable={"bts": False},
    )
    assert selected == "bts" and reason == "single_member_no_contest"


def test_a_bts_label_below_the_floor_keeps_the_singleton():
    selected, _p, reason = _semantic_best_take(
        (BTS,), {"bts": ("bts", 0.80)}, "bts", _ranked("bts"),
        semantic_delete_recommended={"bts": True}, deterministic_unusable={"bts": True},
    )
    assert selected == "bts" and reason == "single_member_no_contest"


def test_a_failed_singleton_is_unchanged_from_d097_b():
    lone = _take("lone", 0.0, 5.0, "Tuve problemas de estómago en una temporada.")
    selected, _p, reason = _semantic_best_take(
        (lone,), {"lone": ("failed", 0.99)}, "lone", _ranked("lone"),
        semantic_delete_recommended={"lone": True}, deterministic_unusable={"lone": True},
    )
    assert selected == "lone" and reason == "single_member_no_contest"


def test_a_recorded_singleton_drop_is_read_as_a_decision_by_the_validator():
    class _Draft:
        diagnostics = {"take_judge_groups": [
            {"group_id": "g1", "no_usable_realization": True, "ranked": [{"clip_id": "bts", "score": 0.5}], "member_usability": {}},
            {"group_id": "g2", "no_usable_realization": False, "ranked": [{"clip_id": "a"}, {"clip_id": "b"}]},
        ]}
    assert _no_usable_realization_clip_ids(_Draft()) == {"bts"}
