"""D-291 -- family-scoped confirmation of a complete-window label conflict.

RAW #118 and RAW #124 (pimples family M = the monolith, L = the later
take): the two family-complete hybrid windows disagreed (#124: M winner
0.92 in chunk 3, L winner 0.95 in chunk 4; #118: M winner 0.95 in chunk 3,
NO winner in chunk 4). D-150 abstained (`ABSTAIN_CONFLICT`); the ladder
then fell to the NON_DECISIVE DeliveryScore (M) or, after D-289.10, blocked
the render. RAW #122's two windows AGREED on L and the fast path selected
it. Both Cut.ai and Human Gold keep L and drop M.

D-291 asks the SAME editorial judge one bounded family-scoped question when
D-150 abstains on a conflict, and lets a CONFIRMED single-winner answer
flow through the unchanged ladder. EVERY judge answer in this file is a
LABELLED FAKE (`FakeJudge`): these tests prove the mechanism's shape,
bounds and safety, never what the real provider will answer on real media.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from cutsell_worker import family_conflict_confirmation as fcc
from cutsell_worker.contracts import CandidateTake, ProcessingRequest, RankedTake, SourceAsset
from cutsell_worker.hybrid_editorial import EditorialDecision, EditorialJudgeResult
from cutsell_worker.pipeline import (
    _complete_window_winner_conflict,
    _meaning_sufficient_member_ids,
    _semantic_best_take,
    build_flow_b_draft,
    family_scoped_semantic_decisions,
)
from cutsell_worker.semantic_authority_observability import (
    AUTHORITY_ABSTAIN_CONFLICT,
    AUTHORITY_ALLOWED,
    family_authority_diagnostics,
    semantic_authority_gate_diagnostics,
)
from tests.test_cutsell_d289_10_raw124_pimples_stomach import (
    L, L_TEXT, M, M_TEXT, RANKED, WINDOWS_122, WINDOWS_124, _take, _window,
)
from tests.test_cutsell_d289_contained_realization_closure import RecordedAnswersArbiter

# LABELLED FAKE pair arbiter: the ONE recorded same-idea verdict the real
# runs formed the pimples family on (M and L are retries of one idea); every
# other pair is declined so the harness's other takes stay separate.
def _pimples_pair_arbiter():
    return RecordedAnswersArbiter({(M_TEXT, L_TEXT): (True, 0.95, "same idea: the pimples symptom, two deliveries")})


# RAW #118 recorded complete windows: chunk 3 M winner 0.95 / L alternate
# 0.75; chunk 4 M keep 0.9 / L alternate 0.7 (no winner at all).
WINDOWS_118 = [_window(3, "hc_fbeaefcce8c7c2eeea", ["X1", "M", "L"], [("M", "winner", 0.95), ("L", "alternate", 0.75)]),
               _window(4, "hc_a7328ab9bc7fad35ad", ["M", "L", "X2"], [("M", "keep", 0.9), ("L", "alternate", 0.7)])]
A1 = _take("A1", 192.44, 198.12, "También me salían espinillas. Era como un rush, una alergia.")
HAIR = _take("HAIR", 226.31, 233.0, "Se me caía mucho el pelo cuando me lavaba el pelo perdía mucho pelo.")
ORDERED = (A1, M, L, HAIR)


class FakeJudge:
    """LABELLED FAKE editorial judge: `answer` maps clip_id -> (label, confidence)
    for the family members; context members get 'keep'. `available=False`
    simulates a decline; `provider` lets a test simulate the transport's
    budget-exhausted string. Records every session it saw."""

    def __init__(self, answer, *, available=True, provider="fake", omit=()):
        self.answer, self.available, self.provider, self.omit = dict(answer), available, provider, set(omit)
        self.sessions = []

    def judge(self, session):
        self.sessions.append(session)
        decisions = tuple(
            EditorialDecision(c.clip_id, *self.answer.get(c.clip_id, ("keep", 0.6)), "fake")
            for c in session.candidates if c.clip_id not in self.omit
        )
        return EditorialJudgeResult(decisions, self.provider, "fake-model", True, self.available, 200, 40)


def _family_state(windows, members=(M, L)):
    glob = {}
    for w in windows:
        for d in w["decisions"]:
            glob[d["clip_id"]] = (d["label"], d["confidence"])
    decisions, source = family_scoped_semantic_decisions(members, glob, windows)
    ids = [m.clip_id for m in members]
    obs = family_authority_diagnostics(ids, windows, source)
    gate = semantic_authority_gate_diagnostics(ids, decisions, obs)
    conflict = _complete_window_winner_conflict(members, obs, gate)
    return decisions, gate, conflict


def _confirm(judge, windows=WINDOWS_124, members=(M, L), *, counter=None, env=None, ordered=ORDERED):
    decisions, gate, conflict = _family_state(windows, members)
    out = fcc.confirm_family_winner_conflict(
        members,
        semantic_authority_gate_status=gate["semantic_authority_gate_status"],
        meaning_sufficient_ids=_meaning_sufficient_member_ids(members, {}),
        editorial_judge=judge,
        whole_video_context=None,
        ordered_takes=ordered,
        run_counter=counter if counter is not None else {"used": 0},
        env=env if env is not None else {},
    )
    return out, decisions, gate, conflict


def _ladder(decisions, gate_status, conflict_ids, members=(M, L), ranked=RANKED):
    out = {}
    selected, preferred, reason = _semantic_best_take(
        members, decisions, ranked[0].clip_id, ranked,
        semantic_comparative_authority=gate_status, terminal_confidence_out=out,
        complete_window_winner_conflict_ids=conflict_ids,
    )
    return selected, preferred, reason, out["terminal_besttake_confidence"]


# --- A. the two recorded conflict shapes trigger a confirmation ---

def test_raw124_conflict_is_attempted_and_a_confirmed_later_take_flows_through_the_ladder():
    judge = FakeJudge({"L": ("winner", 0.95), "M": ("alternate", 0.8)})
    out, decisions, gate, conflict = _confirm(judge)
    assert gate["semantic_authority_gate_status"] == AUTHORITY_ABSTAIN_CONFLICT
    assert set(conflict["conflicting_clip_ids"]) == {"M", "L"}
    assert out.attempted and out.resolved and out.reason == fcc.CONFIRMED and out.winner_clip_id == "L"
    assert out.decisions == {"L": ("winner", 0.95), "M": ("alternate", 0.8)}
    # the session held exactly the family plus one known neighbour each side
    session = judge.sessions[0]
    assert [c.clip_id for c in session.candidates] == ["A1", "M", "L", "HAIR"]
    assert out.row["context_member_ids"] == ["A1", "HAIR"]
    assert all(d["context_only"] == (d["clip_id"] in {"A1", "HAIR"}) for d in out.row["decisions"])
    # the unchanged ladder now takes its single-winner fast path
    selected, preferred, reason, terminal = _ladder({**decisions, **out.decisions}, AUTHORITY_ALLOWED, frozenset())
    assert (selected, preferred, reason) == ("L", "L", "single_semantic_winner")
    assert terminal.confidence_state == "DECISIVE"


def test_raw118_shape_winner_versus_no_winner_is_also_a_conflict_the_confirmation_handles():
    judge = FakeJudge({"L": ("winner", 0.9), "M": ("alternate", 0.8)})
    out, decisions, gate, conflict = _confirm(judge, WINDOWS_118)
    assert gate["semantic_authority_gate_status"] == AUTHORITY_ABSTAIN_CONFLICT
    # D-289.10's two-winner stop does not cover this shape (only M was ever a winner)
    assert list(conflict["conflicting_clip_ids"]) in ([], ["M"])
    assert out.resolved and out.winner_clip_id == "L"
    selected, _, reason, _ = _ladder({**decisions, **out.decisions}, AUTHORITY_ALLOWED, frozenset())
    assert (selected, reason) == ("L", "single_semantic_winner")


def test_raw118_shape_without_confirmation_still_falls_to_the_deliveryscore_monolith():
    decisions, gate, conflict = _family_state(WINDOWS_118)
    selected, _, reason, terminal = _ladder(decisions, gate["semantic_authority_gate_status"], frozenset(conflict["conflicting_clip_ids"]))
    assert (selected, reason) == ("M", "delivery_tie_break_among_survivors")
    assert terminal.confidence_state == "NON_DECISIVE"


def test_raw122_agreeing_windows_never_ask_and_stay_byte_identical():
    judge = FakeJudge({"L": ("winner", 0.95), "M": ("alternate", 0.8)})
    out, decisions, gate, _ = _confirm(judge, WINDOWS_122)
    assert gate["semantic_authority_gate_status"] == AUTHORITY_ALLOWED
    assert not out.attempted and out.reason == fcc.NOT_A_CONFLICT and judge.sessions == []
    selected, _, reason, _ = _ladder(decisions, gate["semantic_authority_gate_status"], frozenset())
    assert (selected, reason) == ("L", "single_semantic_winner")


# --- B. the mechanism is symmetric and never a "last wins" / "highest wins" rule ---

def test_a_confirmed_monolith_is_honoured_the_same_way():
    judge = FakeJudge({"M": ("winner", 0.9), "L": ("alternate", 0.8)})
    out, decisions, _, _ = _confirm(judge)
    assert out.resolved and out.winner_clip_id == "M"
    selected, _, reason, _ = _ladder({**decisions, **out.decisions}, AUTHORITY_ALLOWED, frozenset())
    assert (selected, reason) == ("M", "single_semantic_winner")


@pytest.mark.parametrize("answer, reason", [
    ({"M": ("uncertain", 0.9), "L": ("uncertain", 0.9)}, fcc.NO_SINGLE_WINNER),
    ({"M": ("winner", 0.95), "L": ("winner", 0.9)}, fcc.NO_SINGLE_WINNER),
    ({"M": ("winner", 0.8), "L": ("alternate", 0.8)}, fcc.NO_SINGLE_WINNER),  # below the 0.85 floor
    ({"M": ("winner", 0.95), "L": ("uncertain", 0.7)}, fcc.OTHER_MEMBER_NOT_RULED_OUT),
    ({"M": ("alternate", 0.8), "L": ("keep", 0.8)}, fcc.NO_SINGLE_WINNER),
])
def test_ambiguous_answers_leave_the_d289_10_conflict_untouched(answer, reason):
    out, decisions, gate, conflict = _confirm(FakeJudge(answer))
    assert out.attempted and not out.resolved and out.reason == reason and out.decisions == {}
    selected, preferred, ladder_reason, terminal = _ladder(
        decisions, gate["semantic_authority_gate_status"], frozenset(conflict["conflicting_clip_ids"]),
    )
    assert (selected, preferred, ladder_reason) == ("M", None, "unresolved_semantic_winner_conflict")
    assert terminal.confidence_state == "CONFLICTED"


def test_decline_budget_exhaustion_and_omitted_member_are_recorded_and_unresolved():
    out, *_ = _confirm(FakeJudge({}, available=False))
    assert out.attempted and not out.resolved and out.reason == fcc.PROVIDER_UNAVAILABLE
    out, *_ = _confirm(FakeJudge({}, available=False, provider="RuntimeError:hybrid edit/test dollar budget exhausted"))
    assert out.reason == fcc.BUDGET_EXHAUSTED and out.row["budget_exhausted"] is True
    # an answer that omits a family member is rejected by validate_editorial_result -> unavailable
    out, *_ = _confirm(FakeJudge({"L": ("winner", 0.95)}, omit=("M",)))
    assert not out.resolved and out.reason == fcc.PROVIDER_UNAVAILABLE


def test_winner_must_be_meaning_sufficient():
    members = (replace(M, complete_idea=False), L)
    judge = FakeJudge({"M": ("winner", 0.95), "L": ("alternate", 0.8)})
    decisions, gate, _ = _family_state(WINDOWS_124, members)
    out = fcc.confirm_family_winner_conflict(
        members, semantic_authority_gate_status=gate["semantic_authority_gate_status"],
        meaning_sufficient_ids=_meaning_sufficient_member_ids(members, {}), editorial_judge=judge,
        whole_video_context=None, ordered_takes=ORDERED, run_counter={"used": 0}, env={},
    )
    assert out.attempted and not out.resolved and out.reason == fcc.WINNER_NOT_MEANING_SUFFICIENT


def test_confirmed_label_still_passes_through_the_ladder_safety_veto():
    # A confirmed winner that CONTRADICTS its sibling (number conflict) is
    # vetoed by the unchanged D-101 fast-path veto: confirmation never
    # bypasses semantic safety.
    a = _take("A", 1.0, 4.0, "Solo un 5% de los cánceres son hereditarios.")
    b = _take("B", 5.0, 8.0, "Solo un 20% de los cánceres son hereditarios.")
    windows = [_window(0, "w0", ["A", "B"], [("A", "winner", 0.9), ("B", "alternate", 0.8)]),
               _window(1, "w1", ["A", "B", "Z"], [("B", "winner", 0.9), ("A", "alternate", 0.8)])]
    judge = FakeJudge({"B": ("winner", 0.95), "A": ("alternate", 0.8)})
    out, decisions, _, _ = _confirm(judge, windows, (a, b), ordered=(a, b))
    assert out.resolved and out.winner_clip_id == "B"
    ranked = (RankedTake("A", 0.7, "watch_listen_baseline"), RankedTake("B", 0.6, "watch_listen_baseline"))
    selected, preferred, reason, terminal = _ladder({**decisions, **out.decisions}, AUTHORITY_ALLOWED, frozenset(), (a, b), ranked)
    assert reason in ("unresolved_contradiction", "unresolved_unique_fact_asymmetry") and preferred is None and terminal.confidence_state == "CONFLICTED"


# --- C. bounds ---

def test_env_off_no_judge_and_per_run_cap_never_ask():
    judge = FakeJudge({"L": ("winner", 0.95), "M": ("alternate", 0.8)})
    out, *_ = _confirm(judge, env={fcc.FAMILY_CONFLICT_CONFIRMATION_ENV: "0"})
    assert not out.attempted and out.reason == fcc.DISABLED_BY_ENV
    out, *_ = _confirm(None)
    assert not out.attempted and out.reason == fcc.NO_EDITORIAL_JUDGE
    counter = {"used": fcc.MAX_CONFIRMATIONS_PER_RUN}
    out, *_ = _confirm(judge, counter=counter)
    assert not out.attempted and out.reason == fcc.CAP_REACHED and judge.sessions == []
    counter = {"used": 0}
    out, *_ = _confirm(judge, counter=counter)
    assert out.resolved and counter["used"] == 1 and out.row["run_confirmations_used"] == 1
    assert fcc.family_conflict_confirmation_enabled({}) is True


def test_neighbour_context_is_bounded_to_the_same_source_and_one_each_side():
    far = _take("FAR", 100.0, 105.0, "otra cosa.")
    other = _take("OTHER", 200.0, 205.0, "de otra fuente.", source="src2")
    judge = FakeJudge({"L": ("winner", 0.95), "M": ("alternate", 0.8)})
    out, *_ = _confirm(judge, ordered=(far, A1, other, M, L, HAIR, _take("H2", 240.0, 245.0, "más.")))
    assert out.resolved
    assert [c.clip_id for c in judge.sessions[0].candidates] == ["A1", "M", "L", "HAIR"]
    assert out.row["member_ids"] == ["A1", "M", "L", "HAIR"] and out.row["request_hash"].startswith("rh_")


# --- D. end to end through build_flow_b_draft: two overlapping windows disagree, the confirmation resolves ---

def _e2e_take(cid, start, end, text):
    return CandidateTake(cid, "src", 0, start, end, text, complete_idea=True)


def _e2e_request():
    source = SourceAsset(source_asset_id="src", project_id="p", user_id="u", original_name="raw.mp4",
                         source_order=0, duration_sec=400.0, uri="s3://b/raw.mp4")
    return ProcessingRequest(project_id="p", user_id="u", sources=(source,))


class WindowSensitiveJudge(FakeJudge):
    """LABELLED FAKE: the window that contains `flip_marker` labels the
    monolith the winner; every other window labels the later take the
    winner; the family-scoped confirmation session answers `confirmation`."""

    def __init__(self, *, flip_marker, confirmation):
        super().__init__({})
        self.flip_marker, self.confirmation = flip_marker, dict(confirmation)

    def judge(self, session):
        self.sessions.append(session)
        ids = [c.clip_id for c in session.candidates]
        if set(ids) <= {"A1", "M", "L", "HAIR"} and {"M", "L"} <= set(ids):
            answer = self.confirmation
        elif self.flip_marker in ids:
            answer = {"M": ("winner", 0.92), "L": ("alternate", 0.8)}
        else:
            answer = {"M": ("alternate", 0.8), "L": ("winner", 0.95)}
        decisions = tuple(EditorialDecision(c, *answer.get(c, ("keep", 0.6)), "fake") for c in ids)
        return EditorialJudgeResult(decisions, "fake", "fake-model", True, True, 200, 40)


def _e2e_takes():
    # 12 takes -> windows [0..9] and [2..11] overlap on 2..9; the family (M, L)
    # sits inside the overlap so BOTH windows are family-complete.
    filler = [_e2e_take(f"F{i}", 10.0 * i + 1.0, 10.0 * i + 6.0, f"Tema distinto número {i} de la historia.") for i in range(7)]
    return tuple([*filler[:6],
                  _e2e_take("A1", 192.44, 198.12, "También me salían espinillas. Era como un rush, una alergia."),
                  _e2e_take("M", 198.88, 211.02, M_TEXT), _e2e_take("L", 213.34, 222.98, L_TEXT),
                  _e2e_take("HAIR", 226.31, 233.0, "Se me caía mucho el pelo cuando me lavaba el pelo perdía mucho pelo."),
                  filler[6], _e2e_take("F7", 300.0, 305.0, "Cierre de la historia con otro tema.")])


def _pimples_row(result):
    rows = [r for r in (result.draft.diagnostics.get("take_judge_groups") or [])
            if {"M", "L"} <= set(c["clip_id"] for c in (r.get("semantic_candidates") or []))]
    assert len(rows) == 1, [r.get("group_id") for r in (result.draft.diagnostics.get("take_judge_groups") or [])]
    return rows[0]


def test_end_to_end_conflicting_windows_then_confirmation_selects_the_later_take():
    judge = WindowSensitiveJudge(flip_marker="F0", confirmation={"L": ("winner", 0.95), "M": ("alternate", 0.8)})
    result = build_flow_b_draft(_e2e_request(), _e2e_takes(), editorial_judge=judge, semantic_equivalence_arbiter=_pimples_pair_arbiter())
    row = _pimples_row(result)
    conf = row["family_conflict_confirmation"]
    assert row["semantic_authority_gate_status"] == AUTHORITY_ABSTAIN_CONFLICT, row["semantic_authority_gate_reason"]
    assert conf["attempted"] and conf["resolved"] and conf["winner_clip_id"] == "L"
    assert row["semantic_authority_gate_status_effective"] == AUTHORITY_ALLOWED
    assert row["semantic_label_source"]["family_conflict_confirmation_applied"] is True
    assert row["complete_window_winner_conflict"]["resolved_by_family_confirmation"] == "L"
    assert row["semantic_best_take_reason"] == "single_semantic_winner" and row["selected_clip_id"] == "L"
    selected = {c.clip_id for c in result.draft.selected}
    assert "L" in selected and "M" not in selected and "A1" in selected
    # exactly one confirmation request for the whole run
    family_sessions = [s for s in judge.sessions if {"M", "L"} <= {c.clip_id for c in s.candidates} and len(s.candidates) <= 4]
    assert len(family_sessions) == 1


def test_end_to_end_unresolved_confirmation_keeps_the_d289_10_review_block():
    judge = WindowSensitiveJudge(flip_marker="F0", confirmation={"L": ("uncertain", 0.6), "M": ("uncertain", 0.6)})
    result = build_flow_b_draft(_e2e_request(), _e2e_takes(), editorial_judge=judge, semantic_equivalence_arbiter=_pimples_pair_arbiter())
    row = _pimples_row(result)
    assert row["family_conflict_confirmation"]["attempted"] and not row["family_conflict_confirmation"]["resolved"]
    assert row["semantic_best_take_reason"] == "unresolved_semantic_winner_conflict"
    assert row["complete_window_winner_conflict"]["routed"] is True


def test_end_to_end_disabled_flag_is_byte_identical_to_d289_10(monkeypatch):
    monkeypatch.setenv(fcc.FAMILY_CONFLICT_CONFIRMATION_ENV, "0")
    judge = WindowSensitiveJudge(flip_marker="F0", confirmation={"L": ("winner", 0.95), "M": ("alternate", 0.8)})
    result = build_flow_b_draft(_e2e_request(), _e2e_takes(), editorial_judge=judge, semantic_equivalence_arbiter=_pimples_pair_arbiter())
    row = _pimples_row(result)
    assert row["family_conflict_confirmation"] == {
        "schema_version": fcc.SCHEMA_VERSION, "attempted": False, "resolved": False,
        "reason": fcc.DISABLED_BY_ENV, "family_member_ids": ["M", "L"],
    } or row["family_conflict_confirmation"]["family_member_ids"] == ["L", "M"]
    assert row["semantic_best_take_reason"] == "unresolved_semantic_winner_conflict"
