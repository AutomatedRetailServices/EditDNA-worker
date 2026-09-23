"""D-097.11 (Product Owner decision, real RAW #121 audit) -- a discarded
clip's incidental CONTEXTUAL atom (e.g. a bare year) must not, by itself,
keep the coarse whole-video content_loss check blocking Selection Freeze
when the winning realization already preserves the same event/diagnosis/
consequence.

Real RAW #121 evidence: `clip_e8a18e0e6a19eea1fcf3` was flagged
`REAL_CONTENT_LOSS`/`blocking: true` with `missing_critical_atoms=["2023"]`
-- but `semantic_atom_importance.classify_number_atom` already correctly
classified "2023" as CONTEXTUAL (an incidental year in an ordinary temporal
aside, per D-031), which `blocks_freeze` already treats as non-blocking ON
ITS OWN. The row still blocked because `content_loss` (the SEPARATE, coarse
whole-video missing-content-token check) does not exclude an atom already
independently proven safe-to-lose from its own coverage math -- the atom is
"CONTEXTUAL" for freeze-blocking purposes yet still counted as ordinary lost
vocabulary. For an ungrouped/pre-group clip (this exact real shape: no
`take_judge_groups` entry), the only prior fallback was `_pre_group_restart_
credit`, which asks a semantic-equivalence ARBITER -- exactly the source of
D-097.11's documented run-to-run instability (three runs, three different
verdicts on the same evidence).

Fix: a new deterministic credit path, tried before any arbiter call. It
recomputes whole-video coverage EXCLUDING only the atom(s) already
classified CONTEXTUAL, using the SAME 0.45 floor the original check uses.
It fires only when EVERY missing critical atom on the clip is CONTEXTUAL
(a single CRITICAL/UNCERTAIN atom -- a measurement, price, dose, age,
disease stage, correction or required-chronology date -- keeps the row
blocking exactly as before); substantial OTHER missing content unrelated to
any contextual atom still blocks. No Video00 text/ids; generic fixtures.
"""
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.final_story_coherence_validation import apply_final_story_coherence_validation


def clip(clip_id, start, end, text, *, selected, source="src"):
    return DraftClip(
        clip_id=clip_id, source_asset_id=source, source_order=0,
        start=start, end=end, text=text, caption_text=text, selected=selected,
    )


def draft(*, selected, discarded):
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=selected, alternates=(), discarded=discarded,
        diagnostics={"take_judge_groups": []},
    )


def _row_for(diag, clip_id):
    return next((r for r in diag["lost_semantic_atoms"] if r["clip_id"] == clip_id), None)


WINNER_TEXT = (
    "I dealt with ongoing digestive issues and doctors ran tests that confirmed "
    "gastritis then gave me treatment for it."
)


def test_incidental_year_alone_is_credited_deterministically_no_arbiter():
    winner = clip("winner", 0.0, 5.0, WINNER_TEXT, selected=True)
    discard = clip(
        "discard", 10.0, 15.0,
        "I had ongoing stomach issues at one point back then and doctors eventually ran "
        "several tests and they confirmed gastritis during 2023, after some treatment.",
        selected=False,
    )
    d = draft(selected=(winner,), discarded=(discard,))

    out = apply_final_story_coherence_validation(d)
    diag = out.diagnostics["final_story_coherence_validation"]
    row = _row_for(diag, "discard")

    assert row is not None
    assert row["blocking"] is False
    assert row["classification"] == "SEMANTICALLY_COVERED_BY_SELECTED_REALIZATION"
    assert row["content_loss_suppressed_by"] == "contextual_atom_excluded_coverage_credit"
    assert diag["lost_critical_claims"] == []


def test_substantial_unrelated_content_loss_still_blocks_despite_contextual_atom():
    winner = clip("winner", 0.0, 5.0, WINNER_TEXT, selected=True)
    discard = clip(
        "discard", 10.0, 15.0,
        "I had ongoing stomach issues at one point back then and doctors eventually ran "
        "several tests overnight while I traveled abroad and they confirmed gastritis "
        "during 2023 after some unusual treatment nobody expected.",
        selected=False,
    )
    d = draft(selected=(winner,), discarded=(discard,))

    out = apply_final_story_coherence_validation(d)
    diag = out.diagnostics["final_story_coherence_validation"]
    row = _row_for(diag, "discard")

    assert row is not None
    assert row["blocking"] is True
    assert row["classification"] == "REAL_CONTENT_LOSS"
    assert row.get("content_loss_suppressed_by") is None


def test_a_genuinely_critical_missing_number_still_blocks():
    winner = clip("winner", 0.0, 5.0, WINNER_TEXT, selected=True)
    discard = clip(
        "discard", 10.0, 15.0,
        "I had ongoing stomach issues at one point back then and doctors eventually ran "
        "several tests and they confirmed gastritis during 2023, and it cost me $49 for "
        "the treatment.",
        selected=False,
    )
    d = draft(selected=(winner,), discarded=(discard,))

    out = apply_final_story_coherence_validation(d)
    diag = out.diagnostics["final_story_coherence_validation"]
    row = _row_for(diag, "discard")

    assert row is not None
    assert row["blocking"] is True
    assert row["classification"] == "REAL_CONTENT_LOSS"


def test_a_date_carrying_required_chronology_still_blocks():
    winner = clip("winner", 0.0, 5.0, WINNER_TEXT, selected=True)
    discard = clip(
        "discard", 10.0, 15.0,
        "I had ongoing stomach issues at one point back then and doctors eventually ran "
        "several tests and they confirmed gastritis during 2023, and before that I felt "
        "totally fine.",
        selected=False,
    )
    d = draft(selected=(winner,), discarded=(discard,))

    out = apply_final_story_coherence_validation(d)
    diag = out.diagnostics["final_story_coherence_validation"]
    row = _row_for(diag, "discard")

    assert row is not None
    assert row["blocking"] is True
    assert row["classification"] == "REAL_CONTENT_LOSS"


def test_a_missing_age_still_blocks():
    winner = clip("winner", 0.0, 5.0, WINNER_TEXT, selected=True)
    discard = clip(
        "discard", 10.0, 15.0,
        "I had ongoing stomach issues and doctors eventually ran several tests and "
        "confirmed gastritis when I was 42 years old, after some treatment.",
        selected=False,
    )
    d = draft(selected=(winner,), discarded=(discard,))

    out = apply_final_story_coherence_validation(d)
    diag = out.diagnostics["final_story_coherence_validation"]
    row = _row_for(diag, "discard")

    assert row is not None
    assert row["blocking"] is True
    assert row["classification"] == "REAL_CONTENT_LOSS"


def test_no_missing_critical_atoms_is_unaffected_by_the_new_credit_path():
    # A pure content_loss row with ZERO missing_critical atoms must be
    # completely untouched by this change -- `all(...)` on an empty list is
    # vacuously True in Python, so the gate explicitly requires a non-empty
    # `classifications` list; this proves that guard holds.
    winner = clip("winner", 0.0, 5.0, WINNER_TEXT, selected=True)
    discard = clip(
        "discard", 10.0, 15.0,
        "One time I dealt with a totally unrelated shipping delay problem that had "
        "nothing to do with any of this at all whatsoever honestly.",
        selected=False,
    )
    d = draft(selected=(winner,), discarded=(discard,))

    out = apply_final_story_coherence_validation(d)
    diag = out.diagnostics["final_story_coherence_validation"]
    row = _row_for(diag, "discard")

    assert row is not None
    assert row["missing_critical_atoms"] == []
    assert row["blocking"] is True
    assert row.get("content_loss_suppressed_by") is None
