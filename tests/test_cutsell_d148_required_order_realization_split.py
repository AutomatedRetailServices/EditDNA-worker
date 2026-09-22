"""D-148 (Gate 6 correction, real RAW #118 audit): `required_order` conflated
real causal-order violations with preferred-realization-parity mismatches.

A `required_order` manifest anchor can bundle MULTIPLE facts into one
reference string (e.g. two Human-Gold sentences joined as one anchor). The
previous cursor + single-span `_find_semantic` search required that WHOLE
bundle to be covered by ONE candidate window -- so a candidate that keeps
every fact, in the correct order, but realizes them as its OWN separate,
differently-worded clips registered as `required_sequence_missing_or_
reordered`: a genuine order/content-loss failure and an editorial take-
choice difference were reported as the exact same defect.

Fix: `required_order` now runs the SAME general ordered-alignment primitive
(`align()`, D-032) already proven for `RECHUNKED`/`COMPOSITE` splits, and --
only for a gold segment `align()` still can't place -- falls back to the
SAME engine-confirmed equivalence-credit evidence (D-106) `meaning_
preservation` checks already use. A genuine MISSING fact (nothing in
`selected` or `discarded` covers it, or no engine equivalence links a
covering realization to anything selected at/after the right position)
still fails hard. Order itself is never weakened -- every fallback search is
bounded to start at or after the previous confirmed match's position.

Generic, non-medical-disease fixtures throughout.
"""
import json

from benchmarks.validate_video00_regression_qa import validate


def write_json(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def manifest():
    return {
        "schema_version": "cutsell.video00.regression_qa.v1",
        "baseline_run_id": 1,
        "checks": [
            {"id": "finding_before_conclusion", "kind": "required_order", "texts": [
                "we ran a scan that turned up something unusual in the sample and sent it for further testing.",
                "the follow-up test confirmed it was a serious condition. the symptoms I had were mild but there were signs looking back now.",
            ]},
        ],
    }


def test_order_intact_but_split_into_a_differently_worded_realization_passes_via_equivalence_credit():
    """The reference's compound second anchor (conclusion + symptom
    reflection) is realized by CutSell as TWO separate, differently-worded
    clips. A discarded clip closer to the reference's own phrasing exists,
    and the engine's own diagnostics already credit it as equivalent to the
    clip actually selected -- exactly the D-106 meaning-preservation
    mechanism, now reused here. This must PASS (order and content are both
    genuinely intact), with the realization difference recorded as a
    non-blocking parity note, never as a hard order failure."""
    result = {
        "selected": [
            {"clip_id": "finding", "text": "we ran a scan that turned up something unusual in the sample and sent it for further testing."},
            {"clip_id": "conclusion", "text": "the follow-up test confirmed it was a serious condition."},
            {"clip_id": "reflection_alt", "text": "thinking about it now I realize something was happening even though I never noticed at the time."},
        ],
        "discarded": [
            {"clip_id": "reflection_ref_wording", "text": "the symptoms I had were mild but there were signs looking back now."},
        ],
        "diagnostics": {
            "semantic_idea_equivalence": {
                "merges": [
                    {"left_clip_id": "reflection_ref_wording", "right_clip_id": "reflection_alt", "confidence": 0.93},
                ],
            },
        },
    }
    ok, report = validate(
        write_json(_tmp(), "result.json", result),
        write_json(_tmp(), "manifest.json", manifest()),
    )
    assert ok is True
    assert report["failed_check_count"] == 0
    parity_ids = [row["id"] for row in report["preferred_realization_parity"]["failed_checks"]]
    assert "finding_before_conclusion" in parity_ids
    reasons = {row["reason"] for row in report["preferred_realization_parity"]["failed_checks"]}
    assert "equivalent_realization_credited_not_reference_wording" in reasons


def _tmp():
    import tempfile
    from pathlib import Path
    return Path(tempfile.mkdtemp())


def test_genuinely_reordered_content_still_fails_hard():
    """Negative control: the conclusion is rendered BEFORE its finding --
    a real order violation, not a realization split. Must still FAIL,
    exactly as before this fix."""
    result = {
        "selected": [
            {"clip_id": "conclusion", "text": "the follow-up test confirmed it was a serious condition. the symptoms I had were mild but there were signs looking back now."},
            {"clip_id": "finding", "text": "we ran a scan that turned up something unusual in the sample and sent it for further testing."},
        ],
        "discarded": [],
        "diagnostics": {"semantic_idea_equivalence": {"merges": []}},
    }
    ok, report = validate(
        write_json(_tmp(), "result.json", result),
        write_json(_tmp(), "manifest.json", manifest()),
    )
    assert ok is False
    ids = {row["id"] for row in report["failed_checks"]}
    assert "finding_before_conclusion" in ids


def test_genuinely_missing_content_with_no_equivalence_evidence_still_fails_hard():
    """Negative control: the second anchor's content is not selected, not
    discarded, and has no engine equivalence evidence anywhere -- a real
    content-loss failure, not a realization split. The equivalence-credit
    fallback must never invent a match out of nothing."""
    result = {
        "selected": [
            {"clip_id": "finding", "text": "we ran a scan that turned up something unusual in the sample and sent it for further testing."},
            {"clip_id": "unrelated", "text": "we also talked about an entirely different unrelated topic that day."},
        ],
        "discarded": [],
        "diagnostics": {"semantic_idea_equivalence": {"merges": []}},
    }
    ok, report = validate(
        write_json(_tmp(), "result.json", result),
        write_json(_tmp(), "manifest.json", manifest()),
    )
    assert ok is False
    ids = {row["id"] for row in report["failed_checks"]}
    assert "finding_before_conclusion" in ids


def test_weak_equivalence_confidence_below_the_floor_does_not_credit_a_false_match():
    """Negative control: a discarded clip DOES cover the reference wording,
    and SOME selected clip exists, but the engine's own equivalence
    confidence for that specific pair is below the proven floor -- must not
    be silently credited (mirrors `_evaluate_meaning_preservation`'s own
    `insufficient_equivalence_evidence` caution, applied here to order)."""
    result = {
        "selected": [
            {"clip_id": "finding", "text": "we ran a scan that turned up something unusual in the sample and sent it for further testing."},
            {"clip_id": "reflection_alt", "text": "thinking about it now I realize something was happening even though I never noticed at the time."},
        ],
        "discarded": [
            {"clip_id": "reflection_ref_wording", "text": "the symptoms I had were mild but there were signs looking back now."},
        ],
        "diagnostics": {
            "semantic_idea_equivalence": {
                "merges": [
                    {"left_clip_id": "reflection_ref_wording", "right_clip_id": "reflection_alt", "confidence": 0.40},
                ],
            },
        },
    }
    ok, report = validate(
        write_json(_tmp(), "result.json", result),
        write_json(_tmp(), "manifest.json", manifest()),
    )
    assert ok is False
    ids = {row["id"] for row in report["failed_checks"]}
    assert "finding_before_conclusion" in ids


def test_equivalence_credited_match_cannot_be_placed_before_the_prior_confirmed_position():
    """Real order enforcement guard: even with strong equivalence evidence,
    the credited match must be searched only AT OR AFTER the previous
    confirmed gold segment's candidate position -- an equivalent
    realization appearing BEFORE the finding it depends on must still fail."""
    result = {
        "selected": [
            # The credited reflection realization is rendered FIRST, before
            # its own finding -- still a real order violation.
            {"clip_id": "reflection_alt", "text": "thinking about it now I realize something was happening even though I never noticed at the time."},
            {"clip_id": "finding", "text": "we ran a scan that turned up something unusual in the sample and sent it for further testing."},
        ],
        "discarded": [
            {"clip_id": "reflection_ref_wording", "text": "the symptoms I had were mild but there were signs looking back now."},
        ],
        "diagnostics": {
            "semantic_idea_equivalence": {
                "merges": [
                    {"left_clip_id": "reflection_ref_wording", "right_clip_id": "reflection_alt", "confidence": 0.93},
                ],
            },
        },
    }
    ok, report = validate(
        write_json(_tmp(), "result.json", result),
        write_json(_tmp(), "manifest.json", manifest()),
    )
    assert ok is False
    ids = {row["id"] for row in report["failed_checks"]}
    assert "finding_before_conclusion" in ids


def test_d152_an_earlier_unrelated_segment_is_never_absorbed_into_a_later_correct_match():
    """D-152 regression guard (real RAW #118 audit): an out-of-order,
    semantically UNRELATED earlier segment must never be silently absorbed
    into a wider multi-segment window just because combining it with the
    real, later, correctly-matching segment still clears coverage --
    `_find_semantic` must prefer the smallest sufficient window (the
    precise, later match), not the earliest-starting one. This is the exact
    shape `test_genuinely_reordered_content_still_fails_hard` above also
    covers through the full `validate()` path; this test isolates the
    underlying search primitive itself."""
    from benchmarks.validate_video00_regression_qa import _find_semantic
    texts = [
        "the follow-up test confirmed it was a serious condition and the symptoms were mild but there were signs looking back now",
        "we ran a scan that turned up something unusual in the sample and sent it for further testing",
    ]
    span = _find_semantic(texts, "we ran a scan that turned up something unusual in the sample and sent it for further testing.")
    assert span == (1, 2)  # the precise, later, single-segment match -- never (0, 2)


def test_d152_a_tiny_trailing_fragment_from_a_split_anchor_is_never_independently_required():
    """D-152 regression guard (real RAW #118 audit): sentence-splitting a
    compound anchor must never turn a near-content-free trailing fragment
    (e.g. a single leftover word) into its OWN independently-required fact
    -- the original, unsplit anchor tolerated this as noise within its own
    overall coverage check; requiring it standalone is a real regression a
    RAW #118 audit caught (the `pimples_micro_order` check)."""
    manifest_data = {
        "schema_version": "cutsell.video00.regression_qa.v1",
        "baseline_run_id": 1,
        "checks": [
            {"id": "tiny_trailing_fragment", "kind": "required_order", "texts": [
                "we discussed treatment options at length with the specialist over several visits. Then finally",
            ]},
        ],
    }
    result = {
        "selected": [
            {"text": "we discussed treatment options at length with the specialist over several visits."},
        ],
        "discarded": [],
        "diagnostics": {"semantic_idea_equivalence": {"merges": []}},
    }
    ok, report = validate(
        write_json(_tmp(), "result.json", result),
        write_json(_tmp(), "manifest.json", manifest_data),
    )
    assert ok is True
    assert report["failed_check_count"] == 0


def test_d152_a_substantial_gold_segment_is_never_excused_as_tiny():
    """Negative control: a genuinely substantial missing fact must still
    fail hard -- the tiny-fragment exemption only ever applies to segments
    at or below the established content-token floor."""
    manifest_data = {
        "schema_version": "cutsell.video00.regression_qa.v1",
        "baseline_run_id": 1,
        "checks": [
            {"id": "substantial_fact", "kind": "required_order", "texts": [
                "we discussed treatment options at length with the specialist over several visits. Then the specialist recommended a follow-up procedure",
            ]},
        ],
    }
    result = {
        "selected": [
            {"text": "we discussed treatment options at length with the specialist over several visits."},
        ],
        "discarded": [],
        "diagnostics": {"semantic_idea_equivalence": {"merges": []}},
    }
    ok, report = validate(
        write_json(_tmp(), "result.json", result),
        write_json(_tmp(), "manifest.json", manifest_data),
    )
    assert ok is False
    ids = {row["id"] for row in report["failed_checks"]}
    assert "substantial_fact" in ids
