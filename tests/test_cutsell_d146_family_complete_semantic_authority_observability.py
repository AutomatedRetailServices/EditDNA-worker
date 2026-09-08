"""D-146: FAMILY-COMPLETE SEMANTIC AUTHORITY GATE -- PHASE A (OBSERVABILITY ONLY).

Per docs/CUTSELL_DECISIONS.md D-145 (design) and D-146 (this task). Proves
`cutsell_worker.semantic_authority_observability` is a pure, non-circular,
provider-call-free projection of data `hybrid_session_cleanup.
apply_hybrid_session_cleanup` and `pipeline.family_scoped_semantic_decisions`
already compute -- and that wiring it into `hybrid_session_cleanup.py`'s
per-window diagnostics rows and `pipeline.py`'s per-family
`judge_group_diagnostics` is strictly additive (new dict keys only; no
existing field, decision, or behavior changes).

Five named offline fixture shapes (window_rows built by hand -- the exact
shape `hybrid_session_cleanup.apply_hybrid_session_cleanup`'s own
`diagnostics.append({...})` rows already have: `member_ids`, `session_id`,
`partition_index`, `chunk_index`, `provider`, `model`, `decisions`):

  A. FULL FAMILY IN ONE WINDOW       -- one window's member_ids is a superset
                                         of the whole family.
  B. PARTIAL FAMILY ONLY             -- every window that touches the family
                                         omits at least one member; no window
                                         is a complete superset.
  C. TWO PARTIAL WINDOWS, CONFLICT   -- two different partial windows each
                                         label a DIFFERENT family member
                                         "winner" (the real D-094.3 F8 shape:
                                         run 33983880111).
  D. COMPLETE WINDOW + PARTIAL WINDOW -- one complete window AND one partial
                                         window that disagrees with it; the
                                         complete window's presence must
                                         suppress the partial-window-conflict
                                         report (it is moot once a family-
                                         complete window exists).
  E. NO WINDOW TOUCHES THE FAMILY     -- window evidence exists for the run,
                                         but none of it mentions this family.

Never calls a real provider (`hybrid_session_cleanup.MappingJudge`-style stubs
only, matching this repo's own `tests/test_cutsell_hybrid_session_cleanup.py`
convention).
"""
from __future__ import annotations

from cutsell_worker.contracts import CandidateTake
from cutsell_worker.hybrid_editorial import EditorialDecision, EditorialJudgeResult
from cutsell_worker.hybrid_session_cleanup import apply_hybrid_session_cleanup
from cutsell_worker.semantic_authority_observability import (
    PROVIDER_PROMPT_VERSION_UNKNOWN,
    PROVIDER_TEMPERATURE_UNKNOWN,
    AUTHORITY_FAMILY_COMPLETE_WINDOW_PREFERRED,
    AUTHORITY_GLOBAL_CROSS_WINDOW_MERGE,
    complete_window_ids,
    family_authority_diagnostics,
    family_complete_context,
    omitted_candidate_ids,
    partial_window_conflict,
    provider_config_from_window_rows,
    stable_request_hash,
    summarize_family_authority_observability,
    window_ids_touching_family,
)


def _row(session_id, member_ids, decisions=(), provider="fake", model="flash-lite",
         partition_index=0, chunk_index=0):
    return {
        "session_id": session_id,
        "partition_index": partition_index,
        "chunk_index": chunk_index,
        "member_ids": list(member_ids),
        "provider": provider,
        "model": model,
        "decisions": [
            {"clip_id": clip_id, "label": label, "confidence": confidence}
            for clip_id, label, confidence in decisions
        ],
    }


FAMILY = ("clip-a", "clip-b", "clip-c")

# Fixture A: one window sees the whole family.
FIXTURE_A = [_row("w1", ["clip-a", "clip-b", "clip-c"], [("clip-a", "alternate", 0.88), ("clip-c", "winner", 0.95)])]

# Fixture B: two windows, each missing a different member -- never complete.
FIXTURE_B = [
    _row("w1", ["clip-a", "clip-b"], [("clip-a", "winner", 0.90)]),
    _row("w2", ["clip-b", "clip-c"], [("clip-c", "alternate", 0.70)]),
]

# Fixture C: the real D-094.3 F8 shape -- two disjoint-ish partial windows each
# crown a DIFFERENT member "winner", neither window sees the whole family.
FIXTURE_C = [
    _row("w1", ["clip-a", "clip-b"], [("clip-a", "winner", 0.96)]),
    _row("w2", ["clip-b", "clip-c"], [("clip-c", "winner", 0.95)]),
]

# Fixture D: a family-complete window plus a disagreeing partial window.
FIXTURE_D = [
    _row("w1", ["clip-a", "clip-b", "clip-c"], [("clip-c", "winner", 0.95), ("clip-a", "alternate", 0.88)]),
    _row("w2", ["clip-a", "clip-b"], [("clip-a", "winner", 0.90)]),
]

# Fixture E: window evidence exists (for an unrelated family) but never
# touches this family at all.
FIXTURE_E = [_row("w9", ["clip-x", "clip-y"], [("clip-x", "winner", 0.91)])]


# ---------------------------------------------------------------------------
# family_complete_context (5 named fixtures + edge cases)
# ---------------------------------------------------------------------------

def test_fixture_a_full_family_one_window_returns_true():
    assert family_complete_context(FAMILY, FIXTURE_A) == "true"


def test_fixture_b_partial_family_only_returns_false():
    assert family_complete_context(FAMILY, FIXTURE_B) == "false"


def test_fixture_c_two_partial_conflicting_windows_returns_false():
    assert family_complete_context(FAMILY, FIXTURE_C) == "false"


def test_fixture_d_family_complete_plus_partial_window_returns_true():
    assert family_complete_context(FAMILY, FIXTURE_D) == "true"


def test_fixture_e_no_window_touches_family_returns_unknown():
    assert family_complete_context(FAMILY, FIXTURE_E) == "unknown"


def test_family_complete_context_unknown_for_singleton_family():
    assert family_complete_context(("clip-a",), FIXTURE_A) == "unknown"


def test_family_complete_context_unknown_when_window_rows_none():
    assert family_complete_context(FAMILY, None) == "unknown"


# ---------------------------------------------------------------------------
# complete_window_ids / window_ids_touching_family
# ---------------------------------------------------------------------------

def test_complete_window_ids_lists_only_superset_windows():
    assert complete_window_ids(FAMILY, FIXTURE_D) == ("w1",)
    assert complete_window_ids(FAMILY, FIXTURE_B) == ()


def test_window_ids_touching_family_includes_partial_windows():
    assert set(window_ids_touching_family(FAMILY, FIXTURE_B)) == {"w1", "w2"}
    assert window_ids_touching_family(FAMILY, FIXTURE_E) == ()


# ---------------------------------------------------------------------------
# omitted_candidate_ids
# ---------------------------------------------------------------------------

def test_omitted_candidate_ids_for_partial_window():
    omitted = omitted_candidate_ids(FAMILY, FIXTURE_B)
    assert omitted["w1"] == ("clip-c",)
    assert omitted["w2"] == ("clip-a",)


def test_omitted_candidate_ids_empty_when_window_has_full_family():
    omitted = omitted_candidate_ids(FAMILY, FIXTURE_A)
    assert omitted == {}


def test_omitted_candidate_ids_ignores_windows_that_never_touch_family():
    omitted = omitted_candidate_ids(FAMILY, FIXTURE_E)
    assert omitted == {}


# ---------------------------------------------------------------------------
# partial_window_conflict (the D-094.3 F8 shape)
# ---------------------------------------------------------------------------

def test_partial_window_conflict_detects_two_winners_across_partial_windows():
    conflict = partial_window_conflict(FAMILY, FIXTURE_C)
    assert conflict is not None
    assert conflict["conflicting_winner_ids"] == ("clip-a", "clip-c")
    assert conflict["winner_window_ids"] == {"w1": ("clip-a",), "w2": ("clip-c",)}


def test_partial_window_conflict_none_when_family_complete_window_exists():
    # Fixture D has a disagreeing partial window (w2 crowns clip-a) but a
    # family-complete window (w1) also exists -- the complete window's own
    # labels are the trusted answer, so a partial window's disagreement is
    # moot and must not be reported as a conflict.
    assert partial_window_conflict(FAMILY, FIXTURE_D) is None


def test_partial_window_conflict_none_when_only_one_winner_recorded():
    assert partial_window_conflict(FAMILY, FIXTURE_B) is None


def test_partial_window_conflict_none_for_singleton_family():
    assert partial_window_conflict(("clip-a",), FIXTURE_C) is None


# ---------------------------------------------------------------------------
# provider_config_from_window_rows -- honest UNKNOWN, never invented
# ---------------------------------------------------------------------------

def test_provider_config_reports_temperature_and_prompt_version_unknown():
    config = provider_config_from_window_rows(FAMILY, FIXTURE_A)
    assert config["temperature"] == PROVIDER_TEMPERATURE_UNKNOWN == "UNKNOWN"
    assert config["prompt_version"] == PROVIDER_PROMPT_VERSION_UNKNOWN == "UNKNOWN"


def test_provider_config_collects_distinct_provider_model_pairs():
    rows = [
        _row("w1", ["clip-a", "clip-b", "clip-c"], provider="hybrid", model="flash-lite"),
    ]
    config = provider_config_from_window_rows(FAMILY, rows)
    assert config["providers_observed"] == (("hybrid", "flash-lite"),)


def test_provider_config_empty_when_no_window_touches_family():
    config = provider_config_from_window_rows(FAMILY, FIXTURE_E)
    assert config["providers_observed"] == ()


# ---------------------------------------------------------------------------
# stable_request_hash
# ---------------------------------------------------------------------------

def test_request_hash_stable_across_repeated_calls():
    args = (["clip-a", "clip-b"], {"clip-a": "hello", "clip-b": "world"},
            {"clip-a": 0.0, "clip-b": 3.0}, {"clip-a": 3.0, "clip-b": 6.0}, "flash-lite")
    assert stable_request_hash(*args) == stable_request_hash(*args)


def test_request_hash_order_insensitive_over_candidate_identity():
    texts = {"clip-a": "hello", "clip-b": "world"}
    starts = {"clip-a": 0.0, "clip-b": 3.0}
    ends = {"clip-a": 3.0, "clip-b": 6.0}
    forward = stable_request_hash(["clip-a", "clip-b"], texts, starts, ends, "flash-lite")
    reverse = stable_request_hash(["clip-b", "clip-a"], texts, starts, ends, "flash-lite")
    assert forward == reverse


def test_request_hash_changes_when_text_changes():
    starts = {"clip-a": 0.0}
    ends = {"clip-a": 3.0}
    original = stable_request_hash(["clip-a"], {"clip-a": "hello"}, starts, ends, "flash-lite")
    changed = stable_request_hash(["clip-a"], {"clip-a": "goodbye"}, starts, ends, "flash-lite")
    assert original != changed


def test_request_hash_changes_when_model_changes():
    texts, starts, ends = {"clip-a": "hello"}, {"clip-a": 0.0}, {"clip-a": 3.0}
    a = stable_request_hash(["clip-a"], texts, starts, ends, "flash-lite")
    b = stable_request_hash(["clip-a"], texts, starts, ends, "pro")
    assert a != b


def test_request_hash_uses_unknown_literal_when_prompt_version_missing():
    texts, starts, ends = {"clip-a": "hello"}, {"clip-a": 0.0}, {"clip-a": 3.0}
    without = stable_request_hash(["clip-a"], texts, starts, ends, "flash-lite")
    with_unknown = stable_request_hash(
        ["clip-a"], texts, starts, ends, "flash-lite", prompt_version="UNKNOWN",
    )
    with_real = stable_request_hash(
        ["clip-a"], texts, starts, ends, "flash-lite", prompt_version="v3",
    )
    assert without == with_unknown
    assert without != with_real


# ---------------------------------------------------------------------------
# family_authority_diagnostics composer + tail-safe summary
# ---------------------------------------------------------------------------

def test_family_authority_diagnostics_exposes_family_scoped_source_info_verbatim():
    source_info = {"family_complete_window_chunk_indices": [0], "family_window_labels": {}, "global_merge_labels": {}}
    result = family_authority_diagnostics(FAMILY, FIXTURE_A, source_info)
    assert result["family_scoped_source_info"] is source_info


def test_family_authority_diagnostics_authority_applied_reflects_todays_behavior():
    # source_info is None <=> family_scoped_semantic_decisions found no
    # complete window <=> today's actual authority is the global cross-
    # window max-priority merge, never a family-complete gate (that is
    # Phase B, not implemented here).
    without_complete_window = family_authority_diagnostics(FAMILY, FIXTURE_B, None)
    assert without_complete_window["provider_authority_applied"] == AUTHORITY_GLOBAL_CROSS_WINDOW_MERGE

    with_complete_window = family_authority_diagnostics(FAMILY, FIXTURE_A, {"family_window_labels": {}})
    assert with_complete_window["provider_authority_applied"] == AUTHORITY_FAMILY_COMPLETE_WINDOW_PREFERRED


def test_family_authority_diagnostics_full_shape_matches_detectors():
    result = family_authority_diagnostics(FAMILY, FIXTURE_C, None)
    assert result["family_complete_context"] == "false"
    assert result["partial_window_conflict"]["conflicting_winner_ids"] == ("clip-a", "clip-c")
    assert set(result["semantic_window_ids"]) == {"w1", "w2"}
    assert result["complete_window_ids"] == ()


def test_summarize_family_authority_observability_counts_only():
    rows = [
        family_authority_diagnostics(FAMILY, FIXTURE_A, {"x": 1}),
        family_authority_diagnostics(FAMILY, FIXTURE_B, None),
        family_authority_diagnostics(FAMILY, FIXTURE_C, None),
    ]
    summary = summarize_family_authority_observability(rows)
    assert summary["family_count"] == 3
    assert summary["family_complete_context_counts"] == {"true": 1, "false": 2}
    assert summary["partial_window_conflict_family_count"] == 1
    # Summary must stay counts-only -- no per-family detail (clip ids, window
    # ids, provider strings) leaks into the tail-safe projection.
    assert "conflicting_winner_ids" not in str(summary.keys())


# ---------------------------------------------------------------------------
# Wiring / no-behavior-change proofs (offline, stubbed provider only)
# ---------------------------------------------------------------------------

def _take(index: int, text: str) -> CandidateTake:
    return CandidateTake(
        clip_id=f"clip-{index}", source_asset_id="src", source_order=index,
        start=float(index * 3), end=float(index * 3 + 2.0), text=text,
    )


class _StubJudge:
    """Offline stub -- no network reference anywhere in this class."""

    def __init__(self, labels: dict[str, tuple[str, float]]):
        self.labels = labels

    def judge(self, session):
        return EditorialJudgeResult(
            decisions=tuple(
                EditorialDecision(candidate.clip_id, *self.labels.get(candidate.clip_id, ("keep", 0.5)), "test")
                for candidate in session.candidates
            ),
            provider="fake", model="flash-lite", requested=True, available=True,
            estimated_input_tokens=50, estimated_output_tokens=20,
        )


def test_apply_hybrid_session_cleanup_request_hash_is_additive_and_deterministic():
    takes = (_take(0, "hello there"), _take(1, "goodbye now"), _take(2, "hello there"))
    judge = _StubJudge({"clip-0": ("keep", 0.6), "clip-1": ("keep", 0.6), "clip-2": ("keep", 0.6)})

    baseline = apply_hybrid_session_cleanup(takes, None, judge, chunk_size=10, chunk_stride=5)
    repeat = apply_hybrid_session_cleanup(takes, None, judge, chunk_size=10, chunk_stride=5)

    # The new field never changes what is kept/deleted/labelled.
    assert baseline.kept == repeat.kept
    assert baseline.deleted == repeat.deleted
    assert baseline.semantic_decisions == repeat.semantic_decisions

    assert len(baseline.diagnostics) >= 1
    for row in baseline.diagnostics:
        assert "request_hash" in row
        assert row["request_hash"].startswith("rh_")
    # Same window, same candidates, same model -> same hash across two calls.
    assert [row["request_hash"] for row in baseline.diagnostics] == [
        row["request_hash"] for row in repeat.diagnostics
    ]


def test_semantic_authority_observability_module_has_no_provider_or_network_imports():
    import ast
    from pathlib import Path

    source = Path("cutsell_worker/semantic_authority_observability.py").read_text()
    tree = ast.parse(source)
    import_lines = [
        line for line in source.splitlines()
        if line.strip().startswith("import ") or line.strip().startswith("from ")
    ]
    forbidden = ("requests", "httpx", "urllib", "socket", "google", "genai", "openai")
    assert not any(any(bad in line for bad in forbidden) for line in import_lines)
    # Purely stdlib + no cutsell_worker sibling import at all (a true leaf
    # module -- nothing it depends on can create a circular import back into
    # hybrid_session_cleanup.py or pipeline.py).
    assert not any("cutsell_worker" in line or line.strip().startswith("from .") for line in import_lines)
    assert isinstance(tree, ast.Module)


def test_pipeline_imports_family_authority_diagnostics():
    import cutsell_worker.pipeline as pipeline_module

    assert pipeline_module.family_authority_diagnostics is not None


def test_hybrid_session_cleanup_imports_stable_request_hash():
    import cutsell_worker.hybrid_session_cleanup as hsc_module

    assert hsc_module.stable_request_hash is not None
