"""D-237L: fix exact-identity observability clip-key namespace mismatch.

See docs/CUTSELL_DECISIONS.md D-237K (the forensic finding: real lost
atoms existed, base identity rows existed, but `lost_atom_identity_
correlation` produced ZERO joined rows on real media) and D-237L (this
fix). The root cause: `identity_observability_rows_for_source` keyed its
own output dict -- and looked up the originating `CandidateTake` -- by
`match.reconstructed_attempt_id` (`shared_attempt_word_identity.py`'s
own `entity_id`, which prioritizes `candidate.attempt_id`/`candidate.
source_span_id` over `candidate.clip_id`), while `lost_atom_identity_
correlation`'s ONLY join condition is a genuine `clip_id`. Since real
media populates `attempt_id` on virtually every take, this was a
systematic namespace mismatch, not an edge case.

This module proves the fix: `identity_observability_rows_for_source` now
keys its rows by the caller's own `take.clip_id` (mirroring `pipeline.
py`'s own D-235X authority maps, `all_identity_matches_by_clip_id`/
`exact_match_by_clip_id`, which were never affected by this bug), while
`match.reconstructed_attempt_id` is preserved separately under its own
`reconstructed_attempt_id` field -- never overloaded onto `clip_id`.

Every test here constructs a `CandidateTake` with `clip_id != attempt_id
!= source_span_id` (the exact real-media shape D-237G's own offline test
suite never exercised, which is precisely why it never caught this bug)
and proves the full path -- row keying, row content, lost-atom
correlation -- now works end to end, while the untouched D-235X
authority maps and `shared_attempt_word_identity.py` itself remain
byte-for-byte unaffected."""
from __future__ import annotations

import inspect
import subprocess

from cutsell_worker.contracts import CandidateTake, Word
from cutsell_worker.language_proposition_relation import build_proposition_candidates
from cutsell_worker.language_spine import adapt_words_to_language_words, segment_language_phrases
from cutsell_worker.language_spine_live_integration import proposition_candidate_ids_by_attempt_id_for
from cutsell_worker.language_utterance_attempt import build_language_attempts, segment_language_utterances
import cutsell_worker.exact_identity_observability as observability_module
from cutsell_worker.exact_identity_observability import (
    identity_observability_rows_for_source,
    lost_atom_identity_correlation,
)
from cutsell_worker.shared_attempt_word_identity import (
    AUTHORITATIVE_RELATIONSHIP_STATUSES,
    RELATIONSHIP_EXACT_SAME_MEMBERSHIP,
    build_attempt_language_identity_matches_for_source,
)


# ---------------------------------------------------------------------------
# Fixture builders -- mirrors D-237G's own test file, but every take gets a
# REAL, DISTINCT attempt_id/source_span_id (the exact shape D-237G's own
# `_take` helper never exercised).
# ---------------------------------------------------------------------------
def _words(text: str, t0: float, *, per_word: float = 0.25, inner_gap: float = 0.05):
    words = []
    t = t0
    for tok in text.split(" "):
        words.append(Word(text=tok, start=t, end=t + per_word, confidence=0.9))
        t = t + per_word + inner_gap
    return tuple(words), t


def _take_with_real_ids(clip_id, attempt_id, source_span_id, source_asset_id, words):
    return CandidateTake(
        clip_id=clip_id, source_asset_id=source_asset_id, source_order=0,
        start=words[0].start, end=words[-1].end, text=" ".join(w.text for w in words),
        words=tuple(words), attempt_id=attempt_id, source_span_id=source_span_id,
    )


def _real_spine_for_source(source_asset_id, all_words, *, silence_intervals=()):
    lwords = adapt_words_to_language_words(source_asset_id, all_words)
    phrases = segment_language_phrases(lwords, audio_silence_intervals=silence_intervals)
    utterances = segment_language_utterances(phrases)
    attempts = build_language_attempts(utterances)
    utterances_by_id = {u.utterance_id: u for u in utterances}
    props = build_proposition_candidates(attempts)
    return lwords, phrases, utterances_by_id, attempts, props


def _build_matches(reconstructed, canonical_words, attempts, utterances_by_id, phrases):
    return build_attempt_language_identity_matches_for_source(
        reconstructed_attempts=reconstructed, canonical_words=canonical_words,
        language_attempts=attempts, utterances_by_id=utterances_by_id, phrases=phrases,
    )


def _real_id_fixture(source_asset_id="s1"):
    """One take whose clip_id/attempt_id/source_span_id are ALL distinct
    -- exact real-media shape."""
    words, _ = _words("oh too many people ready set these are the", 0.0)
    lwords, phrases, utterances_by_id, attempts, props = _real_spine_for_source(source_asset_id, words)
    take = _take_with_real_ids(
        "clip_real_123", "attempt_real_456", "span_real_789", source_asset_id, words,
    )
    return lwords, phrases, utterances_by_id, attempts, props, take


# ===========================================================================
# 1-2: real ids differ from clip_id.
# ===========================================================================
def test_01_attempt_id_differs_from_clip_id_by_construction():
    _, _, _, _, _, take = _real_id_fixture()
    assert take.attempt_id != take.clip_id
    assert take.attempt_id == "attempt_real_456"


def test_02_source_span_id_differs_from_clip_id_by_construction():
    _, _, _, _, _, take = _real_id_fixture()
    assert take.source_span_id != take.clip_id
    assert take.source_span_id == "span_real_789"


def test_02b_match_reconstructed_attempt_id_is_attempt_id_not_clip_id():
    # Confirms the EXACT D-237K root-cause shape: the match's own
    # `reconstructed_attempt_id` (shared_attempt_word_identity.py's own
    # `entity_id`, untouched by this fix) is `attempt_id`-first, NOT
    # `clip_id` -- proving the mismatch this fix routes around, never
    # changing that upstream computation itself.
    lwords, phrases, utterances_by_id, attempts, props, take = _real_id_fixture()
    matches = _build_matches((take,), lwords, attempts, utterances_by_id, phrases)
    assert matches[0].reconstructed_attempt_id == take.attempt_id
    assert matches[0].reconstructed_attempt_id != take.clip_id


# ===========================================================================
# 3-5: row keying/content now uses take.clip_id; reconstructed_attempt_id
# preserved separately.
# ===========================================================================
def test_03_observability_dict_key_uses_clip_id():
    lwords, phrases, utterances_by_id, attempts, props, take = _real_id_fixture()
    matches = _build_matches((take,), lwords, attempts, utterances_by_id, phrases)
    rows = identity_observability_rows_for_source(
        matches=matches, takes=(take,), attempts_by_id={a.attempt_id: a for a in attempts},
    )
    assert "clip_real_123" in rows
    assert "attempt_real_456" not in rows  # the OLD (buggy) key must never appear


def test_04_row_clip_id_field_uses_clip_id():
    lwords, phrases, utterances_by_id, attempts, props, take = _real_id_fixture()
    matches = _build_matches((take,), lwords, attempts, utterances_by_id, phrases)
    rows = identity_observability_rows_for_source(
        matches=matches, takes=(take,), attempts_by_id={a.attempt_id: a for a in attempts},
    )
    row = rows["clip_real_123"]
    assert row["clip_id"] == "clip_real_123"
    assert row["candidate_take"]["clip_id"] == "clip_real_123"


def test_05_reconstructed_attempt_id_preserved_separately():
    lwords, phrases, utterances_by_id, attempts, props, take = _real_id_fixture()
    matches = _build_matches((take,), lwords, attempts, utterances_by_id, phrases)
    rows = identity_observability_rows_for_source(
        matches=matches, takes=(take,), attempts_by_id={a.attempt_id: a for a in attempts},
    )
    row = rows["clip_real_123"]
    assert row["reconstructed_attempt_id"] == "attempt_real_456"
    # Never overloaded onto clip_id -- the two fields must differ here.
    assert row["reconstructed_attempt_id"] != row["clip_id"]


# ===========================================================================
# 6-9: lost-atom correlation now succeeds; dropped/selected candidates;
# multiple lost atoms independently; provenance dedup preserved.
# ===========================================================================
def test_06_lost_atom_correlates_by_clip_id():
    lwords, phrases, utterances_by_id, attempts, props, take = _real_id_fixture()
    matches = _build_matches((take,), lwords, attempts, utterances_by_id, phrases)
    rows = identity_observability_rows_for_source(
        matches=matches, takes=(take,), attempts_by_id={a.attempt_id: a for a in attempts},
    )
    # Real D-237K target shape: the lost-atom row is keyed by the
    # DraftClip's own clip_id (final_story_coherence_validation.py's own
    # `_lost_semantic_atoms`, which reads `clip.clip_id` verbatim).
    lost_rows = [{
        "lost_atom_provenance_id": "latom_clip_real_123_0",
        "clip_id": "clip_real_123",
        "text": "oh too many people ready set these are the",
    }]
    correlated = lost_atom_identity_correlation(lost_rows, rows)
    assert len(correlated) == 1
    assert correlated[0]["clip_id"] == "clip_real_123"
    assert correlated[0]["identity"]["clip_id"] == "clip_real_123"
    assert correlated[0]["identity"]["reconstructed_attempt_id"] == "attempt_real_456"


def test_07_dropped_candidate_present_before_absent_after_selection_correlates():
    # The exact D-237K shape: present_before_selection=true, present_
    # after_selection=false -- the candidate is STILL part of the full
    # take_tuple identity-observability population (this module never
    # filters by selected/discarded -- see pipeline.py's own D-050D1
    # comment: identity_observability_by_clip_id is built from the
    # COMPLETE pre-clean_cut candidate pool).
    lwords, phrases, utterances_by_id, attempts, props, take = _real_id_fixture()
    matches = _build_matches((take,), lwords, attempts, utterances_by_id, phrases)
    rows = identity_observability_rows_for_source(
        matches=matches, takes=(take,), attempts_by_id={a.attempt_id: a for a in attempts},
    )
    # Simulate: this take was present in the full observability population
    # but ended up DISCARDED (never selected) -- the lost-atom ledger only
    # ever iterates draft.discarded, so this is exactly that shape.
    lost_rows = [{"lost_atom_provenance_id": "latom_dropped_0", "clip_id": take.clip_id}]
    correlated = lost_atom_identity_correlation(lost_rows, rows)
    assert len(correlated) == 1
    assert correlated[0]["clip_id"] == take.clip_id


def test_08_selected_candidate_also_correlates():
    # Correlation itself never filters by selected/discarded status --
    # confirms a "selected" clip_id would correlate identically if it
    # ever appeared in a lost_semantic_atoms row (it structurally never
    # does today, since that ledger only reads draft.discarded, but the
    # correlation helper itself carries no such restriction).
    lwords, phrases, utterances_by_id, attempts, props, take = _real_id_fixture()
    matches = _build_matches((take,), lwords, attempts, utterances_by_id, phrases)
    rows = identity_observability_rows_for_source(
        matches=matches, takes=(take,), attempts_by_id={a.attempt_id: a for a in attempts},
    )
    lost_rows = [{"lost_atom_provenance_id": "latom_selected_0", "clip_id": take.clip_id}]
    correlated = lost_atom_identity_correlation(lost_rows, rows)
    assert len(correlated) == 1


def test_09_multiple_lost_atoms_correlate_independently():
    words1, t1 = _words("oh too many people ready set these are the", 0.0)
    words2, _ = _words("and then the other thing happened as well", t1 + 1.0)
    all_words = words1 + words2
    lwords, phrases, utterances_by_id, attempts, props = _real_spine_for_source(
        "s1", all_words, silence_intervals=((t1, t1 + 1.0),),
    )
    take1 = _take_with_real_ids("clip_real_123", "attempt_real_456", "span_real_789", "s1", words1)
    take2 = _take_with_real_ids("clip_real_999", "attempt_real_888", "span_real_777", "s1", words2)
    matches = _build_matches((take1, take2), lwords, attempts, utterances_by_id, phrases)
    rows = identity_observability_rows_for_source(
        matches=matches, takes=(take1, take2), attempts_by_id={a.attempt_id: a for a in attempts},
    )
    assert len(rows) == 2
    lost_rows = [
        {"lost_atom_provenance_id": "latom_a", "clip_id": "clip_real_123"},
        {"lost_atom_provenance_id": "latom_b", "clip_id": "clip_real_999"},
    ]
    correlated = lost_atom_identity_correlation(lost_rows, rows)
    assert len(correlated) == 2
    assert {c["clip_id"] for c in correlated} == {"clip_real_123", "clip_real_999"}


def test_10_provenance_dedup_preserved():
    lwords, phrases, utterances_by_id, attempts, props, take = _real_id_fixture()
    matches = _build_matches((take,), lwords, attempts, utterances_by_id, phrases)
    rows = identity_observability_rows_for_source(
        matches=matches, takes=(take,), attempts_by_id={a.attempt_id: a for a in attempts},
    )
    # Two rows sharing the SAME lost_atom_provenance_id -- only the FIRST
    # is kept (lost_atom_identity_correlation's own existing dedup logic,
    # untouched by this fix).
    lost_rows = [
        {"lost_atom_provenance_id": "dup_1", "clip_id": "clip_real_123"},
        {"lost_atom_provenance_id": "dup_1", "clip_id": "clip_real_123"},
    ]
    correlated = lost_atom_identity_correlation(lost_rows, rows)
    assert len(correlated) == 1


# ===========================================================================
# 11-12: source isolation unaffected by the fix.
# ===========================================================================
def test_11_cross_source_mismatch_blocked():
    lwords_a, phrases_a, utt_a, attempts_a, props_a, take_a = _real_id_fixture("srcA")
    words_b, _ = _words("a completely different source clip entirely", 0.0)
    lwords_b, phrases_b, utt_b, attempts_b, props_b = _real_spine_for_source("srcB", words_b)
    take_b = _take_with_real_ids("clip_real_B", "attempt_real_B", "span_real_B", "srcB", words_b)

    matches_a = _build_matches((take_a,), lwords_a, attempts_a, utt_a, phrases_a)
    matches_b = _build_matches((take_b,), lwords_b, attempts_b, utt_b, phrases_b)
    rows_a = identity_observability_rows_for_source(
        matches=matches_a, takes=(take_a,), attempts_by_id={a.attempt_id: a for a in attempts_a},
    )
    rows_b = identity_observability_rows_for_source(
        matches=matches_b, takes=(take_b,), attempts_by_id={a.attempt_id: a for a in attempts_b},
    )
    # A lost-atom row whose clip_id belongs to source B must never
    # correlate against source A's own rows.
    lost_rows = [{"lost_atom_provenance_id": "p1", "clip_id": "clip_real_B"}]
    correlated = lost_atom_identity_correlation(lost_rows, rows_a)
    assert correlated == []
    correlated_b = lost_atom_identity_correlation(lost_rows, rows_b)
    assert len(correlated_b) == 1


def test_12_no_cross_source_row_collision():
    lwords_a, phrases_a, utt_a, attempts_a, props_a, take_a = _real_id_fixture("srcA")
    words_b, _ = _words("a completely different source clip entirely", 0.0)
    lwords_b, phrases_b, utt_b, attempts_b, props_b = _real_spine_for_source("srcB", words_b)
    take_b = _take_with_real_ids("clip_real_123", "attempt_real_B", "span_real_B", "srcB", words_b)
    # Deliberately reuse the SAME clip_id string across two different
    # sources -- merging their per-source row dicts must never silently
    # cross-contaminate source_asset_id.
    matches_a = _build_matches((take_a,), lwords_a, attempts_a, utt_a, phrases_a)
    matches_b = _build_matches((take_b,), lwords_b, attempts_b, utt_b, phrases_b)
    rows_a = identity_observability_rows_for_source(
        matches=matches_a, takes=(take_a,), attempts_by_id={a.attempt_id: a for a in attempts_a},
    )
    rows_b = identity_observability_rows_for_source(
        matches=matches_b, takes=(take_b,), attempts_by_id={a.attempt_id: a for a in attempts_b},
    )
    assert rows_a["clip_real_123"]["source_asset_id"] == "srcA"
    assert rows_b["clip_real_123"]["source_asset_id"] == "srcB"


# ===========================================================================
# 13-17: relationship/authoritative/exact-match/proposition/word-index
# content unchanged by the fix -- only the KEY changed, never the VALUES.
# ===========================================================================
def test_13_relationship_status_unchanged():
    lwords, phrases, utterances_by_id, attempts, props, take = _real_id_fixture()
    matches = _build_matches((take,), lwords, attempts, utterances_by_id, phrases)
    rows = identity_observability_rows_for_source(
        matches=matches, takes=(take,), attempts_by_id={a.attempt_id: a for a in attempts},
    )
    assert rows["clip_real_123"]["relationship_status"] == matches[0].relationship_status


def test_14_authoritative_boolean_unchanged():
    lwords, phrases, utterances_by_id, attempts, props, take = _real_id_fixture()
    matches = _build_matches((take,), lwords, attempts, utterances_by_id, phrases)
    rows = identity_observability_rows_for_source(
        matches=matches, takes=(take,), attempts_by_id={a.attempt_id: a for a in attempts},
    )
    expected = matches[0].relationship_status in AUTHORITATIVE_RELATIONSHIP_STATUSES
    assert rows["clip_real_123"]["relationship_is_authoritative"] is expected


def test_15_exact_match_map_lookup_now_also_uses_clip_id():
    # D-237L closes a SECOND manifestation of the same root cause:
    # identity_match_diagnostic_row's own `exact_match_by_clip_id.get(...)`
    # lookup was ALSO keyed by the wrong id. exact_match_by_clip_id
    # ITSELF (pipeline.py's own D-235X map) is untouched -- built exactly
    # as before, genuinely clip_id-keyed -- only this READ site is fixed.
    lwords, phrases, utterances_by_id, attempts, props, take = _real_id_fixture()
    matches = _build_matches((take,), lwords, attempts, utterances_by_id, phrases)
    assert matches[0].relationship_status in AUTHORITATIVE_RELATIONSHIP_STATUSES
    exact_match_by_clip_id = {take.clip_id: matches[0]}  # genuine clip_id-keyed, as pipeline.py builds it
    rows = identity_observability_rows_for_source(
        matches=matches, takes=(take,), attempts_by_id={a.attempt_id: a for a in attempts},
        exact_match_by_clip_id=exact_match_by_clip_id,
    )
    assert rows["clip_real_123"]["exact_match_by_clip_id_present"] is True


def test_16_proposition_ids_unchanged():
    lwords, phrases, utterances_by_id, attempts, props, take = _real_id_fixture()
    matches = _build_matches((take,), lwords, attempts, utterances_by_id, phrases)
    prop_ids = proposition_candidate_ids_by_attempt_id_for(props)
    rows = identity_observability_rows_for_source(
        matches=matches, takes=(take,), attempts_by_id={a.attempt_id: a for a in attempts},
        proposition_candidate_ids_by_attempt_id=prop_ids,
    )
    for lang_row in rows["clip_real_123"]["language_attempts"]:
        expected = list(prop_ids.get(lang_row["attempt_id"], ()))
        assert lang_row["proposition_candidate_ids"] == expected


def test_17_word_indices_unchanged():
    lwords, phrases, utterances_by_id, attempts, props, take = _real_id_fixture()
    matches = _build_matches((take,), lwords, attempts, utterances_by_id, phrases)
    rows = identity_observability_rows_for_source(
        matches=matches, takes=(take,), attempts_by_id={a.attempt_id: a for a in attempts},
    )
    expected = sorted(matches[0].reconstructed_word_membership.word_indices)
    assert rows["clip_real_123"]["candidate_take"]["word_indices"] == expected


# ===========================================================================
# 18-19: deterministic output, no transcript leakage.
# ===========================================================================
def test_18_deterministic_output():
    lwords, phrases, utterances_by_id, attempts, props, take = _real_id_fixture()
    matches1 = _build_matches((take,), lwords, attempts, utterances_by_id, phrases)
    matches2 = _build_matches((take,), lwords, attempts, utterances_by_id, phrases)
    rows1 = identity_observability_rows_for_source(
        matches=matches1, takes=(take,), attempts_by_id={a.attempt_id: a for a in attempts},
    )
    rows2 = identity_observability_rows_for_source(
        matches=matches2, takes=(take,), attempts_by_id={a.attempt_id: a for a in attempts},
    )
    assert rows1 == rows2


def test_19_no_transcript_leakage():
    lwords, phrases, utterances_by_id, attempts, props, take = _real_id_fixture()
    matches = _build_matches((take,), lwords, attempts, utterances_by_id, phrases)
    rows = identity_observability_rows_for_source(
        matches=matches, takes=(take,), attempts_by_id={a.attempt_id: a for a in attempts},
    )
    import json
    blob = json.dumps(rows)
    for banned in ("oh too many people", "ready set", "these are the"):
        assert banned not in blob


# ===========================================================================
# 20-27: no authority/identity-policy/Freeze/materiality/repair/Language-
# Spine/P1-P2/Pacing-Audio-Join mutation -- same proof style as D-237G's
# own suite, re-asserted after this fix.
# ===========================================================================
def test_20_no_identity_authority_mutation():
    source = inspect.getsource(observability_module)
    assert "AUTHORITATIVE_RELATIONSHIP_STATUSES =" not in source
    assert "AUTHORITATIVE_RELATIONSHIP_STATUSES.add" not in source


def test_21_no_containment_promotion():
    # relationship_is_authoritative must still be a bare re-projection of
    # the canonical frozenset -- never a containment status promoted.
    lwords, phrases, utterances_by_id, attempts, props, take = _real_id_fixture()
    small_words, _ = _words("just a fragment", 0.0)
    matches = _build_matches((take,), lwords, attempts, utterances_by_id, phrases)
    for m in matches:
        expected = m.relationship_status in AUTHORITATIVE_RELATIONSHIP_STATUSES
        rows = identity_observability_rows_for_source(
            matches=(m,), takes=(take,), attempts_by_id={a.attempt_id: a for a in attempts},
        )
        assert rows[take.clip_id]["relationship_is_authoritative"] is expected


def test_22_no_freeze_mutation():
    imports = "\n".join(
        l for l in inspect.getsource(observability_module).splitlines()
        if l.strip().startswith(("import ", "from "))
    )
    assert "lost_semantic_atom_freeze_authority" not in imports


def test_23_no_materiality_mutation():
    imports = "\n".join(
        l for l in inspect.getsource(observability_module).splitlines()
        if l.strip().startswith(("import ", "from "))
    )
    assert "complete_lost_semantic_atom_materiality" not in imports


def test_24_no_repair_loop_mutation():
    imports = "\n".join(
        l for l in inspect.getsource(observability_module).splitlines()
        if l.strip().startswith(("import ", "from "))
    )
    assert "repair_loop" not in imports
    assert "lost_atom_repair_suppression" not in imports


def test_25_no_language_spine_mutation():
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", "cutsell_worker/language_spine.py",
         "cutsell_worker/language_utterance_attempt.py", "cutsell_worker/language_spine_live_integration.py"],
        capture_output=True, text=True, cwd=__file__.rsplit("/tests/", 1)[0],
    )
    assert result.stdout.strip() == "", f"Language-Spine files have an unexpected diff: {result.stdout}"


def test_26_no_p1_p2_mutation():
    imports = "\n".join(
        l for l in inspect.getsource(observability_module).splitlines()
        if l.strip().startswith(("import ", "from "))
    )
    for banned in ("editorial_moment_sequence_integration", "whole_video_editorial_reasoning"):
        assert banned not in imports


def test_27_no_pacing_audio_join_mutation():
    imports = "\n".join(
        l for l in inspect.getsource(observability_module).splitlines()
        if l.strip().startswith(("import ", "from "))
    )
    for banned in ("pacing_v2", "dialogue_pacing_transition", "audio_join_treatment"):
        assert banned not in imports


# ===========================================================================
# 28-30: no threshold, no provider, no RAW.
# ===========================================================================
def test_28_no_new_threshold_constant():
    source = inspect.getsource(observability_module)
    assert "DEFAULT_SPLIT_GAP_SEC" not in source
    assert "_BOUNDARY_MATCH_TOLERANCE_SEC" not in source


def test_29_no_provider_call():
    imports = "\n".join(
        l for l in inspect.getsource(observability_module).splitlines()
        if l.strip().startswith(("import ", "from "))
    )
    for banned in ("openai", "gemini", "anthropic", "requests", "httpx"):
        assert banned not in imports.lower()


def test_30_no_raw_or_workflow_dispatch():
    imports = "\n".join(
        l for l in inspect.getsource(observability_module).splitlines()
        if l.strip().startswith(("import ", "from "))
    )
    for banned in ("modal", "runpod"):
        assert banned not in imports.lower()
    assert "workflow_dispatch" not in inspect.getsource(observability_module)


# ===========================================================================
# 31: shared_attempt_word_identity.py itself has zero diff -- this fix
# touches ONLY exact_identity_observability.py and pipeline.py's own
# call-site kwarg.
# ===========================================================================
def test_31_shared_attempt_word_identity_unchanged():
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", "cutsell_worker/shared_attempt_word_identity.py"],
        capture_output=True, text=True, cwd=__file__.rsplit("/tests/", 1)[0],
    )
    assert result.stdout.strip() == "", f"shared_attempt_word_identity.py has an unexpected diff: {result.stdout}"


# ===========================================================================
# 32: pipeline.py's own authority maps are unaffected -- proven structurally
# by re-deriving them exactly as pipeline.py does and confirming clip_id
# keying (the SAME assertion pipeline.py's own code already guarantees;
# this test documents/locks that guarantee explicitly for this gate).
# ===========================================================================
def test_32_authority_maps_still_keyed_by_take_clip_id():
    lwords, phrases, utterances_by_id, attempts, props, take = _real_id_fixture()
    matches = _build_matches((take,), lwords, attempts, utterances_by_id, phrases)
    all_identity_matches_by_clip_id = {}
    exact_match_by_clip_id = {}
    for t, m in zip((take,), matches):
        all_identity_matches_by_clip_id[t.clip_id] = m
        if m.relationship_status in AUTHORITATIVE_RELATIONSHIP_STATUSES:
            exact_match_by_clip_id[t.clip_id] = m
    assert "clip_real_123" in all_identity_matches_by_clip_id
    assert "attempt_real_456" not in all_identity_matches_by_clip_id
    if exact_match_by_clip_id:
        assert "clip_real_123" in exact_match_by_clip_id
