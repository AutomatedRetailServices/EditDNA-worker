"""D-286 (RAW #120 audit) -- a bare negation particle must never be treated
as removable "contentless" debris.

Real RAW #120 evidence: clip_id `clip_3f1ef845b87b2fb31920`, text "No"
(0.8s, 269.37-270.17), immediately preceded a later clip "quiero sonar a
conspiración pero..." (276.09-283.67). Together they form the single
composite proposition "No quiero sonar a conspiración pero..." -- removing
the "No" silently FLIPS the meaning of the sentence from negated to
affirmed. D-283's `_safe_contentless_alternate_debris` (RAW #119 audit) was
written to remove exactly this token shape (a bare, sub-4-character
"contentless" utterance) as recording-process debris, using RAW #119's own
"No" orphan as its positive control -- but that generalization did not
account for negation carrying real, load-bearing meaning independent of
its (short) length. Per CLAUDE.md's own editorial doctrine ("WHEN
UNCERTAIN, KEEP"; "Never invent speech"), a structural/length-only debris
heuristic must never be trusted to silently delete a negation -- there is
no general, reliable way to distinguish "negation prefixing real
continuation" from "negation as pure recording debris" from structure
alone, so the safe general rule is: a negation particle is never removed
by this rule at all. Reuses the SAME `_critical()` negation/numeric
vocabulary `_safe_short_alternate_debris` already trusts elsewhere in this
file for polarity safety -- not a new or Video00-specific word list.
Generic fixtures only (no Video00 text).
"""
from cutsell_worker.contracts import CandidateTake, MediaSignals, Word
from cutsell_worker.hybrid_retry_completion_integrity import (
    _safe_contentless_alternate_debris,
    apply_hybrid_retry_completion_integrity,
)
from cutsell_worker.hybrid_session_cleanup import HybridSessionCleanupResult


def _words(text, start=0.0, step=0.25):
    output = []
    cursor = float(start)
    for token in text.split():
        output.append(Word(token, cursor, cursor + step, 0.95))
        cursor += step
    return tuple(output)


def _take(clip_id, start, end, text, *, complete=True):
    return CandidateTake(
        clip_id=clip_id, source_asset_id="src", source_order=0,
        start=float(start), end=float(end), text=text,
        words=_words(text, start), signals=MediaSignals("src", float(start), float(end)),
        complete_idea=complete,
    )


def _result(kept, deleted=(), decisions=()):
    return HybridSessionCleanupResult(
        kept=tuple(kept), deleted=tuple(deleted),
        requested_chunk_count=1, available_chunk_count=1,
        diagnostics=(), semantic_decisions=tuple(decisions),
    )


# ---------------------------------------------------------------------------
# Unit tests: the negation guard on _safe_contentless_alternate_debris
# ---------------------------------------------------------------------------

def test_bare_negation_particle_is_never_removed_even_with_every_other_condition_met():
    previous = _take("previous", 0.0, 4.0, "We reviewed the whole shipping timeline last week.")
    negation = _take("negation", 4.5, 5.3, "No", complete=True)
    following = _take("following", 10.0, 14.0, "want to sound like I am exaggerating but the delay was real.")
    semantic = {"negation": ("alternate", 0.98)}  # highest possible confidence

    assert _safe_contentless_alternate_debris(negation, previous, following, semantic) is False


def test_other_negation_vocabulary_is_also_protected():
    previous = _take("previous", 0.0, 4.0, "We reviewed the whole shipping timeline last week.")
    negation = _take("negation", 4.5, 5.3, "Never.", complete=True)
    following = _take("following", 10.0, 14.0, "would I recommend skipping the inspection step.")
    semantic = {"negation": ("alternate", 0.95)}

    assert _safe_contentless_alternate_debris(negation, previous, following, semantic) is False


def test_bare_numeric_particle_is_also_protected():
    # `_critical()` protects numerics alongside negation (same shared
    # vocabulary this file already reuses for polarity safety elsewhere) --
    # a bare "3." immediately before "months of testing were required." is
    # the same load-bearing-fragment shape as a bare negation.
    previous = _take("previous", 0.0, 4.0, "We reviewed the whole shipping timeline last week.")
    numeral = _take("numeral", 4.5, 5.3, "3.", complete=True)
    following = _take("following", 10.0, 14.0, "months of testing were required before launch.")
    semantic = {"numeral": ("alternate", 0.95)}

    assert _safe_contentless_alternate_debris(numeral, previous, following, semantic) is False


def test_non_negating_filler_orphan_is_still_removed_unaffected_by_the_guard():
    # The guard is scoped to negation/numeric tokens specifically -- a
    # genuinely contentless, non-negating filler is still handled exactly
    # as D-283 intended.
    previous = _take("previous", 0.0, 4.0, "We reviewed the whole shipping timeline last week.")
    filler = _take("filler", 4.5, 5.3, "Um.", complete=True)
    following = _take("following", 10.0, 14.0, "I want to explain what actually happened with the order.")
    semantic = {"filler": ("alternate", 0.90)}

    assert _safe_contentless_alternate_debris(filler, previous, following, semantic) is True


# ---------------------------------------------------------------------------
# Integration: apply_hybrid_retry_completion_integrity end to end
# ---------------------------------------------------------------------------

def test_negation_survives_the_integrity_pass_end_to_end():
    previous = _take("previous", 0.0, 4.0, "We reviewed the whole shipping timeline last week.")
    negation = _take("negation", 4.5, 5.3, "No")
    following = _take("following", 10.0, 14.0, "want to sound like I am exaggerating but the delay was real.")
    result = _result(
        (previous, negation, following),
        decisions=(("previous", "keep", 0.90), ("negation", "alternate", 0.97), ("following", "keep", 0.90)),
    )

    repaired = apply_hybrid_retry_completion_integrity(result, (previous, negation, following))

    assert "negation" in {t.clip_id for t in repaired.kept}
