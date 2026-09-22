"""D-156 (RAW #119 audit): incomplete-realization retry completion defeated
by verb-conjugation/inflection variance in the shared-content-word check.

Real RAW #119 evidence (run 35756418346, four-way quality-ladder attribution
IdeaClusterer/RetryFamilyFormation): an incomplete delivery ("Ahi fue cuando
me mando.") and its later complete realization ("Ahi fue cuando me mandaron
a hacer sonografia de tiroides y otras sonografias.") were never grouped
into one retry family (both clip_id-level `retry_family`/`idea_id` were
`null` in the engine's own diagnostics), so BOTH survived to KEEP instead of
the complete one winning -- the incomplete fragment must not be kept once
its later completion exists (D-019/D-020: complete delivery dominates an
incomplete/abandoned retry of the same idea).

Root cause: `incomplete_attempt_completed_by_retry` (and its siblings
`same_opening_restart` / `multimodal_corroborated_retry`) require real
shared content words BEYOND the opening -- correctly so, to keep an
unrelated aside sharing only a topic opener from merging (D-097.12's own
"Tuve problemas de estomago ... no hay que preguntar." negative control).
But the shared-word check was EXACT-TOKEN equality: "mando" (he/she sent,
singular) and "mandaron" (they sent, plural) are the same verb with a
different subject/number agreement, and share zero exact tokens -- the
floor could never fire on the single most literal shape of self-correction
(fixing agreement/number while continuing the same idea), independent of
where the floor was set. Fixed generally (`_content_words_match` /
`_shared_content_count` in take_grouping.py): two content words of real
length (>= 5 chars) whose shared prefix covers >= 80% of the shorter
word's length count as shared -- "mando"/"mandaron" clear it (4/5 = 80%);
a coincidental collision like "contest"/"context" does not (5/7 = 71%).
This is inflection-tolerance, not Video00 vocabulary -- the fixtures below
use unrelated Spanish/English text about unrelated topics and never
reference Video00 clip ids, phrases or timestamps.
"""
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.take_grouping import (
    _content_words_match,
    incomplete_attempt_completed_by_retry,
    multimodal_corroborated_retry,
    same_opening_restart,
)


def _take(clip_id, start, end, text, complete_idea, source="src"):
    return CandidateTake(clip_id, source, 0, start, end, text, complete_idea=complete_idea)


# ---------------------------------------------------------------------------
# The exact reported shape: an incomplete take sharing its 2-word opening
# with its completion, whose only "rest" content overlap beyond the shared
# discourse word is a subject/number conjugation variant of one verb.
# ---------------------------------------------------------------------------

def test_incomplete_retry_completed_with_reconjugated_verb_now_groups():
    incomplete = _take("a", 0.0, 2.0, "Ahí fue cuando la llamó.", complete_idea=False)
    complete = _take(
        "b", 2.6, 6.5,
        "Ahí fue cuando la llamaron para avisarle del resultado del examen.",
        complete_idea=True,
    )
    assert incomplete_attempt_completed_by_retry(incomplete, complete) == "incomplete_attempt_completed_by_retry"


def test_same_opening_restart_credits_a_reconjugated_shared_word():
    left = _take("a", 0.0, 3.0, "So then the office called about the missing package again.", complete_idea=True)
    right = _take("b", 3.5, 7.0, "So then the office calling everyone about the missing shipment yesterday.", complete_idea=True)
    assert same_opening_restart(left, right) == "same_opening_restart"


def test_multimodal_corroborated_retry_credits_a_reconjugated_shared_word():
    incomplete = _take("a", 0.0, 4.0, "The numbers arriving late again that week.", complete_idea=False)
    complete = _take("b", 5.0, 9.0, "The numbers finally arrived late again after the second request.", complete_idea=True)
    events = {"src": (("wrong_take", 3.5, 4.8),)}
    result = multimodal_corroborated_retry(incomplete, complete, events)
    assert result is not None and result[0] == "multimodal_corroborated_retry"


# ---------------------------------------------------------------------------
# Negative controls: stem-matching must not turn a coincidental short
# shared prefix into false evidence, and must not fire on tokens below the
# length floor.
# ---------------------------------------------------------------------------

def test_short_tokens_never_stem_match_even_with_shared_prefix():
    assert _content_words_match("casa", "caso") is False  # both below the 5-char floor


def test_coincidental_prefix_collision_does_not_stem_match():
    # "contest"/"context": share 5 of "contest"'s 7 characters (71%) purely
    # by coincidence -- below the 80% floor, and genuinely unrelated words.
    assert _content_words_match("contest", "context") is False


def test_genuine_word_family_variant_does_stem_match():
    # "reporter"/"reported": share 7 of 8 characters (87.5%) -- the same
    # word family, correctly credited.
    assert _content_words_match("reporter", "reported") is True


def test_unrelated_pair_with_only_a_coincidental_short_prefix_never_groups():
    incomplete = _take("a", 0.0, 2.0, "Ahí fue cuando la llamó.", complete_idea=False)
    unrelated = _take(
        "b", 2.6, 6.5,
        "Ahí fue cuando decidieron cancelar todo el evento del viernes.",
        complete_idea=True,
    )
    assert incomplete_attempt_completed_by_retry(incomplete, unrelated) is None


def test_exact_token_repeats_are_unaffected_by_the_stem_change():
    incomplete = _take("a", 0.0, 2.0, "This was when the results arrived.", complete_idea=False)
    complete = _take("b", 2.4, 6.0, "This was when the results arrived after the lab finally confirmed everything.", complete_idea=True)
    assert incomplete_attempt_completed_by_retry(incomplete, complete) == "incomplete_attempt_completed_by_retry"


def test_completely_unrelated_pair_still_never_groups():
    incomplete = _take("a", 0.0, 2.0, "That was when they called.", complete_idea=False)
    unrelated = _take("b", 2.6, 6.0, "That was actually a completely different topic about vacation planning.", complete_idea=True)
    assert incomplete_attempt_completed_by_retry(incomplete, unrelated) is None
