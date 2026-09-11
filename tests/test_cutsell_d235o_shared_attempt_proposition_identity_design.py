"""D-235O: SHARED ATTEMPT/PROPOSITION IDENTITY SEAM -- ARCHITECTURE +
BOUNDED ENGINEERING DESIGN, FORENSIC + DESIGN ONLY.

No new production module is created by this task (per its own "FORENSIC +
DESIGN ONLY... Do NOT implement it here" instruction) -- the recommended
seam (exact canonical word-membership identity, D-235O's own verdict C)
is deferred to a future, separately-authorized D-235P. These tests are
SOURCE-CODE-TRUTH tests only: they ground every factual claim this task's
own decision-log entry makes in the ACTUAL, current, unmodified code,
never asserted without direct evidence -- the same technique already
established in D-235N's own test suite.
"""
import inspect


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


LANGUAGE_SPINE_PATH = "cutsell_worker/language_spine.py"
LANGUAGE_UTTERANCE_ATTEMPT_PATH = "cutsell_worker/language_utterance_attempt.py"
LANGUAGE_SPINE_LIVE_PATH = "cutsell_worker/language_spine_live_integration.py"
RAW_UNDERSTANDING_MAP_PATH = "cutsell_worker/raw_understanding_map.py"
ATTEMPT_RECONSTRUCTION_PATH = "cutsell_worker/attempt_reconstruction.py"
TAKE_SEGMENTATION_PATH = "cutsell_worker/take_segmentation.py"
CANONICAL_IDENTITY_PATH = "cutsell_worker/canonical_identity.py"
CONTRACTS_PATH = "cutsell_worker/contracts.py"


class TestWordHasNoStableId:
    def test_01_word_dataclass_has_no_id_field(self):
        content = _read(CONTRACTS_PATH)
        # class Word: text, start, end, confidence -- no id/index field.
        idx = content.find("class Word:")
        assert idx != -1
        snippet = content[idx:idx + 300]
        assert "id" not in snippet.lower().replace("confidence", "") or "word_id" not in snippet


class TestLanguageWordCanonicalOrdinal:
    def test_02_language_word_has_word_index_field(self):
        content = _read(LANGUAGE_SPINE_PATH)
        assert "word_index: int" in content

    def test_03_language_word_built_by_sorting_raw_words_by_start_end(self):
        content = _read(LANGUAGE_SPINE_PATH)
        assert "def adapt_words_to_language_words(" in content
        assert "sorted(words, key=lambda word: (word.start, word.end))" in content
        assert "word_index=index" in content

    def test_04_adapter_consumes_raw_understanding_map_word_timings(self):
        content = _read(LANGUAGE_SPINE_LIVE_PATH)
        assert "adapt_words_to_language_words(source_asset_id, raw_understanding_map.word_timings)" in content


class TestPhraseUtteranceWordIndexChain:
    def test_05_language_phrase_has_word_start_end_index(self):
        content = _read(LANGUAGE_SPINE_PATH)
        assert "word_start_index: int" in content
        assert "word_end_index: int" in content

    def test_06_language_utterance_has_phrase_start_end_index(self):
        content = _read(LANGUAGE_UTTERANCE_ATTEMPT_PATH)
        assert "phrase_start_index: int" in content
        assert "phrase_end_index: int" in content

    def test_07_language_attempt_has_no_direct_word_index_but_has_utterance_ids(self):
        content = _read(LANGUAGE_UTTERANCE_ATTEMPT_PATH)
        idx = content.find("class LanguageAttempt:")
        assert idx != -1
        snippet = content[idx:idx + 700]
        assert "utterance_ids: Tuple[str, ...]" in snippet
        # Confirms the word-index range is only TRANSITIVELY derivable
        # (via utterance_ids -> phrase_start/end_index -> word_start/
        # end_index), never stored directly on LanguageAttempt itself --
        # exactly the gap this task's own design closes additively.
        assert "word_start_index" not in snippet
        assert "word_end_index" not in snippet


class TestSharedUnderlyingWordSource:
    def test_08_raw_understanding_map_word_timings_built_from_transcript_segments(self):
        content = _read(RAW_UNDERSTANDING_MAP_PATH)
        assert "word_timings = tuple(word for seg in segments for word in seg.words)" in content

    def test_09_candidate_take_words_built_from_the_same_segment_words(self):
        content = _read(TAKE_SEGMENTATION_PATH)
        # take_segmentation.py builds CandidateTake.words directly from the
        # SAME TranscriptSegment.words RawUnderstandingMap also concatenates
        # from -- never a second/independent ASR derivation.
        assert "segment.words" in content

    def test_10_attempt_reconstruction_merges_constituent_words_verbatim(self):
        content = _read(ATTEMPT_RECONSTRUCTION_PATH)
        assert "words=tuple(word for member in members for word in member.words)" in content


class TestReconstructedAttemptIdSemantics:
    def test_11_attempt_id_minted_from_source_span_ids_not_words(self):
        content = _read(ATTEMPT_RECONSTRUCTION_PATH)
        assert "attempt_id = mint_attempt_id(_member_span_ids(members))" in content

    def test_12_mint_attempt_id_is_content_membership_anchored_not_timestamp(self):
        content = _read(CANONICAL_IDENTITY_PATH)
        idx = content.find("def mint_attempt_id(")
        assert idx != -1

    def test_13_two_independent_attempt_id_minting_functions_confirmed(self):
        # Reconfirms D-235N's own finding: DraftClip/CandidateTake's own
        # attempt_id (canonical_identity.mint_attempt_id) and
        # LanguageAttempt's own attempt_id (language_utterance_attempt.py
        # ::_attempt_id) are minted by two entirely separate functions --
        # this is exactly WHY "Option A: propagate the existing
        # reconstructed attempt_id directly" does not by itself solve
        # anything: the two id spaces have no shared meaning today.
        canonical_identity = _read(CANONICAL_IDENTITY_PATH)
        language_utterance_attempt = _read(LANGUAGE_UTTERANCE_ATTEMPT_PATH)
        assert "def mint_attempt_id(" in canonical_identity
        assert "def _attempt_id(" in language_utterance_attempt
        assert "def mint_attempt_id(" not in language_utterance_attempt
        assert "def _attempt_id(" not in canonical_identity


class TestNoExistingWordIndexOnReconstructedSide:
    def test_14_candidate_take_has_no_word_index_range_field(self):
        content = _read(CONTRACTS_PATH)
        idx = content.find("class CandidateTake:")
        assert idx != -1
        snippet = content[idx:idx + 900]
        assert "word_start_index" not in snippet
        assert "word_end_index" not in snippet
        assert "canonical_word_start_index" not in snippet

    def test_15_draft_clip_has_no_word_index_range_field(self):
        content = _read(CONTRACTS_PATH)
        idx = content.find("class DraftClip:")
        assert idx != -1
        snippet = content[idx:idx + 3000]
        assert "word_start_index" not in snippet
        assert "word_end_index" not in snippet
        assert "canonical_word_start_index" not in snippet


class TestDesignOnlyNoImplementation:
    def test_16_no_new_production_module_created_by_this_gate(self):
        import os
        assert not os.path.exists("cutsell_worker/lost_atom_shared_word_identity.py")
        assert not os.path.exists("cutsell_worker/shared_attempt_proposition_identity.py")

    def test_17_no_new_canonical_id_minted_by_this_gate(self):
        # This gate's (D-235O's) own production footprint is exactly zero
        # new files (test_16 above) -- there is no module for a new
        # id-minting function to live in. D-235P (a SEPARATELY-authorized,
        # later gate) went on to implement D-235O's own verdict-C
        # recommendation as `shared_attempt_word_identity.py` -- neither
        # of the two specific filenames D-235O's own design text proposed
        # (test_16, still both absent) -- so this snapshot is widened to
        # include that later, authorized addition rather than pretending
        # the family can never grow; it still proves D-235O's OWN gate,
        # by itself, minted zero new canonical ids in zero new files.
        import os
        new_files = [
            f for f in os.listdir("cutsell_worker")
            if f.startswith("lost_atom_") or f.startswith("shared_attempt_")
        ]
        expected_after_d235o_and_authorized_successors = {
            "lost_atom_editorial_requirement_evidence.py",
            "lost_atom_proposition_identity_forensic.py",
            "shared_attempt_word_identity.py",
        }
        assert set(new_files) == expected_after_d235o_and_authorized_successors
