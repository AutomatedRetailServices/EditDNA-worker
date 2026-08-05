from worker import pipeline


def test_but_wait_sales_language_is_not_filler():
    text = "But wait, there's more."
    assert pipeline.looks_like_filler(text) is False
    clip = pipeline.make_base_clip("wait", 0, 1, text)
    pipeline.tag_clips_heuristic([clip])
    assert clip["meta"]["keep"] is True
    assert "filler_rule" not in clip["meta"]


def test_wait_restart_language_remains_filler():
    text = "Wait, let me do that again."
    assert pipeline.looks_like_filler(text) is True
    clip = pipeline.make_base_clip("redo", 0, 1, text)
    pipeline.tag_clips_heuristic([clip])
    assert clip["meta"]["keep"] is False
    assert clip["meta"]["filler_rule"] == "restart_or_interruption_language"


def test_benchmark_fallback_slot_examples():
    examples = {
        "I found the perfect gift for our lip gloss girlies.": "HOOK",
        "You get six lip glosses, but wait, there's so much more to these.": "FEATURES",
        "stocking, the Santa hat, a Christmas tree, and let's not forget, a snowman.": "FEATURES",
        "These are so cute, they are all lip glosses.": "BENEFITS",
        "So if you know anyone who loves lip glosses or you yourself want them, grab some up. You can grab them up in a set of one, two, or three.": "CTA",
    }
    for text, expected in examples.items():
        assert pipeline.classify_slot(text) == expected


def _clip(cid, start, end, text, words=None):
    return pipeline.make_base_clip(cid, start, end, text, words=words)


def test_complete_cta_followed_by_restarted_hook_is_not_merged():
    cta = _clip("cta", 50, 55, "So I'm going to drop it down below so you can go check these out.")
    hook = _clip("hook", 55.1, 56.0, "I found the perfect.")
    merged = pipeline.merge_incomplete_phrases([cta, hook])
    assert len(merged) == 2
    assert merged[0]["text"] == cta["text"]
    assert merged[1]["text"] == hook["text"]
    assert merged[0]["meta"]["merge_diagnostic"] == "merge_prevented_semantic_restart"


def test_multiword_zero_duration_residual_fragment_is_repaired_or_not_emitted():
    fragment = _clip("frag", 59.62, 59.64, "I found the perfect.", words=[
        {"start": 59.62, "end": 59.75, "word": " I"},
        {"start": 59.76, "end": 59.90, "word": " found"},
        {"start": 59.91, "end": 60.02, "word": " the"},
        {"start": 60.03, "end": 60.20, "word": " perfect."},
    ])
    validated = pipeline.validate_clip_boundaries([fragment])
    assert len(validated) == 1
    assert validated[0]["end"] - validated[0]["start"] >= 0.08
    assert validated[0]["meta"]["boundary_diagnostic"] == "repaired_from_word_timestamps"


def test_duplicate_residual_text_is_not_emitted_across_adjacent_clips():
    first = _clip("a", 0, 1, "I found the perfect.")
    duplicate = _clip("b", 1.01, 1.5, "I found the perfect.")
    validated = pipeline.validate_clip_boundaries([first, duplicate])
    assert [clip["id"] for clip in validated] == ["a"]


def test_existing_valid_incomplete_phrase_merge_still_works():
    first = _clip("a", 0, 1, "this continues")
    second = _clip("b", 1, 2, "into a complete phrase.")
    merged = pipeline.merge_incomplete_phrases([first, second])
    assert len(merged) == 1
    assert merged[0]["text"] == "this continues into a complete phrase."
