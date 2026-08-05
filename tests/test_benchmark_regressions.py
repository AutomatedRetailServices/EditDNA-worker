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


def test_composer_preserves_source_order_for_early_cta():
    cta = _clip("cta", 0, 1, "Get yours today.")
    feature = _clip("feature", 2, 3, "It comes with three shades.")
    benefit = _clip("benefit", 4, 5, "These are so cute.")
    for clip in [cta, feature, benefit]:
        pipeline.tag_clips_heuristic([clip])
        clip["semantic_score"] = clip["meta"]["semantic_score"] = 0.95
    composer = pipeline.build_composer([cta, feature, benefit])
    assert composer["used_clip_ids"] == ["cta", "feature", "benefit"]
    assert composer["cta_id"] == "cta"


def test_clean_cut_restart_detection_does_not_call_slot_classifier(monkeypatch):
    monkeypatch.setattr(pipeline, "classify_slot", lambda _text: (_ for _ in ()).throw(AssertionError("slot classifier called")))
    cta = _clip("cta", 50, 55, "So I'm going to drop it down below so you can go check these out")
    hook = _clip("hook", 55.1, 56.0, "I found the perfect.")
    merged = pipeline.merge_incomplete_phrases([cta, hook])
    assert [clip["id"] for clip in merged] == ["cta", "hook"]
    assert merged[0]["meta"]["merge_diagnostic"] == "merge_prevented_semantic_restart"


def test_adjacent_duplicate_residual_clips_are_reduced_to_one():
    first = _clip("a", 0, 1, "Get yours today.")
    duplicate = _clip("b", 1.01, 1.2, "Get yours today.")
    diagnostics = []
    validated = pipeline.validate_clip_boundaries([first, duplicate], discarded_diagnostics=diagnostics)
    assert [clip["id"] for clip in validated] == ["a"]
    assert diagnostics == [{
        "clip_id": "b",
        "reason": "discarded_duplicate_residual_text",
        "start": 1.01,
        "end": 1.2,
        "source_start": 1.01,
        "source_end": 1.2,
        "text": "Get yours today.",
    }]


def test_repeated_cta_after_meaningful_gap_is_preserved():
    first = _clip("early", 0, 1, "Get yours today.")
    repeated = _clip("late", 20, 21, "Get yours today.")
    validated = pipeline.validate_clip_boundaries([first, repeated])
    assert [clip["id"] for clip in validated] == ["early", "late"]


def test_identical_text_from_distinct_source_takes_is_preserved():
    first = _clip("source0", 0, 1, "Get yours today.")
    second = _clip("source1", 1.01, 1.2, "Get yours today.")
    first.update(source_index=0, source_local="first.mp4", source_start=0, source_end=1)
    second.update(source_index=1, source_local="second.mp4", source_start=1.01, source_end=1.2)
    validated = pipeline.validate_clip_boundaries([first, second])
    assert [clip["id"] for clip in validated] == ["source0", "source1"]


def test_word_timestamp_repair_updates_source_bounds_when_present():
    fragment = {
        "id": "frag",
        "start": 59.62,
        "end": 59.64,
        "source_start": 59.62,
        "source_end": 59.64,
        "text": "I found the perfect.",
        "meta": {},
        "words": [
            {"start": 59.62, "end": 59.75, "word": " I"},
            {"start": 59.76, "end": 59.90, "word": " found"},
            {"start": 59.91, "end": 60.02, "word": " the"},
            {"start": 60.03, "end": 60.20, "word": " perfect."},
        ],
    }
    validated = pipeline.validate_clip_boundaries([fragment])
    assert len(validated) == 1
    assert validated[0]["start"] == validated[0]["source_start"] == 59.62
    assert validated[0]["end"] == validated[0]["source_end"] == 60.20


def test_buy_token_cta_uses_word_boundaries():
    assert pipeline.classify_slot("Buy this today.") == "CTA"
    assert pipeline.classify_slot("You can buy it below.") == "CTA"
    assert pipeline.classify_slot("Buy now.") == "CTA"


def test_buy_substrings_are_not_cta_commands():
    assert pipeline.classify_slot("Buyers love the results.") != "CTA"
    assert pipeline.classify_slot("Buying this was part of my story.") != "CTA"
    assert pipeline.classify_slot("The buyer reviewed the product.") != "CTA"
