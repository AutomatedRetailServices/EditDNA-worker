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


def test_restart_filler_requires_production_direction_context():
    filler_examples = [
        "Wait, let me do that again.",
        "Hold on.",
        "Hold on, let me restart.",
        "No, restart.",
        "Restart that.",
        "Let me redo that.",
        "Do that again.",
        "Wait, no.",
    ]
    for text in filler_examples:
        assert pipeline.looks_like_filler(text), text


def test_restart_words_inside_ad_copy_remain_usable():
    usable_examples = [
        "These lashes hold on all day.",
        "Restart your routine.",
        "This helps you start again.",
        "The makeup holds on through sweat.",
        "But wait, there’s more.",
    ]
    for text in usable_examples:
        assert not pipeline.looks_like_filler(text), text
        clip = pipeline.make_base_clip("usable", 0, 1, text)
        pipeline.tag_clips_heuristic([clip])
        assert clip["meta"]["keep"] is True, text


def _composer_clip(cid, start, slot, score=0.95, keep=True, fallback_rule=None):
    clip = pipeline.make_base_clip(cid, start, start + 0.8, cid)
    clip["slot"] = slot
    clip["semantic_score"] = score
    clip["score"] = score
    clip["meta"].update({"keep": keep, "slot": slot, "semantic_score": score, "score": score})
    if fallback_rule:
        clip["meta"]["fallback_slot_rule"] = fallback_rule
    return clip


def test_blooper_preserves_keepable_unclassified_product_context():
    other = _composer_clip(
        "ordinary", 0, "OTHER", fallback_rule="unclassified_product_context"
    )
    composer = pipeline.build_composer([other], mode="blooper")
    assert composer["used_clip_ids"] == ["ordinary"]


def test_blooper_excludes_filler_meta_other():
    filler = _composer_clip("filler", 0, "OTHER", fallback_rule="production_meta_phrase")
    filler["meta"]["filler_rule"] = "production_meta_phrase"
    composer = pipeline.build_composer([filler], mode="blooper")
    assert composer["used_clip_ids"] == []


def test_blooper_recognized_slot_behavior_is_unchanged():
    clips = [
        _composer_clip("story", 0, "STORY"),
        _composer_clip("hook", 1, "HOOK"),
        _composer_clip("features", 2, "FEATURES"),
        _composer_clip("benefits", 3, "BENEFITS"),
        _composer_clip("cta", 4, "CTA"),
    ]
    composer = pipeline.build_composer(clips, mode="blooper")
    assert composer["used_clip_ids"] == ["story", "hook", "cta"]


def test_composer_excludes_lower_scoring_non_selected_cta_without_moving_selected_cta():
    early_cta = _composer_clip("early_cta", 0, "CTA", score=0.95)
    feature = _composer_clip("feature", 1, "FEATURES", score=0.95)
    late_cta = _composer_clip("late_cta", 2, "CTA", score=0.80)
    composer = pipeline.build_composer([early_cta, feature, late_cta])
    assert composer["cta_id"] == "early_cta"
    assert composer["used_clip_ids"] == ["early_cta", "feature"]


def test_selected_late_cta_keeps_source_position_and_non_selected_early_cta_is_excluded():
    weak_early_cta = _composer_clip("weak_early_cta", 0, "CTA", score=0.75)
    feature = _composer_clip("feature", 1, "FEATURES", score=0.95)
    strong_late_cta = _composer_clip("strong_late_cta", 2, "CTA", score=0.99)
    composer = pipeline.build_composer([weak_early_cta, feature, strong_late_cta])
    assert composer["cta_id"] == "strong_late_cta"
    assert composer["used_clip_ids"] == ["feature", "strong_late_cta"]


def test_selected_multi_clip_cta_block_remains_intact_in_source_order():
    feature = _composer_clip("feature", 0, "FEATURES", score=0.95)
    cta_one = _composer_clip("cta_one", 1.0, "CTA", score=0.94)
    cta_two = _composer_clip("cta_two", 1.5, "CTA", score=0.94)
    weaker_cta = _composer_clip("weaker_cta", 5, "CTA", score=0.80)
    composer = pipeline.build_composer([feature, cta_one, cta_two, weaker_cta])
    assert composer["cta_id"] == "cta_two"
    assert composer["used_clip_ids"] == ["feature", "cta_one", "cta_two"]
