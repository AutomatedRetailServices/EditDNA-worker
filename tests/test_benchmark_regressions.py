import pytest

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


def test_shop_token_cta_uses_word_boundaries():
    assert pipeline.classify_slot("Shop now.") == "CTA"
    assert pipeline.classify_slot("You can shop the collection below.") == "CTA"
    assert pipeline.classify_slot("Shop these shades.") == "CTA"
    assert pipeline.classify_slot("Tap the link to shop.") == "CTA"


def test_shop_substrings_are_not_cta_commands():
    assert pipeline.classify_slot("The workshop covers skincare basics.") != "CTA"
    assert pipeline.classify_slot("I edited this in Photoshop.") != "CTA"
    assert pipeline.classify_slot("The workshops begin tomorrow.") != "CTA"
    assert pipeline.classify_slot("This was photoshopped.") != "CTA"


def test_single_token_cta_verbs_require_action_context():
    cta_examples = [
        "Buy this today.",
        "Buy now.",
        "You can buy it below.",
        "Shop now.",
        "Shop these shades.",
        "You can shop the collection below.",
        "Tap the link to shop.",
        "Click below to buy.",
    ]
    for text in cta_examples:
        assert pipeline.classify_slot(text) == "CTA", text


def test_buy_and_shop_mentions_without_viewer_action_are_not_cta():
    non_cta_examples = [
        "I went to the shop with my mom.",
        "The shop closes at five.",
        "I decided to buy it yesterday.",
        "This is what I buy every summer.",
        "She went shopping after work.",
    ]
    for text in non_cta_examples:
        assert pipeline.classify_slot(text) != "CTA", text


def test_commas_alone_do_not_force_features_over_story():
    story_examples = [
        "Honestly, for me, this lasted all day.",
        "At first, for me, it felt a little different.",
        "When I opened it, honestly, I noticed the texture first.",
    ]
    for text in story_examples:
        assert pipeline.classify_slot(text) == "STORY", text


def test_product_enumeration_still_classifies_as_features():
    assert pipeline.classify_slot("a stocking, a Santa hat, a Christmas tree, and a snowman.") == "FEATURES"


def test_isolated_dependent_tail_fragments_remain_excluded():
    for text in ["So.", "But.", "And."]:
        clip = pipeline.make_base_clip("tail", 0, 1, text)
        pipeline.tag_clips_heuristic([clip])
        assert clip["meta"]["keep"] is False, text


def test_complete_sentences_starting_with_dependent_words_remain_eligible():
    for text in [
        "So this is the shade I use every day.",
        "But this one feels much softer.",
        "And it comes with three colors.",
    ]:
        clip = pipeline.make_base_clip("complete", 0, 1, text)
        pipeline.tag_clips_heuristic([clip])
        assert clip["meta"]["keep"] is True, text


def test_take_two_slate_language_is_production_meta():
    slate_examples = [
        "Take two.",
        "Okay, take two.",
        "This is take two.",
        "Take two, let's go.",
        "Take number two.",
    ]
    for text in slate_examples:
        assert pipeline.classify_slot_rule(text) == ("OTHER", "production_meta_phrase"), text
        clip = pipeline.make_base_clip("slate", 0, 1, text)
        pipeline.tag_clips_heuristic([clip])
        assert clip["meta"]["keep"] is False, text
        assert clip["meta"]["filler_rule"] == "production_meta_phrase", text


def test_take_two_dosage_and_usage_language_is_keepable_product_content():
    usage_examples = [
        "I take two gummies every morning.",
        "Take two capsules daily.",
        "You can take two tablets with food.",
        "She takes two scoops after training.",
        "I usually take two before bed.",
    ]
    for text in usage_examples:
        slot, rule = pipeline.classify_slot_rule(text)
        assert (slot, rule) != ("OTHER", "production_meta_phrase"), text
        clip = pipeline.make_base_clip("usage", 0, 1, text)
        pipeline.tag_clips_heuristic([clip])
        assert clip["meta"]["keep"] is True, text
        assert clip["meta"].get("filler_rule") != "production_meta_phrase", text


def test_take_two_change_preserves_recognized_feature_benefit_story_behavior():
    assert pipeline.classify_slot("It comes with three shades.") == "FEATURES"
    assert pipeline.classify_slot("It helps you feel confident.") == "BENEFITS"
    assert pipeline.classify_slot("Honestly, for me, this lasted all day.") == "STORY"


def test_start_over_commands_are_production_meta():
    command_examples = [
        "Start over.",
        "Let's start over.",
        "Okay, start over.",
        "No, start over.",
        "Start over from the beginning.",
        "Can we start over?",
        "I need to start over.",
    ]
    for text in command_examples:
        assert pipeline.classify_slot_rule(text) == ("OTHER", "production_meta_phrase"), text
        clip = pipeline.make_base_clip("restart", 0, 1, text)
        pipeline.tag_clips_heuristic([clip])
        assert clip["meta"]["keep"] is False, text
        assert clip["meta"]["filler_rule"] in {"production_meta_phrase", "restart_or_interruption_language"}, text


def test_start_over_inside_valid_narration_is_keepable():
    narration_examples = [
        "This routine helps you start over with clearer skin.",
        "The program lets you start over whenever you need.",
        "I had to start over after changing my routine.",
        "You can start over with a clean base.",
        "Starting over was the best decision for my skin.",
    ]
    for text in narration_examples:
        slot, rule = pipeline.classify_slot_rule(text)
        assert (slot, rule) != ("OTHER", "production_meta_phrase"), text
        clip = pipeline.make_base_clip("narration", 0, 1, text)
        pipeline.tag_clips_heuristic([clip])
        assert clip["meta"]["keep"] is True, text
        assert clip["meta"].get("filler_rule") != "production_meta_phrase", text


def test_existing_wait_redo_restart_cue_still_behaves_as_filler():
    clip = pipeline.make_base_clip("redo", 0, 1, "Wait, let me do that again.")
    pipeline.tag_clips_heuristic([clip])
    assert clip["meta"]["keep"] is False
    assert clip["meta"]["filler_rule"] == "restart_or_interruption_language"


def test_camera_rolling_slates_are_discarded_from_clean_cut():
    slate_examples = [
        "Camera rolling.",
        "Okay, camera rolling.",
        "Camera is rolling.",
        "The camera is rolling.",
        "Rolling.",
        "And rolling.",
        "We're rolling.",
    ]
    clips = []
    for idx, text in enumerate(slate_examples):
        assert pipeline.classify_slot_rule(text) == ("OTHER", "production_meta_phrase"), text
        clip = pipeline.make_base_clip(f"slate{idx}", idx, idx + 0.8, text)
        pipeline.tag_clips_heuristic([clip])
        assert clip["meta"]["keep"] is False, text
        assert clip["meta"]["filler_rule"] == "production_meta_phrase", text
        clips.append(clip)
    assert pipeline.select_clean_cut_clip_ids(clips) == []


def test_rolling_product_narration_remains_keepable():
    narration_examples = [
        "This applicator keeps the product rolling on smoothly.",
        "I kept rolling the serum into my routine.",
        "The cart keeps rolling easily.",
        "We are rolling out three new shades.",
    ]
    for text in narration_examples:
        slot, rule = pipeline.classify_slot_rule(text)
        assert (slot, rule) != ("OTHER", "production_meta_phrase"), text
        clip = pipeline.make_base_clip("rolling", 0, 1, text)
        pipeline.tag_clips_heuristic([clip])
        assert clip["meta"]["keep"] is True, text
        assert clip["meta"].get("filler_rule") != "production_meta_phrase", text


def test_camera_rolling_change_preserves_other_restart_slate_detection():
    for text in ["Take two.", "Start over.", "Wait, let me do that again."]:
        clip = pipeline.make_base_clip("restart", 0, 1, text)
        pipeline.tag_clips_heuristic([clip])
        assert clip["meta"]["keep"] is False, text


def test_holiday_item_words_need_product_enumeration_or_context_for_features():
    feature_examples = [
        "It includes a stocking, a Santa hat, a Christmas tree, and a snowman.",
        "You get four designs: a stocking, Santa hat, tree, and snowman.",
        "The set comes with Christmas tree and snowman variants.",
        "Included are a stocking, a Santa hat, and two ornament designs.",
        "stocking, the Santa hat, a Christmas tree, and let's not forget, a snowman.",
    ]
    for text in feature_examples:
        assert pipeline.classify_slot(text) == "FEATURES", text


def test_single_holiday_or_item_word_in_narration_is_not_features():
    non_feature_examples = [
        "I'm stocking up because this lasts all day.",
        "We placed it beside the Christmas tree.",
        "She wore a Santa hat in the video.",
        "The snowman was in the background.",
        "Stocking the shelves took all morning.",
    ]
    for text in non_feature_examples:
        assert pipeline.classify_slot(text) != "FEATURES", text
        clip = pipeline.make_base_clip("holiday", 0, 1, text)
        pipeline.tag_clips_heuristic([clip])
        assert clip["meta"]["keep"] is True, text


def test_holiday_feature_change_preserves_other_slot_behavior():
    assert pipeline.classify_slot("I found the perfect gift for our lip gloss girlies.") == "HOOK"
    assert pipeline.classify_slot("Honestly, for me, this lasted all day.") == "STORY"
    assert pipeline.classify_slot("These are so cute, they are all lip glosses.") == "BENEFITS"
    assert pipeline.classify_slot("Buy this today.") == "CTA"
    assert pipeline.classify_slot("Ordinary narration without a cue.") == "OTHER"


def test_grab_language_requires_viewer_directed_purchase_context():
    cta_examples = [
        "Grab some below.",
        "You can grab some using the link.",
        "Grab them while they're available.",
        "Go grab yours.",
        "Tap the link and grab one.",
        "Grab this set today.",
        "You can grab them in a set of three.",
        "So if you know anyone who loves lip glosses or you yourself want them, grab some up. You can grab them up in a set of one, two, or three.",
    ]
    for text in cta_examples:
        assert pipeline.classify_slot(text) == "CTA", text


def test_grab_narration_without_viewer_action_is_not_cta():
    non_cta_examples = [
        "I grab some before the gym.",
        "I need to grab someone to help.",
        "She told me to grab them from the table.",
        "I had to grab some supplies.",
        "We grabbed them before leaving.",
    ]
    for text in non_cta_examples:
        assert pipeline.classify_slot(text) != "CTA", text
        clip = pipeline.make_base_clip("grab", 0, 1, text)
        pipeline.tag_clips_heuristic([clip])
        assert clip["meta"]["keep"] is True, text


def test_grab_cta_change_preserves_other_slot_behavior():
    assert pipeline.classify_slot("Get yours today.") == "CTA"
    assert pipeline.classify_slot("Shop now.") == "CTA"
    assert pipeline.classify_slot("Buy now.") == "CTA"
    assert pipeline.classify_slot("Tap the link to shop.") == "CTA"
    assert pipeline.classify_slot("Click below to buy.") == "CTA"
    assert pipeline.classify_slot("Order now.") == "CTA"
    assert pipeline.classify_slot("Honestly, for me, this lasted all day.") == "STORY"
    assert pipeline.classify_slot("I get so many compliments.") == "PROOF"
    assert pipeline.classify_slot("These are so cute, they are all lip glosses.") == "BENEFITS"
    assert pipeline.classify_slot("It includes a stocking, a Santa hat, a Christmas tree, and a snowman.") == "FEATURES"
    assert pipeline.classify_slot("Ordinary narration without a cue.") == "OTHER"


def test_smart_and_ascii_apostrophes_normalize_equivalently():
    pairs = [
        ("Let’s start over.", "Let's start over."),
        ("We’re rolling.", "We're rolling."),
        ("I’m starting again.", "I'm starting again."),
        ("That’s take two.", "That's take two."),
    ]
    for smart, ascii_text in pairs:
        assert pipeline._normalized_words(smart) == pipeline._normalized_words(ascii_text)


def test_smart_apostrophe_production_commands_are_removed():
    production_examples = [
        "Let’s start over.",
        "We’re rolling.",
        "I’m starting again.",
        "That’s take two.",
        "We’re on take three.",
        "Let's start over.",
        "We're rolling.",
        "I'm starting again.",
        "That's take two.",
    ]
    for text in production_examples:
        slot, rule = pipeline.classify_slot_rule(text)
        assert (slot, rule) == ("OTHER", "production_meta_phrase"), text
        clip = pipeline.make_base_clip("prod", 0, 1, text)
        pipeline.tag_clips_heuristic([clip])
        assert clip["meta"]["keep"] is False, text
        assert clip["meta"]["filler_rule"] == "production_meta_phrase", text


def test_commercial_narration_with_contractions_remains_keepable():
    narration_examples = [
        "It’s one of my favorite products.",
        "You’re going to love this texture.",
        "I’m using this every morning.",
        "That’s why I recommend it.",
    ]
    for text in narration_examples:
        slot, rule = pipeline.classify_slot_rule(text)
        assert (slot, rule) != ("OTHER", "production_meta_phrase"), text
        clip = pipeline.make_base_clip("contract", 0, 1, text)
        pipeline.tag_clips_heuristic([clip])
        assert clip["meta"]["keep"] is True, text


HEURISTIC_MATRIX_CASES = [
    ("unicode", "Let’s start over.", "OTHER", False),
    ("unicode", "Let's start over.", "OTHER", False),
    ("unicode", "We’re rolling.", "OTHER", False),
    ("unicode", "We're rolling.", "OTHER", False),
    ("unicode", "It’s one of my favorite products.", "OTHER", True),
    ("unicode", "You’re going to love this texture.", "OTHER", True),
    ("production", "Wait, let me do that again.", "OTHER", False),
    ("production", "Hold on.", "OTHER", False),
    ("production", "No, restart.", "OTHER", False),
    ("production", "Start over from the beginning.", "OTHER", False),
    ("production", "Take two.", "OTHER", False),
    ("production", "Camera is rolling.", "OTHER", False),
    ("production_negative", "These lashes hold on all day.", "OTHER", True),
    ("production_negative", "This routine helps you start over with clearer skin.", "BENEFITS", True),
    ("production_negative", "I take two gummies every morning.", "OTHER", True),
    ("production_negative", "We are rolling out three new shades.", "FEATURES", True),
    ("cta", "Buy this today.", "CTA", True),
    ("cta", "You can buy it below.", "CTA", True),
    ("cta", "Shop these shades.", "CTA", True),
    ("cta", "Tap the link to shop.", "CTA", True),
    ("cta", "Grab them while they're available.", "CTA", True),
    ("cta", "Order now.", "CTA", True),
    ("cta_negative", "I decided to buy it yesterday.", "OTHER", True),
    ("cta_negative", "The shop closes at five.", "OTHER", True),
    ("cta_negative", "I grab some before the gym.", "OTHER", True),
    ("cta_negative", "We grabbed them before leaving.", "OTHER", True),
    ("cta_negative", "The workshop covers skincare basics.", "OTHER", True),
    ("features", "It includes a stocking, a Santa hat, a Christmas tree, and a snowman.", "FEATURES", True),
    ("features", "You get four designs: a stocking, Santa hat, tree, and snowman.", "FEATURES", True),
    ("features", "The set comes with Christmas tree and snowman variants.", "FEATURES", True),
    ("features", "It comes with three shades.", "FEATURES", True),
    ("features_negative", "Honestly, for me, this lasted all day.", "STORY", True),
    ("features_negative", "We placed it beside the Christmas tree.", "OTHER", True),
    ("features_negative", "The snowman was in the background.", "OTHER", True),
    ("story", "At first, for me, it felt a little different.", "STORY", True),
    ("story", "When I opened it, honestly, I noticed the texture first.", "STORY", True),
    ("proof", "I get so many compliments.", "PROOF", True),
    ("proof", "Before and after was measurable.", "PROOF", True),
    ("benefits", "It helps you feel confident.", "BENEFITS", True),
    ("benefits", "These are so cute, they are all lip glosses.", "BENEFITS", True),
    ("hook", "I found the perfect gift for our lip gloss girlies.", "HOOK", True),
    ("hook", "Wait until you see the next feature.", "HOOK", True),
    ("other", "Ordinary narration without a cue.", "OTHER", True),
    ("other", "I opened the box and noticed the texture.", "OTHER", True),
    ("tail", "So.", "OTHER", False),
    ("tail", "But.", "OTHER", False),
    ("tail", "And.", "OTHER", False),
    ("tail_negative", "So this is the shade I use every day.", "OTHER", True),
    ("tail_negative", "But this one feels much softer.", "OTHER", True),
    ("tail_negative", "And it comes with three colors.", "FEATURES", True),
]


@pytest.mark.parametrize("category,text,expected_slot,expected_keep", HEURISTIC_MATRIX_CASES)
def test_consolidated_heuristic_matrix(category, text, expected_slot, expected_keep):
    assert pipeline.classify_slot(text) == expected_slot, category
    clip = pipeline.make_base_clip(category, 0, 1, text)
    pipeline.tag_clips_heuristic([clip])
    assert clip["slot"] == expected_slot, category
    assert clip["meta"]["keep"] is expected_keep, category
