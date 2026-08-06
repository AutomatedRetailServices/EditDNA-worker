import copy
import io
import json

import pytest
from botocore.exceptions import ClientError

import benchmark
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
        "original_clip_id": "b",
        "namespaced_clip_id": "source_000:b",
        "diagnostic_id": "source_000:b:discarded_duplicate_residual_text",
        "source_index": 0,
        "source_local": "source_000",
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

FILLER_META_INVENTORY_POSITIVE_CASES = [
    ("thanks", "standalone_meta_token"),
    ("Thank you.", "standalone_meta_token"),
    ("Okay, thanks.", "standalone_meta_token"),
    ("All right, thank you.", "standalone_meta_token"),
    ("Thanks, that's it.", "end_of_take_filler"),
    ("Thank you, we're done.", "end_of_take_filler"),
    ("Okay.", "standalone_meta_token"),
    ("Alright.", "standalone_meta_token"),
    ("Wait.", "standalone_meta_token"),
    ("Wait, no.", "standalone_production_direction"),
    ("Hold on.", "standalone_production_direction"),
    ("Hold on, let me restart.", "restart_or_interruption_language"),
    ("Let me redo that.", "restart_or_interruption_language"),
    ("Do that again.", "standalone_production_direction"),
    ("Restart that.", "restart_or_interruption_language"),
    ("Start over.", "production_meta_phrase"),
    ("Take two.", "production_meta_phrase"),
    ("Take three.", "production_meta_phrase"),
    ("Take number two.", "production_meta_phrase"),
    ("Camera rolling.", "production_meta_phrase"),
    ("We're rolling.", "production_meta_phrase"),
    ("Is that good?", "standalone_meta_token"),
    ("Am I saying it right?", "standalone_meta_token"),
    ("Cut that.", "standalone_production_direction"),
]


@pytest.mark.parametrize("text,expected_rule", FILLER_META_INVENTORY_POSITIVE_CASES)
def test_filler_meta_inventory_positive_cases_are_discarded(text, expected_rule):
    assert pipeline.filler_rule(text) == expected_rule
    clip = pipeline.make_base_clip("filler", 0, 1, text)
    pipeline.tag_clips_heuristic([clip])
    assert clip["meta"]["keep"] is False
    assert pipeline.select_clean_cut_clip_ids([clip]) == []


FILLER_META_INVENTORY_NEGATIVE_CASES = [
    "Thanks to this serum, I feel confident.",
    "Thanks to the formula, my skin stays hydrated.",
    "Thank you for making this so easy to use.",
    "I recommend it because the results speak for themselves, thank you all for watching.",
    "This works thanks to the lightweight texture.",
    "Okay, this shade is really pretty.",
    "Alright, this is the one I use every day.",
    "But wait, there's more.",
    "Wait until you see the next feature.",
    "These lashes hold on all day.",
    "The formula holds on through the workout.",
    "This helps you start again after a setback.",
    "Restart your routine.",
    "I take two gummies every morning.",
    "Take two capsules daily.",
    "That's why I recommend two shades.",
    "This applicator keeps the product rolling on smoothly.",
    "We are rolling out three new shades.",
    "I cut that crease with this brush.",
    "That one good ingredient makes it work.",
    "I remember why this formula worked for me.",
]


@pytest.mark.parametrize("text", FILLER_META_INVENTORY_NEGATIVE_CASES)
def test_filler_meta_inventory_negative_cases_remain_keepable(text):
    assert pipeline.filler_rule(text) is None, text
    clip = pipeline.make_base_clip("valid", 0, 1, text)
    pipeline.tag_clips_heuristic([clip])
    assert clip["meta"]["keep"] is True, text
    assert pipeline.select_clean_cut_clip_ids([clip]) == ["valid"], text


def test_no_generic_filler_collection_uses_phrase_matching():
    assert not hasattr(pipeline, "FILLER_PATTERNS")
    assert pipeline.SAFE_MULTIWORD_META_PHRASES == ()


CTA_ACTION_POSITIVE_CASES = [
    ("buy", "Buy now."),
    ("buy", "Buy this today."),
    ("buy", "You can buy it below."),
    ("buy", "Click below to buy."),
    ("shop", "Shop now."),
    ("shop", "Shop these shades."),
    ("shop", "Tap the link to shop."),
    ("shop", "You can shop the collection below."),
    ("order", "Order now."),
    ("order", "Order yours today."),
    ("order", "You can order it below."),
    ("order", "Tap the link to order."),
    ("order", "Order this set while it's available."),
    ("order", "Go order yours."),
    ("grab", "Grab some below."),
    ("grab", "Grab them while they're available."),
    ("grab", "Go grab yours."),
    ("grab", "You can grab them in a set of three."),
    ("get", "Get yours today."),
    ("get", "You can get this set below."),
    ("click", "Click the link."),
    ("click", "Click below to buy."),
    ("tap", "Tap the link."),
    ("tap", "Tap the link to order."),
    ("check", "Check it out below."),
    ("check", "Check these out below."),
    ("check", "Check these out."),
    ("check", "Check them out."),
    ("drop", "Drop it down below."),
    ("pick", "Pick yours today."),
    ("add", "Add to cart."),
    ("buy", "Please buy this today."),
    ("shop", "So shop now while it's available."),
]


@pytest.mark.parametrize("cue,text", CTA_ACTION_POSITIVE_CASES)
def test_context_aware_cta_action_matrix_accepts_viewer_actions(cue, text):
    assert pipeline.cta_action_rule(text) is not None, cue
    assert pipeline.classify_slot(text) == "CTA", cue


CTA_ACTION_NEGATIVE_CASES = [
    ("buy", "Buying this was part of my routine."),
    ("buy", "Buyers love the texture."),
    ("buy", "I decided to buy it yesterday."),
    ("shop", "I went to the shop yesterday."),
    ("shop", "The shop closes at five."),
    ("shop", "The workshop starts tomorrow."),
    ("order", "Order of application matters for this serum."),
    ("order", "The order arrived yesterday."),
    ("order", "My order came in damaged."),
    ("order", "I changed the order of the clips."),
    ("order", "In order to use this, shake it first."),
    ("order", "The correct order is cleanser, serum, moisturizer."),
    ("order", "Order numbers appear on the receipt."),
    ("order", "Order of application matters today."),
    ("order", "Order it alphabetically for the ingredient list."),
    ("grab", "I grab some before the gym."),
    ("grab", "She told me to grab them from the table."),
    ("grab", "We grabbed them before leaving."),
    ("grab", "Grab them from the table."),
    ("get", "I get headaches from strong scents."),
    ("get", "She gets ready at six."),
    ("get", "I get this question every day."),
    ("get", "Get headaches checked by a professional."),
    ("click", "I heard a click near the cap."),
    ("click", "The lid clicks into place."),
    ("tap", "The tap was leaking yesterday."),
    ("tap", "I felt a tap on my shoulder."),
    ("check", "I checked the ingredients yesterday."),
    ("check", "Check the ingredients before mixing the formula."),
    ("drop", "I dropped it on the table."),
    ("drop", "A drop of serum is enough."),
    ("pick", "I picked this up yesterday."),
    ("pick", "The pick was part of my routine."),
    ("add", "I add this after moisturizer."),
    ("add", "Add two drops to the mixture."),
    ("buy", "She told me to buy now, but I waited."),
    ("shop", "He asked us to shop now for tomorrow's event."),
    ("order", "She told him to order now, and he did."),
    ("tap", "She told me to tap the link yesterday."),
    ("click", "He asked me to click below during the demo."),
    ("check", "She reminded me to check it out below later."),
]


@pytest.mark.parametrize("cue,text", CTA_ACTION_NEGATIVE_CASES)
def test_context_aware_cta_action_matrix_rejects_narration_and_nouns(cue, text):
    assert pipeline.cta_action_rule(text) is None, cue
    assert pipeline.classify_slot(text) != "CTA", cue


def test_false_cta_narration_cannot_displace_real_cta_in_composer():
    narration = _composer_clip("order_explanation", 0, "OTHER", score=0.99)
    narration["text"] = "Order of application matters for this serum."
    narration["slot"] = pipeline.classify_slot(narration["text"])
    real_cta = _composer_clip("real_cta", 1, "CTA", score=0.80)
    real_cta["text"] = "Order yours today."
    real_cta["slot"] = pipeline.classify_slot(real_cta["text"])

    composer = pipeline.build_composer([narration, real_cta])

    assert composer["cta_id"] == "real_cta"
    assert composer["used_clip_ids"] == ["order_explanation", "real_cta"]


POSITIONAL_CTA_FRAME_CASES = [
    ("Go get yours.", "viewer_directed_go"),
    ("You can grab it below.", "viewer_directed_modal"),
    ("Please click the link.", "imperative_action"),
    ("Make sure you order today.", "viewer_directed_reminder"),
    ("Don't forget to shop the collection.", "viewer_directed_reminder"),
    ("You should check it out.", "viewer_directed_modal"),
    ("Go ahead and add it to your cart.", "viewer_directed_go"),
    ("Okay, buy yours today.", "imperative_action"),
    ("Well, you can order it below.", "viewer_directed_modal"),
]


@pytest.mark.parametrize("text,expected_frame", POSITIONAL_CTA_FRAME_CASES)
def test_position_aware_viewer_action_frames_accept_connected_commands(text, expected_frame):
    frames = pipeline.cta_action_frames(text)
    assert any(frame.frame_type == expected_frame for frame in frames), text
    assert pipeline.classify_slot(text) == "CTA"


POSITIONAL_CTA_NARRATION_CASES = [
    ("I go to get this set every week.", "first_person_narration"),
    ("Well, I go to get this set every week.", "first_person_narration"),
    ("She told me to go get the package.", "reported_speech"),
    ("I told him you can buy it later.", "reported_speech"),
    ("The first thing I did was click the old link.", "historical_action"),
    ("I usually get this set in summer.", "historical_action"),
    ("You can see why I decided to buy it.", "historical_action"),
    ("She said, 'go order it,' but I waited.", "reported_speech"),
    ("The word go appears before another unrelated action to get attention.", "unrelated_prefix"),
    ("I buy it because you can see the difference.", "first_person_narration"),
    ("He can get this set whenever it is available.", "third_person_narration"),
    ("She can buy it later if she wants.", "third_person_narration"),
    ("You can see the benefit, I get this set every week.", "first_person_narration"),
    ("I get this set every week, you can see why it works.", "first_person_narration"),
    ("She told me, make sure you order today.", "reported_speech"),
]


@pytest.mark.parametrize("text,expected_frame", POSITIONAL_CTA_NARRATION_CASES)
def test_position_aware_viewer_action_frames_reject_unconnected_prefixes(text, expected_frame):
    frames = pipeline.cta_action_frames(text)
    assert frames, text
    assert any(frame.frame_type == expected_frame for frame in frames), text
    assert pipeline.cta_action_rule(text) is None
    assert pipeline.classify_slot(text) != "CTA"


@pytest.mark.parametrize("text", [
    "Buy yours today, I use mine every morning.",
    "I use mine every morning, so buy yours today.",
    "I usually get this set, and you should buy yours today.",
    "I checked the old link yesterday, so click the link below.",
    "She said this formula works. Buy yours today.",
])
def test_genuine_cta_clause_is_not_qualified_by_or_borrowed_from_other_clause(text):
    assert pipeline.classify_slot(text) == "CTA"


def test_multiple_actions_bind_to_their_own_local_cta_frames():
    text = "I usually get this set, and you should buy yours today."
    frames = pipeline.cta_action_frames(text)
    assert [(frame.action, frame.frame_type) for frame in frames] == [
        ("get", "historical_action"),
        ("buy", "viewer_directed_modal"),
    ]


def test_false_viewer_prefix_narration_cannot_displace_real_cta_in_composer():
    narration = _composer_clip("false_prefix", 0, "OTHER", score=0.99)
    narration["text"] = "I go to get this set every week."
    narration["slot"] = pipeline.classify_slot(narration["text"])
    real_cta = _composer_clip("real_cta", 1, "CTA", score=0.90)
    real_cta["text"] = "You should buy yours today."
    real_cta["slot"] = pipeline.classify_slot(real_cta["text"])
    alternative = _composer_clip("alternative_cta", 2, "CTA", score=0.70)
    alternative["text"] = "Shop now."

    composer = pipeline.build_composer([narration, real_cta, alternative])

    assert narration["slot"] != "CTA"
    assert composer["cta_id"] == "real_cta"
    assert composer["used_clip_ids"] == ["false_prefix", "real_cta"]


EXPLICIT_LINK_CTA_CASES = [
    "Link below.",
    "Link in bio.",
    "Check the link.",
    "Check the link below.",
    "The link is below.",
    "The link is in my bio.",
    "Tap the link.",
    "Click the link below.",
    "So, link below.",
    "Okay, check the link.",
    "And the link is in my bio.",
]


@pytest.mark.parametrize("text", EXPLICIT_LINK_CTA_CASES)
def test_explicit_link_instructions_are_cta(text):
    assert pipeline.cta_action_rule(text) == "explicit_link_instruction"
    assert pipeline.classify_slot(text) == "CTA"


DESCRIPTIVE_LINK_NARRATION_CASES = [
    "The link between these ingredients is interesting.",
    "I checked the link yesterday.",
    "This chain has a broken link.",
    "The website link was incorrect in the old post.",
    "She said the link was unavailable.",
]


@pytest.mark.parametrize("text", DESCRIPTIVE_LINK_NARRATION_CASES)
def test_descriptive_or_historical_link_narration_is_not_cta(text):
    assert pipeline.cta_action_rule(text) is None
    assert pipeline.classify_slot(text) != "CTA"


def test_link_cta_is_selected_without_false_link_narration_displacing_it():
    false_cta = _composer_clip("link_narration", 0, "OTHER", score=0.99)
    false_cta["text"] = "I checked the link yesterday."
    false_cta["slot"] = pipeline.classify_slot(false_cta["text"])
    link_cta = _composer_clip("link_cta", 1, "CTA", score=0.80)
    link_cta["text"] = "Link below."
    link_cta["slot"] = pipeline.classify_slot(link_cta["text"])
    alternative = _composer_clip("alternative_cta", 2, "CTA", score=0.70)
    alternative["text"] = "Shop now."

    composer = pipeline.build_composer([false_cta, link_cta, alternative])

    assert composer["cta_id"] == "link_cta"
    assert composer["used_clip_ids"] == ["link_narration", "link_cta"]


def _source_fragment(source_index, source_local, *, duplicate=False):
    text = "Repeated residual text." if duplicate else "Impossible residual fragment."
    clip = _clip("ASR0000_c0", 1.0, 1.02, text)
    pipeline.add_source_metadata([clip], source_index, source_local, namespace_ids=True)
    return clip


def test_multisource_discard_diagnostics_have_deterministic_source_namespaces():
    diagnostics = []
    for source_index, source_local in enumerate(("first.mp4", "second.mp4")):
        fragment = _source_fragment(source_index, source_local)
        assert pipeline.validate_clip_boundaries(
            [fragment], discarded_diagnostics=diagnostics
        ) == []

    assert [item["original_clip_id"] for item in diagnostics] == ["ASR0000_c0", "ASR0000_c0"]
    assert [item["namespaced_clip_id"] for item in diagnostics] == [
        "source_000:ASR0000_c0", "source_001:ASR0000_c0",
    ]
    assert [item["diagnostic_id"] for item in diagnostics] == [
        "source_000:ASR0000_c0:discarded_invalid_microfragment",
        "source_001:ASR0000_c0:discarded_invalid_microfragment",
    ]
    assert [item["source_local"] for item in diagnostics] == ["first.mp4", "second.mp4"]
    assert all(item["source_start"] == 1.0 and item["source_end"] == 1.02 for item in diagnostics)
    assert all(item["text"] == "Impossible residual fragment." for item in diagnostics)


def test_single_source_clip_id_stays_backward_compatible_with_global_diagnostic_id():
    fragment = _clip("ASR0000_c0", 1.0, 1.02, "Impossible residual fragment.")
    pipeline.add_source_metadata([fragment], 0, "only.mp4", namespace_ids=False)
    diagnostics = []

    pipeline.validate_clip_boundaries([fragment], discarded_diagnostics=diagnostics)

    assert fragment["id"] == "ASR0000_c0"
    assert diagnostics[0]["clip_id"] == "ASR0000_c0"
    assert diagnostics[0]["namespaced_clip_id"] == "source_000:ASR0000_c0"
    assert diagnostics[0]["source_local"] == "only.mp4"


def test_namespaced_adjacent_duplicate_residual_keeps_source_diagnostics():
    first = _clip("ASR0000_c0", 0, 1, "Repeated residual text.")
    duplicate = _clip("ASR0000_c1", 1.01, 1.2, "Repeated residual text.")
    clips = [first, duplicate]
    pipeline.add_source_metadata(clips, 1, "second.mp4", namespace_ids=True)
    diagnostics = []

    validated = pipeline.validate_clip_boundaries(clips, discarded_diagnostics=diagnostics)

    assert [clip["id"] for clip in validated] == ["source_001:ASR0000_c0"]
    assert diagnostics[0]["reason"] == "discarded_duplicate_residual_text"
    assert diagnostics[0]["original_clip_id"] == "ASR0000_c1"
    assert diagnostics[0]["namespaced_clip_id"] == "source_001:ASR0000_c1"
    assert diagnostics[0]["source_local"] == "second.mp4"


def test_repaired_boundary_retains_source_identity_and_consistent_timing():
    fragment = _clip("ASR0000_c0", 4.0, 4.02, "A repaired spoken fragment.")
    fragment["words"] = [
        {"start": 4.0, "end": 4.12, "word": " A"},
        {"start": 4.13, "end": 4.30, "word": " repaired"},
        {"start": 4.31, "end": 4.50, "word": " fragment."},
    ]
    pipeline.add_source_metadata([fragment], 2, "third.mp4", namespace_ids=True)

    repaired = pipeline.validate_clip_boundaries([fragment])[0]

    assert repaired["meta"]["boundary_diagnostic"] == "repaired_from_word_timestamps"
    assert repaired["source_index"] == 2
    assert repaired["source_local"] == "third.mp4"
    assert repaired["id"] == repaired["namespaced_clip_id"] == "source_002:ASR0000_c0"
    assert repaired["start"] == repaired["source_start"] == 4.0
    assert repaired["end"] == repaired["source_end"] == 4.5


class _BenchmarkDiagnosticS3:
    def __init__(self):
        self.storage = {}

    def list_objects_v2(self, **_kwargs):
        return {"Contents": [{"Key": "Editdna good videos/clip.mp4", "Size": 2000}], "IsTruncated": False}

    def get_object(self, Bucket, Key):
        if Key in self.storage:
            body = self.storage[Key]
        elif Key.endswith("take_judge_dataset.jsonl"):
            body = b'{"session_id":"clip","clip_id":"old","text":"hello","keep":true,"slot":"HOOK","source":"good"}\n'
        else:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"ContentLength": len(body), "Body": io.BytesIO(body)}

    def put_object(self, **kwargs):
        self.storage[kwargs["Key"]] = kwargs["Body"]


def test_benchmark_session_output_preserves_enriched_discard_diagnostics(monkeypatch):
    for name, value in {
        "S3_BUCKET": "test", "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test",
    }.items():
        monkeypatch.setenv(name, value)
    diagnostic = {
        "source_index": 1, "source_local": "/tmp/private-worker/second.mp4", "original_clip_id": "ASR0000_c0",
        "namespaced_clip_id": "source_001:ASR0000_c0",
        "diagnostic_id": "source_001:ASR0000_c0:discarded_invalid_microfragment",
        "clip_id": "ASR0000_c0", "source_start": 1.0, "source_end": 1.02,
        "start": 1.0, "end": 1.02, "reason": "discarded_invalid_microfragment", "text": "fragment",
    }
    s3 = _BenchmarkDiagnosticS3()
    request = {
        "dataset_key": "editdna/training/take_judge_dataset.jsonl",
        "source_prefixes": ["Editdna good videos/"], "mode": "old_vs_new",
    }

    benchmark.run_benchmark(
        "diagnostic-job", request, s3=s3,
        pipeline=lambda *_args: {
            "clips": [],
            "clean_cut_discard_diagnostics": [diagnostic],
        },
    )

    session_key = next(key for key in s3.storage if "/sessions/" in key)
    session = json.loads(s3.storage[session_key])
    persisted = session["clean_cut_discard_diagnostics"][0]
    assert persisted == {**diagnostic, "source_local": "second.mp4"}
    assert "/tmp/" not in json.dumps(session)


UNICODE_TRANSCRIPT_CASES = [
    "这款产品让皮肤感觉很柔软。",
    "この商品は毎朝使っています。",
    "이 제품은 피부에 부드럽게 발려요.",
    "Этот продукт легко наносится на кожу.",
    "هذا المنتج لطيف على البشرة.",
    "Αυτό το προϊόν είναι απαλό στο δέρμα.",
    "המוצר הזה נעים לשימוש בכל בוקר.",
    "यह उत्पाद त्वचा पर बहुत हल्का लगता है।",
    "Este sérum deja la piel hidratada.",
    "Ce produit est très léger et agréable.",
    "Product رائع للبشرة اليومية.",
]


@pytest.mark.parametrize("text", UNICODE_TRANSCRIPT_CASES)
def test_unicode_transcripts_produce_tokens_and_remain_clean_cut_eligible(text):
    normalized = pipeline.normalized_text(text)
    assert normalized.tokens
    clip = _clip("unicode", 0, 2, text)
    pipeline.tag_clips_heuristic([clip])
    assert clip["slot"] == "OTHER"
    assert clip["meta"]["fallback_slot_rule"] == "unclassified_product_context"
    assert clip["meta"]["keep"] is True
    assert pipeline.select_clean_cut_clip_ids([clip]) == ["unicode"]


@pytest.mark.parametrize("text", ["...", "？！", "— —", "‘’", "   "])
def test_punctuation_only_transcripts_remain_empty(text):
    assert pipeline.normalized_text(text).tokens == ()
    assert pipeline.filler_rule(text) == "empty"


def test_unicode_normalizer_preserves_mixed_scripts_and_smart_contractions():
    mixed = pipeline.normalized_text("新しい Serum 2026 رائع")
    assert mixed.tokens == ("新しい", "serum", "2026", "رائع")
    assert pipeline.normalized_text("We’re ready").tokens == pipeline.normalized_text("We're ready").tokens


COMPOUND_TAKE_SLATE_CASES = [
    "Take two.",
    "Okay, take three.",
    "All right, take 2.",
    "So, take 3.",
    "Take two, wait, start over.",
    "Take three, hold on.",
    "Take number two, let me restart.",
    "Take number three, do that again.",
    "Okay, take two, we’re rolling.",
    "Okay, take two, we’re starting over.",
    "Take two, no, redo that.",
    "This is take two.",
    "That’s take two.",
    "We’re on take three.",
]


@pytest.mark.parametrize("text", COMPOUND_TAKE_SLATE_CASES)
def test_compound_take_slates_are_production_meta_and_excluded_from_clean_cut(text):
    assert pipeline.is_compound_take_slate(pipeline.normalized_text(text))
    assert pipeline.classify_slot_rule(text) == ("OTHER", "production_meta_phrase")
    clip = _clip("slate", 0, 1, text)
    pipeline.tag_clips_heuristic([clip])
    assert clip["meta"]["keep"] is False
    assert clip["meta"]["filler_rule"] == "production_meta_phrase"
    assert pipeline.select_clean_cut_clip_ids([clip]) == []


TAKE_PRODUCT_NARRATION_CASES = [
    "Take two gummies every morning.",
    "Take three capsules with food.",
    "I take two before bed.",
    "You can take three tablets daily.",
    "This routine takes two minutes.",
    "Take two shades and blend them together.",
    "Take number two from the numbered product samples.",
]


@pytest.mark.parametrize("text", TAKE_PRODUCT_NARRATION_CASES)
def test_take_dosage_quantity_and_product_instructions_are_not_slates(text):
    assert not pipeline.is_compound_take_slate(pipeline.normalized_text(text))
    assert pipeline.production_meta_rule(text) is None
    clip = _clip("product", 0, 2, text)
    pipeline.tag_clips_heuristic([clip])
    assert clip["meta"]["keep"] is True
    assert pipeline.select_clean_cut_clip_ids([clip]) == ["product"]


@pytest.mark.parametrize("source,expected", [
    ("/tmp/job-123/input.mp4", "input.mp4"),
    ("/var/lib/worker/nested/video.mov", "video.mov"),
    ("safe-name.mp4", "safe-name.mp4"),
    ("uploads/session/video.mp4", "uploads/session/video.mp4"),
    ("s3://private-bucket/uploads/session/video.mp4", "uploads/session/video.mp4"),
    ("https://user:secret@example.test/private/video.mp4?token=secret", "video.mp4"),
])
def test_persisted_diagnostic_source_identifiers_are_sanitized(source, expected):
    assert pipeline.sanitize_source_identifier(source, 4) == expected


def test_result_json_persistence_defensively_sanitizes_diagnostics(monkeypatch):
    captured = {}

    class Client:
        def put_object(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(pipeline, "S3_BUCKET", "test-bucket")
    monkeypatch.setattr(pipeline.boto3, "client", lambda _name: Client())
    result = {
        "session_id": "unicode-test",
        "clean_cut_discard_diagnostics": [{
            "source_index": 0,
            "source_local": "/tmp/private/session/input.mp4",
            "original_clip_id": "ASR0000_c0",
            "namespaced_clip_id": "source_000:ASR0000_c0",
            "diagnostic_id": "source_000:ASR0000_c0:discarded_invalid_microfragment",
            "reason": "discarded_invalid_microfragment",
            "source_start": 1.0,
            "source_end": 1.02,
            "text": "fragment",
        }],
    }

    pipeline.save_result_json_to_s3(result)

    persisted = json.loads(captured["Body"])
    assert persisted["clean_cut_discard_diagnostics"][0]["source_local"] == "input.mp4"
    assert "/tmp/" not in captured["Body"].decode()


UNICODE_SCORING_CASES = [
    "这款精华让肌肤保持水润柔软而且每天使用都非常舒服。",
    "この美容液は肌をしっとり柔らかく保ち毎朝快適に使えます。",
    "이 세럼은 피부를 촉촉하고 부드럽게 유지해서 매일 편하게 사용할 수 있어요.",
    "เซรั่มนี้ช่วยให้ผิวนุ่มชุ่มชื้นและใช้ได้สบายทุกเช้า",
    "សេរ៉ូមនេះជួយឱ្យស្បែកទន់មានសំណើមនិងប្រើបានរាល់ព្រឹក",
    "ເຊຣັ່ມນີ້ຊ່ວຍໃຫ້ຜິວນຸ່ມຊຸ່ມຊື່ນແລະໃຊ້ສະບາຍທຸກເຊົ້າ",
    "ဒီဆီရမ်က အသားအရေကို နူးညံ့စိုပြေစေပြီး မနက်တိုင်း သုံးရတာ အဆင်ပြေပါတယ်။",
    "Эта сыворотка делает кожу мягкой и увлажненной и подходит для ежедневного ухода.",
    "هذا المصل يجعل البشرة ناعمة ورطبة ومريحة للاستخدام كل صباح.",
    "המוצר הזה משאיר את העור רך ולח ונעים לשימוש בכל בוקר.",
    "यह सीरम त्वचा को मुलायम और नम रखता है और हर सुबह आसानी से लगाया जाता है।",
    "Αυτός ο ορός διατηρεί το δέρμα απαλό και ενυδατωμένο για άνετη καθημερινή χρήση.",
    "Este sérum ligero mantiene mi piel hidratada suave cómoda y luminosa durante todo el día.",
    "Daily精华让肌肤保持水润柔软and feels comfortable every morning.",
]


@pytest.mark.parametrize("text", UNICODE_SCORING_CASES)
def test_shared_unicode_content_measure_scores_meaningful_transcripts_for_composer(text):
    measure = pipeline.semantic_content_measure(text)
    assert measure.token_count > 0
    assert measure.alphanumeric_count > 0
    assert measure.effective_semantic_units > 0
    assert measure.scoring_rule
    clip = _clip("meaningful", 0, 3, text)
    pipeline.tag_clips_heuristic([clip])
    assert clip["semantic_score"] >= pipeline.COMPOSER_MIN_SEMANTIC
    assert clip["meta"]["semantic_content_measure"]["effective_semantic_units"] == measure.effective_semantic_units
    assert clip["meta"]["keep"] is True


@pytest.mark.parametrize("text", ["好", "あ", "가", "ก", "...", "？！", "😀😀😀", "✨ — ✨"])
def test_short_or_symbol_only_unicode_content_remains_ineligible(text):
    measure = pipeline.semantic_content_measure(text)
    clip = _clip("short", 0, 0.5, text)
    pipeline.tag_clips_heuristic([clip])
    assert clip["semantic_score"] < pipeline.COMPOSER_MIN_SEMANTIC
    if measure.effective_semantic_units == 0:
        assert clip["meta"]["keep"] is False


def test_english_content_scoring_remains_compatible_and_bounded():
    short = "These five words remain normally scored."
    assert pipeline.semantic_content_measure(short).scoring_rule == "normalized_token_count"
    assert pipeline.semantic_content_measure(short).effective_semantic_units == 6
    assert pipeline.semantic_content_score(short) == pytest.approx(0.58)
    long_text = " ".join(f"word{index}" for index in range(40))
    assert pipeline.semantic_content_score(long_text) == 0.95


def test_unsegmented_clips_enter_human_and_blooper_composers_at_default_threshold():
    texts = [
        "这款精华让肌肤保持水润柔软而且每天使用都非常舒服。",
        "この美容液は肌をしっとり柔らかく保ち毎朝快適に使えます。",
    ]
    clips = []
    for index, text in enumerate(texts):
        clip = _clip(f"unicode-{index}", index * 4, index * 4 + 3, text)
        pipeline.tag_clips_heuristic([clip])
        clips.append(clip)

    human = pipeline.build_composer([copy.deepcopy(clip) for clip in clips], mode="human")
    blooper = pipeline.build_composer([copy.deepcopy(clip) for clip in clips], mode="blooper")

    assert human["used_clip_ids"] == ["unicode-0", "unicode-1"]
    assert blooper["used_clip_ids"] == ["unicode-0", "unicode-1"]


def test_unicode_scoring_keeps_clean_cut_and_filler_behavior_separate():
    narration = _clip("unicode", 0, 3, "这款精华让肌肤保持水润柔软而且每天使用都非常舒服。")
    filler = _clip("filler", 4, 5, "Wait, let me redo that.")
    pipeline.tag_clips_heuristic([narration, filler])
    assert pipeline.select_clean_cut_clip_ids([narration, filler]) == ["unicode"]
    assert filler["semantic_score"] == 0.0
    assert filler["meta"]["keep"] is False


def test_unicode_content_measure_flows_into_semantic_v2_and_take_judge_fallbacks():
    text = "这款精华让肌肤保持水润柔软而且每天使用都非常舒服。"
    clip = _clip("unicode", 0, 3, text)
    pipeline.tag_clips_heuristic([clip])
    clause = pipeline.build_clause_inputs([clip])[0]
    delivery = pipeline.delivery_features(clip)
    expected_units = pipeline.semantic_content_measure(text).effective_semantic_units
    assert clause.word_count == expected_units
    assert clause.sentence_completeness == 1.0
    assert delivery.word_count == expected_units
    assert delivery.incomplete_phrase is False
