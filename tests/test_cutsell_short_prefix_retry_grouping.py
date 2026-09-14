from cutsell_worker.contracts import CandidateTake
from cutsell_worker.take_grouping import group_takes


def take(clip_id, text, start, end, source="src"):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id=source,
        source_order=0,
        start=start,
        end=end,
        text=text,
    )


def groups(*takes):
    return tuple(group_takes(takes).values())


def test_three_word_exact_prefix_false_start_joins_nearby_long_retry():
    short = take("short", "I'm literally all", 10.0, 11.0)
    full = take("full", "I'm literally all about finding the best hacks", 13.0, 16.0)

    result = groups(short, full)

    assert len(result) == 1
    assert result[0] == (short, full)


def test_one_word_reaction_never_joins_longer_prefix_take():
    short = take("short", "What", 10.0, 10.4)
    full = take("full", "What just happened to this package today", 11.0, 14.0)

    result = groups(short, full)

    assert len(result) == 2


def test_distant_three_word_prefix_remains_independent():
    short = take("short", "I'm literally all", 10.0, 11.0)
    full = take("full", "I'm literally all about finding the best hacks", 30.0, 33.0)

    result = groups(short, full)

    assert len(result) == 2


def test_cross_source_three_word_prefix_remains_independent():
    short = take("short", "I'm literally all", 10.0, 11.0, source="a")
    full = take("full", "I'm literally all about finding the best hacks", 12.0, 15.0, source="b")

    result = groups(short, full)

    assert len(result) == 2


def test_changed_short_phrase_does_not_gain_prefix_exception():
    short = take("short", "I'm honestly all", 10.0, 11.0)
    full = take("full", "I'm literally all about finding the best hacks", 12.0, 15.0)

    result = groups(short, full)

    assert len(result) == 2
