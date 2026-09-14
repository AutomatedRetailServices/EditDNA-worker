from cutsell_worker.contracts import CandidateTake
from cutsell_worker.take_segmentation import _repair_boundary_fragments


def _take(clip_id, start, end, text, *, source="src", complete=False):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id=source,
        source_order=0,
        start=start,
        end=end,
        text=text,
        complete_idea=complete,
    )


def test_dependent_connector_fragment_bridges_short_asr_gap_forward():
    takes = (
        _take("a", 201.85, 205.03, "And I just think that this was such a great video", complete=True),
        _take("bridge", 205.67, 207.77, "not just because"),
        _take("b", 208.33, 214.61, "it was hot but because it was funny and fun to make", complete=True),
    )
    repaired = _repair_boundary_fragments(takes)
    assert len(repaired) == 2
    assert repaired[0].clip_id == "a"
    assert repaired[1].start == 205.67
    assert repaired[1].end == 214.61
    assert repaired[1].text == "not just because it was hot but because it was funny and fun to make"


def test_short_nonconnector_fragment_does_not_bridge_wider_gap():
    takes = (
        _take("prefix", 181.18, 183.20, "Por temporada"),
        _take("full", 183.70, 189.00, "me salía un acné en la espalda con la que yo resolvía con resorcina", complete=True),
    )
    repaired = _repair_boundary_fragments(takes)
    assert tuple(take.clip_id for take in repaired) == ("prefix", "full")


def test_intentional_short_line_does_not_bridge_without_connector():
    takes = (
        _take("short", 10.0, 11.2, "That was wild", complete=True),
        _take("next", 11.65, 16.0, "Then we started filming the next scene", complete=True),
    )
    repaired = _repair_boundary_fragments(takes)
    assert tuple(take.clip_id for take in repaired) == ("short", "next")


def test_bridge_never_crosses_source_boundary():
    takes = (
        _take("bridge", 5.0, 6.4, "not just because", source="src-a"),
        _take("next", 6.8, 10.0, "it was funny and easy to make", source="src-b", complete=True),
    )
    repaired = _repair_boundary_fragments(takes)
    assert tuple(take.clip_id for take in repaired) == ("bridge", "next")


def test_bridge_never_crosses_real_pause():
    takes = (
        _take("bridge", 5.0, 6.4, "not just because"),
        _take("next", 7.2, 10.0, "it was funny and easy to make", complete=True),
    )
    repaired = _repair_boundary_fragments(takes)
    assert tuple(take.clip_id for take in repaired) == ("bridge", "next")
