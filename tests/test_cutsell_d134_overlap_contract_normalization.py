"""D-134: engine-facing request-contract hardening tests.

Covers the ONE normalization point (`cutsell_worker.serde.
_normalize_dialogue_overlap`, wired into `request_from_dict`) for the
canonical `dialogue_overlap_enabled` field vs. the legacy iOS `audio_overlap`
field (D-129 naming), the optional descriptive-only iOS media-ingestion
metadata contract (D-133 reality), serde backward compatibility, source
identity immutability, and structural non-influence on D-123/D-128/Boundary/
render-plan (this task implements NO Overlap pacing behavior and NO fallback
authority -- see docs/CUTSELL_DECISIONS.md D-134).
"""
from __future__ import annotations

from pathlib import Path

from cutsell_worker.contracts import ProcessingRequest
from cutsell_worker.serde import (
    DIALOGUE_OVERLAP_SOURCE_CANONICAL,
    DIALOGUE_OVERLAP_SOURCE_DEFAULT,
    DIALOGUE_OVERLAP_SOURCE_LEGACY,
    _normalize_dialogue_overlap,
    request_from_dict,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _payload(**overrides) -> dict:
    base = {
        "project_id": "proj-1",
        "user_id": "user-1",
        "sources": [
            {
                "source_asset_id": "src-1",
                "original_name": "clip.mp4",
                "source_order": 0,
                "duration_sec": 12.5,
                "uri": "s3://bucket/clip.mp4",
            }
        ],
        "language_hint": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1-9: normalization precedence + diagnostics (Part 2, Part 5)
# ---------------------------------------------------------------------------

def test_legacy_false_normalizes_false():
    enabled, diag = _normalize_dialogue_overlap({"audio_overlap": False})
    assert enabled is False
    assert diag["dialogue_overlap_source"] == DIALOGUE_OVERLAP_SOURCE_LEGACY


def test_legacy_true_normalizes_true():
    enabled, diag = _normalize_dialogue_overlap({"audio_overlap": True})
    assert enabled is True
    assert diag["dialogue_overlap_source"] == DIALOGUE_OVERLAP_SOURCE_LEGACY


def test_canonical_false_normalizes_false():
    enabled, diag = _normalize_dialogue_overlap({"dialogue_overlap_enabled": False})
    assert enabled is False
    assert diag["dialogue_overlap_source"] == DIALOGUE_OVERLAP_SOURCE_CANONICAL


def test_canonical_true_normalizes_true():
    enabled, diag = _normalize_dialogue_overlap({"dialogue_overlap_enabled": True})
    assert enabled is True
    assert diag["dialogue_overlap_source"] == DIALOGUE_OVERLAP_SOURCE_CANONICAL


def test_canonical_wins_over_conflicting_legacy():
    enabled, diag = _normalize_dialogue_overlap(
        {"dialogue_overlap_enabled": True, "audio_overlap": False}
    )
    assert enabled is True
    assert diag["dialogue_overlap_source"] == DIALOGUE_OVERLAP_SOURCE_CANONICAL
    assert diag["legacy_audio_overlap_present"] is True
    assert diag["canonical_dialogue_overlap_present"] is True

    enabled2, diag2 = _normalize_dialogue_overlap(
        {"dialogue_overlap_enabled": False, "audio_overlap": True}
    )
    assert enabled2 is False
    assert diag2["dialogue_overlap_source"] == DIALOGUE_OVERLAP_SOURCE_CANONICAL


def test_neither_present_preserves_current_default():
    enabled, diag = _normalize_dialogue_overlap({})
    assert enabled is False  # current effective default, confirmed from code (Part 4)
    assert diag["dialogue_overlap_source"] == DIALOGUE_OVERLAP_SOURCE_DEFAULT
    assert diag["legacy_audio_overlap_present"] is False
    assert diag["canonical_dialogue_overlap_present"] is False


def test_source_diagnostic_canonical_request():
    _, diag = _normalize_dialogue_overlap({"dialogue_overlap_enabled": True})
    assert diag["dialogue_overlap_source"] == "canonical_request"


def test_source_diagnostic_legacy_audio_overlap():
    _, diag = _normalize_dialogue_overlap({"audio_overlap": True})
    assert diag["dialogue_overlap_source"] == "legacy_audio_overlap"


def test_source_diagnostic_default():
    _, diag = _normalize_dialogue_overlap({"project_id": "x"})
    assert diag["dialogue_overlap_source"] == "default"


# ---------------------------------------------------------------------------
# 10-14: serde / request-shape backward compatibility (Part 12)
# ---------------------------------------------------------------------------

def test_old_serialized_request_loads():
    """A. old request: audio_overlap only -> loads."""
    request = request_from_dict(_payload(audio_overlap=True))
    assert isinstance(request, ProcessingRequest)
    assert request.audio_overlap is True
    assert request.dialogue_overlap_enabled is True
    assert request.overlap_diagnostics["dialogue_overlap_source"] == "legacy_audio_overlap"


def test_canonical_request_loads():
    """B. new request: dialogue_overlap_enabled only -> loads."""
    request = request_from_dict(_payload(dialogue_overlap_enabled=True))
    assert request.dialogue_overlap_enabled is True
    assert request.audio_overlap is False  # legacy field's own default, untouched
    assert request.overlap_diagnostics["dialogue_overlap_source"] == "canonical_request"


def test_mixed_request_loads():
    """C. mixed request: both present -> canonical wins."""
    request = request_from_dict(
        _payload(audio_overlap=False, dialogue_overlap_enabled=True)
    )
    assert request.dialogue_overlap_enabled is True
    assert request.overlap_diagnostics["dialogue_overlap_source"] == "canonical_request"


def test_optional_ios_metadata_accepted():
    request = request_from_dict(
        _payload(
            sources=[
                {
                    "source_asset_id": "src-1",
                    "original_name": "clip.mov",
                    "source_order": 0,
                    "duration_sec": 9.0,
                    "uri": "s3://bucket/clip.mov",
                    "metadata": {
                        "container": "mov",
                        "codec": "hvc1",
                        "width": 1080,
                        "height": 1920,
                        "fps": 30.0,
                        "orientation_degrees": 90,
                        "mirrored_hint": True,
                        "duration": 9.0,
                        "file_size": 12345678,
                        "audio_track_present": True,
                        "audio_sample_rate": 44100,
                        "device_model": "iPhone15,3",
                        "ios_version": "17.4",
                        "capture_origin": "camera",
                    },
                }
            ]
        )
    )
    source = request.sources[0]
    assert source.metadata["codec"] == "hvc1"
    assert source.metadata["mirrored_hint"] is True
    assert source.metadata["device_model"] == "iPhone15,3"


def test_absent_metadata_accepted():
    request = request_from_dict(_payload())
    assert request.sources[0].metadata == {}


# ---------------------------------------------------------------------------
# 15-16: D-133 known-vs-unknown honesty preserved (Part 7)
# ---------------------------------------------------------------------------

def test_mirrored_hint_does_not_become_proven_mirrored():
    request = request_from_dict(
        _payload(
            sources=[
                {
                    "source_asset_id": "src-1",
                    "original_name": "clip.mov",
                    "source_order": 0,
                    "duration_sec": 5.0,
                    "uri": "s3://bucket/clip.mov",
                    "metadata": {"mirrored_hint": True},
                }
            ]
        )
    )
    metadata = request.sources[0].metadata
    assert metadata["mirrored_hint"] is True
    assert "mirrored" not in metadata  # never promoted to a bare "proven" key


def test_unknown_vfr_remains_unknown():
    request = request_from_dict(_payload())  # no metadata supplied at all
    assert request.sources[0].metadata.get("variable_frame_rate_hint") is None
    assert "variable_frame_rate_hint" not in request.sources[0].metadata


# ---------------------------------------------------------------------------
# 17-18: source identity immutability (Part 9)
# ---------------------------------------------------------------------------

def test_source_identity_unchanged_by_overlap_or_metadata_fields():
    base = request_from_dict(_payload())
    with_overlap = request_from_dict(_payload(dialogue_overlap_enabled=True))
    with_metadata = request_from_dict(
        _payload(
            sources=[
                {
                    "source_asset_id": "src-1",
                    "original_name": "clip.mp4",
                    "source_order": 0,
                    "duration_sec": 12.5,
                    "uri": "s3://bucket/clip.mp4",
                    "metadata": {"codec": "avc1", "device_model": "iPhone14,5"},
                }
            ]
        )
    )
    for other in (with_overlap, with_metadata):
        assert other.sources[0].source_asset_id == base.sources[0].source_asset_id
        assert other.sources[0].uri == base.sources[0].uri


def test_source_timestamps_and_duration_unchanged():
    base = request_from_dict(_payload())
    with_overlap = request_from_dict(_payload(dialogue_overlap_enabled=True))
    assert with_overlap.sources[0].duration_sec == base.sources[0].duration_sec
    assert with_overlap.sources[0].source_order == base.sources[0].source_order


# ---------------------------------------------------------------------------
# 19-22: no editorial difference at the contract level (Part 14)
# ---------------------------------------------------------------------------

def _non_overlap_fields(request: ProcessingRequest) -> tuple:
    return (
        request.project_id,
        request.user_id,
        tuple(
            (s.source_asset_id, s.original_name, s.source_order, s.duration_sec, s.uri)
            for s in request.sources
        ),
        request.preferred_source_order,
        request.language_hint,
    )


def test_no_overlap_fields_edit_unchanged():
    no_fields = request_from_dict(_payload())
    assert no_fields.dialogue_overlap_enabled is False


def test_legacy_false_edit_unchanged():
    legacy_false = request_from_dict(_payload(audio_overlap=False))
    no_fields = request_from_dict(_payload())
    assert _non_overlap_fields(legacy_false) == _non_overlap_fields(no_fields)
    assert legacy_false.dialogue_overlap_enabled == no_fields.dialogue_overlap_enabled is False


def test_canonical_false_edit_unchanged():
    canonical_false = request_from_dict(_payload(dialogue_overlap_enabled=False))
    no_fields = request_from_dict(_payload())
    assert _non_overlap_fields(canonical_false) == _non_overlap_fields(no_fields)
    assert canonical_false.dialogue_overlap_enabled == no_fields.dialogue_overlap_enabled is False


def test_canonical_true_edit_unchanged_no_consumer_exists():
    """dialogue_overlap_enabled=True is preserved in diagnostics but D-134
    implements no consumer, so every other request field -- and therefore
    the current edit -- is identical to the no-overlap-fields case."""
    canonical_true = request_from_dict(_payload(dialogue_overlap_enabled=True))
    no_fields = request_from_dict(_payload())
    assert _non_overlap_fields(canonical_true) == _non_overlap_fields(no_fields)
    assert canonical_true.dialogue_overlap_enabled is True
    assert canonical_true.overlap_diagnostics["dialogue_overlap_enabled"] is True


# ---------------------------------------------------------------------------
# 24-26: structural non-influence on D-128 Class B trigger / Boundary /
# render-plan selection (Part 15/Part 8) -- these modules never import or
# reference the new fields at all, verified directly against their source.
# ---------------------------------------------------------------------------

def test_d128_fallback_module_never_references_overlap_or_media_metadata():
    source = (REPO_ROOT / "cutsell_worker" / "multimodal_besttake_fallback.py").read_text()
    for forbidden in ("dialogue_overlap_enabled", "audio_overlap", "mirrored_hint", "device_model"):
        assert forbidden not in source


def test_boundary_engine_never_reads_dialogue_overlap_enabled():
    source = (REPO_ROOT / "cutsell_worker" / "boundary_engine_pass.py").read_text()
    assert "dialogue_overlap_enabled" not in source


def test_render_plan_never_reads_dialogue_overlap_enabled_or_media_metadata():
    source = (REPO_ROOT / "cutsell_worker" / "render_plan.py").read_text()
    assert "dialogue_overlap_enabled" not in source
    assert "mirrored_hint" not in source


# ---------------------------------------------------------------------------
# 27: no provider/network call introduced by this normalization
# ---------------------------------------------------------------------------

def test_normalization_module_makes_no_network_or_provider_import():
    source = (REPO_ROOT / "cutsell_worker" / "serde.py").read_text()
    for forbidden in ("requests", "httpx", "urllib", "genai", "google.generativeai", "modal", "runpod"):
        assert forbidden not in source
