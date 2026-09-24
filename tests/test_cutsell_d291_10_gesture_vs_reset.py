"""D-291.10: a kinematic reset CANDIDATE during continuous speech is a
gesture, not a reset -- applied in the three authorities where RAW #126
(project video00-modal-35931561397-1) let it decide:

1. BestTake CASE B materiality: the thyroid family {T1 25.6-34.6 (the
   abandoned first attempt, `alternate` 0.8), T2 35.46-46.42 (the complete
   retake, `winner` 0.95)} -- T2 carried 7 D-097-countable hand-motion
   candidates against T1's 4, so the semantic fast path was bypassed and
   DeliveryScore chose T1. On the frames every one of those candidates is
   the mic hand moving while she speaks; only 1 (T2) vs 2 (T1) lie near a
   measured pause. Material = pause-corroborated (D-149 / D-285).
2. Interior gap trim: covered by tests/test_cutsell_post_selection_
   interior_gap_trim.py (D-291.10 section).
3. Watch+Listen reset debris at edges: an edge candidate that overlaps a
   word the frozen clip itself carries is happening during speech --
   UNCERTAIN, never a confirmed FAIL (RAW #126: 6 FAILs, all of this shape,
   blocked delivery).
"""
from __future__ import annotations

from cutsell_worker import perceptual_watch_listen as pwl
from cutsell_worker.case_b_performance_evidence import build_case_b_performance_evidence
from cutsell_worker.contracts import CandidateTake, DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION, SemanticRole, Word
from cutsell_worker.pipeline import _case_b_fast_path_conflict, _material_delivery_event_count
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.render_plan import RenderSegment
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext

SRC = "src"
# RAW #126 whole_video_context events around the two thyroid takes (kind, start, end, confidence)
T1_EVENTS = [["audio_silence_interval", 22.209, 25.541, 0.9], ["audio_silence_interval", 22.384, 25.217, 1.0], ["hand_motion_reset_candidate", 25.202, 25.269, 0.843], ["hand_motion_reset_candidate", 25.402, 25.469, 1.0], ["hand_motion_reset_candidate", 31.136, 31.203, 0.898], ["facial_expression_shift_candidate", 31.336, 31.403, 0.728], ["hand_motion_reset_candidate", 31.736, 31.803, 1.0], ["hand_motion_reset_candidate", 32.469, 32.536, 0.925], ["audio_silence_interval", 32.668, 35.602, 0.9], ["hand_motion_reset_candidate", 33.336, 33.403, 1.0], ["audio_silence_interval", 34.796, 35.591, 1.0], ["facial_expression_shift_candidate", 34.936, 35.003, 0.733], ["hand_motion_reset_candidate", 35.136, 35.203, 0.902]]
T2_EVENTS = [["hand_motion_reset_candidate", 35.336, 35.403, 1.0], ["hand_motion_reset_candidate", 38.203, 38.27, 1.0], ["hand_motion_reset_candidate", 38.403, 38.47, 1.0], ["hand_motion_reset_candidate", 38.603, 38.67, 1.0], ["hand_motion_reset_candidate", 39.137, 39.203, 1.0], ["hand_motion_reset_candidate", 40.337, 40.403, 0.903], ["hand_motion_reset_candidate", 40.537, 40.603, 0.866], ["hand_motion_reset_candidate", 43.47, 43.537, 0.958], ["audio_silence_interval", 44.703, 48.884, 0.9], ["audio_silence_interval", 45.214, 48.884, 1.0], ["hand_motion_reset_candidate", 45.804, 45.87, 1.0]]
T1_WORDS = [["Nunca", 25.6, 25.84], ["se", 25.84, 25.98], ["nos", 25.98, 26.16], ["ocurrió", 26.16, 26.7], ["hacer", 26.7, 27.1], ["un", 27.1, 27.32], ["chequeo", 27.32, 27.66], ["de", 27.66, 27.78], ["sonografía", 27.78, 28.44], ["de", 28.44, 28.58], ["la", 28.58, 28.76], ["tiroides,", 28.76, 29.34], ["pues", 29.64, 29.82], ["porque", 29.82, 30.18], ["cada", 30.18, 30.5], ["año", 30.5, 30.78], ["que", 30.78, 30.96], ["me", 30.96, 31.1], ["hacía", 31.1, 31.46], ["mínimo", 31.46, 31.82], ["dos", 31.82, 32.42], ["estados.", 32.42, 34.6]]
T2_WORDS = [["Nunca", 35.46, 35.88], ["se", 35.88, 36.02], ["nos", 36.02, 36.3], ["ocurrió", 36.3, 36.84], ["hacer", 36.84, 37.36], ["un", 37.36, 37.62], ["chequeo", 37.62, 38.2], ["de", 38.2, 38.36], ["la", 38.36, 38.56], ["tiroides", 38.56, 39.02], ["por", 39.02, 39.28], ["sonografía", 39.28, 40.22], ["porque", 40.22, 40.62], ["siempre", 40.62, 41.18], ["en", 41.18, 41.3], ["mis", 41.3, 41.44], ["exámenes", 41.44, 42.32], ["la", 42.32, 42.58], ["tiroides", 42.58, 43.0], ["salía", 43.0, 43.36], ["como", 43.36, 43.62], ["que", 43.62, 43.86], ["estaba", 43.86, 44.16], ["funcionando", 44.16, 44.84], ["perfectamente.", 45.86, 46.42]]


def _take(cid, start, end, words):
    ws = tuple(Word(t, s, e) for t, s, e in words)
    return CandidateTake(cid, SRC, 0, start, end, " ".join(t for t, _, _ in words), words=ws, complete_idea=True)


def _context(events, *, with_silence=True):
    evs = tuple(
        TemporalEvent(SRC, float(s), float(e), kind, float(c), "")
        for kind, s, e, c in events if with_silence or kind != "audio_silence_interval"
    )
    return WholeVideoContext(
        sources=(SourceVideoContext(source_asset_id=SRC, summary="", dominant_style="", creator_intent="", events=evs),),
        status=ProviderStatus("test", True, True, "applied"),
    )


T1 = _take("T1", 25.6, 34.6, T1_WORDS)
T2 = _take("T2", 35.46, 46.42, T2_WORDS)


def test_raw126_thyroid_gestures_no_longer_bypass_the_decisive_semantic_winner():
    context = _context(T1_EVENTS + T2_EVENTS)
    evidence = {t.clip_id: build_case_b_performance_evidence(t, context) for t in (T1, T2)}
    # raw D-097-countable candidates: the pre-fix "material" counts
    assert sum(1 for e in evidence["T1"].delivery_events if e.d097_would_be_counted) == 4
    assert sum(1 for e in evidence["T2"].delivery_events if e.d097_would_be_counted) == 7
    # pause-corroborated: 2 (T1) vs 1 (T2) -- the retake is not the worse delivery
    assert _material_delivery_event_count(evidence["T1"]) == 2
    assert _material_delivery_event_count(evidence["T2"]) == 1
    assert _case_b_fast_path_conflict("T2", "T1", {"T1", "T2"}, evidence) is None


def test_without_any_silence_measurement_corroboration_is_not_evaluated_and_the_legacy_count_stands():
    context = _context(T1_EVENTS + T2_EVENTS, with_silence=False)
    evidence = {t.clip_id: build_case_b_performance_evidence(t, context) for t in (T1, T2)}
    assert all(e.pause_corroborated is None for e in evidence["T2"].delivery_events)
    assert _material_delivery_event_count(evidence["T2"]) == 7
    conflict = _case_b_fast_path_conflict("T2", "T1", {"T1", "T2"}, evidence)
    assert conflict is not None and conflict["semantic_fast_path_candidate_material_event_count"] == 7


def test_a_reset_next_to_a_measured_pause_still_counts_as_material():
    # the retake's ONE pause-corroborated candidate (45.804, next to the 45.214 pause) keeps counting
    context = _context(T1_EVENTS + T2_EVENTS)
    ev = build_case_b_performance_evidence(T2, context)
    corroborated = [e for e in ev.delivery_events if e.pause_corroborated]
    assert [round(e.start, 3) for e in corroborated] == [45.804]


# --- Watch+Listen: edge candidates during a spoken word -----------------------

def _draft(events, clips):
    diag = {"whole_video_context": {"sources": [{"source_asset_id": SRC, "events": list(events)}]}}
    return DraftTimeline(schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
                         selected=tuple(clips), alternates=(), discarded=(), diagnostics=diag)


def _clip(cid, start, end, words):
    return DraftClip(clip_id=cid, source_asset_id=SRC, source_order=0, start=start, end=end, text="x", caption_text="x",
                     words=tuple(Word(t, s, e) for t, s, e in words), semantic_role=SemanticRole.STORY, selected=True)


def _seg(cid, start, end):
    return RenderSegment(clip_id=cid, source_asset_id=SRC, source_path="/x.mp4", start=start, end=end)


def test_an_edge_candidate_during_the_first_spoken_word_is_uncertain_not_fail():
    # RAW #126 cruise clip entry: the clip starts at 13.78 on the word, the
    # pre-speech silence ends at 13.781, the hand moves the mic at 13.80-13.87
    events = [
        {"kind": "audio_silence_interval", "start": 10.996, "end": 13.781, "confidence": 1.0},
        {"kind": "hand_motion_reset_candidate", "start": 13.801, "end": 13.868, "confidence": 1.0},
    ]
    clip = _clip("cruise", 13.78, 23.28, [("No", 13.78, 13.95), ("es", 13.95, 14.10), ("secreto", 14.10, 14.60)])
    report = pwl._reset_debris_at_edges(_draft(events, [clip]), (_seg("cruise", 13.78, 23.28),), [(0.0, 9.5)])
    assert report.status == pwl.UNCERTAIN
    assert report.findings[0].severity == "UNCERTAIN"
    assert report.findings[0].detail["measured_pause_nearby"] is True
    assert report.findings[0].detail["during_spoken_word"] is True


def test_an_edge_candidate_before_the_first_word_next_to_a_pause_stays_fail():
    events = [
        {"kind": "audio_silence_interval", "start": 10.996, "end": 13.781, "confidence": 1.0},
        {"kind": "hand_motion_reset_candidate", "start": 13.801, "end": 13.868, "confidence": 1.0},
    ]
    clip = _clip("cruise", 13.78, 23.28, [("No", 13.95, 14.10), ("es", 14.10, 14.30)])  # speech starts after the motion
    report = pwl._reset_debris_at_edges(_draft(events, [clip]), (_seg("cruise", 13.78, 23.28),), [(0.0, 9.5)])
    assert report.status == pwl.EVALUATED_FAIL
    assert report.findings[0].detail["during_spoken_word"] is False


def test_without_word_evidence_the_d149_pause_rule_is_unchanged():
    events = [
        {"kind": "audio_silence_interval", "start": 9.60, "end": 10.02, "confidence": 1.0},
        {"kind": "hand_motion_reset_candidate", "start": 10.05, "end": 10.30, "confidence": 0.93},
    ]
    report = pwl._reset_debris_at_edges(_draft(events, []), (_seg("a", 10.0, 15.0),), [(0.0, 5.0)])
    assert report.status == pwl.EVALUATED_FAIL
