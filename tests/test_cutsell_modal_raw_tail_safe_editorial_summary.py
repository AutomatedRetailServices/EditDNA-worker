"""The "Tail-safe editorial summary" step of cutsell-video00-modal-raw.yml.

GitHub's job-log API serves only a bounded tail; on a real run the KEEP/
DISCARD record printed early scrolls out of reach. This step re-prints the
compact editorial record last. The test executes the step's ACTUAL embedded
script (yaml.safe_load, the same dedent Actions performs) against a
synthetic engine JSON and against a missing JSON, proving it re-projects
source times/text/ASR audit/QC status verbatim and never fails the job.
No paid compute; no cutsell_worker file touched.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import yaml

WORKFLOW_PATH = Path(".github/workflows/cutsell-video00-modal-raw.yml")
STEP_NAME = "Tail-safe editorial summary (sibling-safe; never fails)"


def _step_script() -> str:
    doc = yaml.safe_load(WORKFLOW_PATH.read_text())
    for step in doc["jobs"]["video00-modal-raw"]["steps"]:
        if step.get("name") == STEP_NAME:
            assert step.get("if") == "always()"
            return step["run"]
    raise AssertionError(f"step {STEP_NAME!r} not found")


def _run(tmp_path: Path, engine) -> tuple[subprocess.CompletedProcess, dict | None]:
    art = tmp_path / "artifact"
    art.mkdir()
    if engine is not None:
        (art / "video00-modal.json").write_text(json.dumps(engine, ensure_ascii=False))
    proc = subprocess.run(["bash", "-c", _step_script()], cwd=str(tmp_path),
                          env={**os.environ, "SOURCE_KEY": "Yaskira/01.mp4"},
                          capture_output=True, text=True, timeout=60)
    out = art / "editorial-summary-tail.json"
    return proc, (json.loads(out.read_text()) if out.is_file() else None)


def test_step_reprints_selection_asr_and_qc_from_engine_json(tmp_path):
    engine = {
        "ok": True,
        "source_media_sha256": "750e989c",
        "output_duration_sec": 100.5,
        "asr_provider_audit": {"provider": "faster-whisper-medium-whisperx", "model": "medium",
                               "status": "passed", "word_count": 321, "segment_count": 20,
                               "detected_language": "en", "fallback": None,
                               "alignment_runtime": {"device": "cuda"}},
        "selected": [{"clip_id": "a", "start": 0.4, "end": 57.9, "text": "I was like, oh, I…", "take_group_id": "g1"},
                     {"clip_id": "b", "start": 59.2, "end": 114.5, "text": "He is…", "take_group_id": "g2"}],
        "discarded": [{"clip_id": "c", "start": 57.9, "end": 59.2, "text": "because…"}],
        "live_render_qc": {"status": "PASS", "delivery_status": "DELIVERABLE_PENDING_HUMAN_WATCH_LISTEN",
                           "deliverable": True, "human_watch_listen_required": True, "render_attempt_count": 1},
        "stage_status": {"canonical_asr_evidence": "ok"},
    }
    proc, summary = _run(tmp_path, engine)
    assert proc.returncode == 0, proc.stderr
    assert "TAIL-SAFE EDITORIAL SUMMARY" in proc.stdout
    assert summary["source_key"] == "Yaskira/01.mp4"
    assert summary["selected"] == [
        {"clip_id": "a", "start": 0.4, "end": 57.9, "duration": 57.5, "text": "I was like, oh, I…"},
        {"clip_id": "b", "start": 59.2, "end": 114.5, "duration": 55.3, "text": "He is…"},
    ]
    assert summary["discarded"] == [{"clip_id": "c", "start": 57.9, "end": 59.2, "duration": 1.3, "text": "because…"}]
    assert summary["selected_total_sec"] == 112.8 and summary["discarded_total_sec"] == 1.3
    assert summary["asr_provider_audit_compact"]["provider"] == "faster-whisper-medium-whisperx"
    assert summary["asr_provider_audit_compact"]["word_count"] == 321
    assert summary["live_render_qc_compact"]["delivery_status"] == "DELIVERABLE_PENDING_HUMAN_WATCH_LISTEN"
    # The printed log carries the same record (that is the whole point).
    assert "because…" in proc.stdout and "57.9" in proc.stdout


def test_step_never_fails_without_engine_json(tmp_path):
    proc, summary = _run(tmp_path, None)
    assert proc.returncode == 0, proc.stderr
    assert summary == {"source_status": "engine_json_missing", "source_key": "Yaskira/01.mp4"}


def test_step_never_fails_on_malformed_engine_json(tmp_path):
    art = tmp_path / "artifact"; art.mkdir()
    (art / "video00-modal.json").write_text("{not json")
    proc = subprocess.run(["bash", "-c", _step_script()], cwd=str(tmp_path),
                          env={**os.environ, "SOURCE_KEY": "Yaskira/01.mp4"}, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert json.loads((art / "editorial-summary-tail.json").read_text())["selected_count"] == 0


def test_step_mentions_no_gold_labels():
    script = _step_script()
    for forbidden in ("0:58", "because", "Gold", "gold_delete", "HUMAN_GOLD"):
        assert forbidden not in script
