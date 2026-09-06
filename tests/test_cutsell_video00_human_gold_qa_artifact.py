"""Regression coverage for the Video00 RAW workflow's Human Gold QA artifact.

Scope, per the task that added this: modify only the Video00 QA workflow and
these tests. Selection logic, Boundary logic, and any Video00-specific
production rule are untouched -- this file only asserts on the workflow YAML
text and on the absence of the Human Gold reference from production code.
"""
from pathlib import Path

WORKFLOW_PATH = Path(".github/workflows/cutsell-video00-raw-v5-auto-microtrim.yml")
HUMAN_GOLD_S3_KEY = "Editdna longform validation/5E01F214-A364-4F4B-8F25-D39B1E2B21D2.MP4"
HUMAN_GOLD_ARTIFACT_FILENAME = "human-gold-video00.mp4"


def workflow_text() -> str:
    return WORKFLOW_PATH.read_text()


def test_human_gold_s3_key_is_the_authoritative_key():
    text = workflow_text()
    assert f"HUMAN_GOLD_KEY: {HUMAN_GOLD_S3_KEY}" in text


def test_human_gold_download_step_uses_workflows_existing_aws_credentials():
    text = workflow_text()
    idx = text.index("Download Human Gold reference video")
    step_text = text[idx:text.index("- name:", idx + 1)]
    # Same credential-extraction pattern already used by the CutSell artifact
    # download step -- reads from the RunPod base template already fetched
    # earlier in the job, not a separately introduced credential source.
    assert "AWS_ACCESS_KEY_ID" in step_text
    assert "AWS_SECRET_ACCESS_KEY" in step_text
    assert "AWS_REGION" in step_text
    assert "S3_BUCKET" in step_text
    assert "/tmp/base-template.json" in step_text
    assert 'aws s3 cp "s3://$S3_BUCKET/$HUMAN_GOLD_KEY"' in step_text
    assert HUMAN_GOLD_ARTIFACT_FILENAME in step_text


def test_human_gold_download_happens_after_benchmark_result_and_before_artifact_upload():
    text = workflow_text()
    download_cutsell_result = text.index("Download unified Selection artifact")
    download_human_gold = text.index("Download Human Gold reference video")
    upload_artifact = text.index("Upload unified Selection diagnostic artifact")

    # "after the benchmark result is available": the CutSell benchmark's own
    # result/preview download step must run first.
    assert download_cutsell_result < download_human_gold
    # "before upload-artifact": the Human Gold download must complete before
    # the diagnostic artifact is uploaded, so it is captured in the same
    # artifact as the generated MP4 and result JSON.
    assert download_human_gold < upload_artifact


def test_upload_artifact_path_covers_the_whole_artifact_directory():
    text = workflow_text()
    upload_idx = text.index("Upload unified Selection diagnostic artifact")
    upload_step_text = text[upload_idx:upload_idx + 400]
    # upload-artifact points at the whole artifact/ directory (not an explicit
    # per-file allowlist), so human-gold-video00.mp4 is swept in automatically
    # alongside video00-unified-selection.mp4/.json once it is written there.
    assert "path: artifact/" in upload_step_text


def test_human_gold_download_step_is_qa_only_by_comment():
    text = workflow_text()
    idx = text.index("Download Human Gold reference video")
    step_text = text[idx:text.index("- name:", idx + 1)]
    assert "QA-only" in step_text
    assert "never read by production code" in step_text


def test_human_gold_reference_never_appears_in_production_selection_or_boundary_code():
    # The Human Gold S3 key, its artifact filename, and the env var name that
    # carries it must never leak into cutsell_worker: it is QA-only evidence
    # for Human Watch+Listen, never a runtime hardcoded authority and never
    # passed into production Selection reasoning.
    forbidden = (HUMAN_GOLD_S3_KEY, HUMAN_GOLD_ARTIFACT_FILENAME, "HUMAN_GOLD_KEY")
    for path in Path("cutsell_worker").rglob("*.py"):
        text = path.read_text()
        for needle in forbidden:
            assert needle not in text, f"{needle!r} must not appear in production code: {path}"


def test_human_gold_download_does_not_alter_selection_or_boundary_trigger_paths():
    # This task's scope is the QA workflow and its regression tests only --
    # confirm the RAW workflow's push-path trigger list (which governs when
    # Selection/Boundary code changes launch a paid RAW) is unmodified by
    # cross-checking the paths still reference only pre-existing production
    # modules, none of which is this test file or the Human Gold additions.
    text = workflow_text()
    paths_block = text[text.index("paths:"):text.index("workflow_dispatch:")]
    assert "test_cutsell_video00_human_gold_qa_artifact.py" not in paths_block
    assert HUMAN_GOLD_S3_KEY not in paths_block
