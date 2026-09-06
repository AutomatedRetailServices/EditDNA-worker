import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CLEAN_WORKER_CI = REPO_ROOT / ".github/workflows/cutsell-clean-worker-ci.yml"
IOS_CI = REPO_ROOT / ".github/workflows/cutsell-ios-ci.yml"
VIDEO00_RAW = REPO_ROOT / ".github/workflows/cutsell-video00-raw-v5-auto-microtrim.yml"

WORKFLOWS = [CLEAN_WORKER_CI, IOS_CI, VIDEO00_RAW]

STEP_MARKER = "- name: Notify Claude Routine"

# Anchors that must precede the notify step in each workflow: the last piece
# of real work in the job, proving the step was appended at the end rather
# than spliced into the middle of the existing evidence-preservation flow.
LAST_WORK_STEP_ANCHOR = {
    CLEAN_WORKER_CI: "Boot API container and verify health endpoint",
    IOS_CI: "Build CutSell for iOS Simulator",
    VIDEO00_RAW: "RunPod teardown completed",
}

# Fields the wake-up payload is allowed to carry (CLAUDE.md-adjacent request:
# repository, workflow name, run id, run attempt, head branch, head sha,
# conclusion/status if available).
REQUIRED_PAYLOAD_FIELDS = [
    "repository",
    "workflow",
    "run_id",
    "run_attempt",
    "head_branch",
    "head_sha",
    "status",
]

# Substrings that must never appear inside the notify step: signs of secrets,
# logs, artifact contents, S3 URLs, or provider keys leaking into the payload.
FORBIDDEN_IN_NOTIFY_STEP = [
    "artifact/",
    "S3_BUCKET",
    "GEMINI_API_KEY",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_ACCESS_KEY_ID",
    "RUNPOD_API_KEY",
    "preview_uri",
    "result_uri",
    "aws s3",
    "cat /tmp",
    "cat artifact",
]


def _step_block(text: str) -> str:
    """Return just the 'Notify Claude Routine' step's YAML text."""
    start = text.index(STEP_MARKER)
    line_start = text.rfind("\n", 0, start) + 1
    indent = text[line_start:start]
    lines = text[line_start:].split("\n")
    block_lines = [lines[0]]
    next_step_prefix = f"{indent}- name:"
    for line in lines[1:]:
        if line.strip() == "":
            block_lines.append(line)
            continue
        if line.startswith(next_step_prefix) or (
            not line.startswith(indent + " ") and not line.startswith(indent + "-")
        ):
            break
        block_lines.append(line)
    return "\n".join(block_lines)


def test_all_three_workflows_declare_the_notify_step_exactly_once():
    for path in WORKFLOWS:
        text = path.read_text()
        assert text.count(STEP_MARKER) == 1, path


def test_notify_step_is_appended_after_the_existing_job_work():
    for path, anchor in LAST_WORK_STEP_ANCHOR.items():
        text = path.read_text()
        assert text.index(anchor) < text.index(STEP_MARKER), path


def test_notify_step_always_runs_regardless_of_job_outcome():
    for path in WORKFLOWS:
        block = _step_block(path.read_text())
        assert "if: always()" in block, path


def test_notify_step_uses_only_the_two_named_repository_secrets():
    secret_names = re.compile(r"secrets\.([A-Z0-9_]+)")
    for path in WORKFLOWS:
        block = _step_block(path.read_text())
        found = set(secret_names.findall(block))
        assert found == {"CUTSELL_CLAUDE_ROUTINE_URL", "CUTSELL_CLAUDE_ROUTINE_TOKEN"}, (path, found)


def test_notify_step_never_hardcodes_the_url_or_token():
    for path in WORKFLOWS:
        block = _step_block(path.read_text())
        # The only http(s) literal permitted anywhere in the block is the
        # GitHub Actions expression that reads the URL secret -- never a
        # bare literal endpoint.
        for line in block.splitlines():
            if "http://" in line or "https://" in line:
                assert "secrets.CUTSELL_CLAUDE_ROUTINE_URL" in line, (path, line)


def test_notify_step_sends_bearer_auth_header_built_from_the_token_secret():
    for path in WORKFLOWS:
        block = _step_block(path.read_text())
        assert "Authorization: Bearer $CLAUDE_ROUTINE_TOKEN" in block, path
        assert "CLAUDE_ROUTINE_TOKEN: ${{ secrets.CUTSELL_CLAUDE_ROUTINE_TOKEN }}" in block, path


def test_notify_step_posts_to_the_url_secret_only():
    for path in WORKFLOWS:
        block = _step_block(path.read_text())
        assert '--url "$CLAUDE_ROUTINE_URL"' in block, path
        assert "CLAUDE_ROUTINE_URL: ${{ secrets.CUTSELL_CLAUDE_ROUTINE_URL }}" in block, path


def test_notify_step_payload_contains_only_small_safe_run_identity_fields():
    for path in WORKFLOWS:
        block = _step_block(path.read_text())
        for field in REQUIRED_PAYLOAD_FIELDS:
            assert f"{field}:" in block or f'"{field}"' in block, (path, field)
        for bad in FORBIDDEN_IN_NOTIFY_STEP:
            assert bad not in block, (path, bad)


def test_notify_step_uses_job_status_context_for_conclusion():
    for path in WORKFLOWS:
        block = _step_block(path.read_text())
        assert "WORKFLOW_STATUS: ${{ job.status }}" in block, path


def test_notify_step_never_fails_the_job_it_runs_in():
    for path in WORKFLOWS:
        block = _step_block(path.read_text())
        assert "set +e" in block, path
        assert block.rstrip().splitlines()[-1].strip() == "exit 0", path


def test_notify_step_skips_cleanly_when_url_secret_is_unset():
    for path in WORKFLOWS:
        block = _step_block(path.read_text())
        assert 'if [ -z "${CLAUDE_ROUTINE_URL:-}" ]' in block, path


def test_notify_step_is_the_final_step_in_its_job():
    for path in WORKFLOWS:
        text = path.read_text()
        last_step_index = text.rindex("- name:")
        assert text.index(STEP_MARKER) == last_step_index, path
