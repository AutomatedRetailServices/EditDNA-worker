"""Manual entrypoint for the "CutSell Video00 Pod RAW" workflow's
health-only gate (D-042: CutSell QA GPU execution fallback -- RunPod Pod
On-Demand automation).

Reads configuration from environment variables (set by the workflow
step), runs `RunPodPodExecutionProvider.ensure_ready()` + `health_check()`,
ALWAYS tears the Pod down in `finally` regardless of outcome, and writes a
JSON summary the workflow uploads as an artifact and prints. This script
is deliberately health-only: per the standing directive, full Video00
benchmarking on a Pod is not run here and stays gated on separate,
explicit authorization -- see docs/CUTSELL_DECISIONS.md D-042.

Infrastructure/orchestration only -- no CutSell editorial logic lives
here; the canonical pipeline itself is invoked (when authorized) through
`cutsell_worker.serverless_handler.run_op`, unchanged either way.
"""
from __future__ import annotations

import json
import os
import sys

from runpod_orchestration import UrllibTransport, _default_log
from runpod_pod_provider import DEFAULT_COST_CEILING_USD_PER_HR, PodExecutionConfig, RunPodPodExecutionProvider


def main() -> int:
    api_key = os.environ["RUNPOD_API_KEY"]
    image = os.environ["POD_IMAGE"]
    existing_pod_id = os.environ.get("EXISTING_POD_ID") or None
    cost_ceiling = float(os.environ.get("QA_POD_COST_CEILING_USD_PER_HR", str(DEFAULT_COST_CEILING_USD_PER_HR)))
    start_command = os.environ.get("POD_START_COMMAND") or "python3 -m cutsell_worker.pod_job_server"

    config = PodExecutionConfig(
        api_key=api_key,
        image=image,
        pod_name=os.environ.get("POD_NAME", "cutsell-qa-pod"),
        start_command=start_command,
        cost_ceiling_usd_per_hr=cost_ceiling,
    )
    provider = RunPodPodExecutionProvider(
        UrllibTransport(),
        config,
        existing_pod_id=existing_pod_id,
        log=_default_log,
    )

    summary: dict = {"execution_provider": "RUNPOD_POD"}
    exit_code = 1
    try:
        result = provider.health_check()
        summary.update(
            {
                "passed": result.passed,
                "classification": result.classification,
                "elapsed_s": result.elapsed_s,
                "detail": result.detail,
            }
        )
        exit_code = 0 if result.passed else 1
    finally:
        # Guaranteed cleanup: this runs whether health_check passed,
        # failed, or raised. provider.teardown() itself never raises.
        provider.teardown()
        summary["pod_id"] = provider.pod_id
        with open("pod-health-summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
        print("--- pod-health-summary.json ---")
        print(json.dumps(summary, indent=2, sort_keys=True))

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
