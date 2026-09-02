"""Modal GPU minimal smoke test (D-043: CutSell Modal GPU execution --
first live validation). Deliberately minimal and controlled: this proves
Modal can execute a bounded, serverless GPU function for this account on
an approved GPU type (L4 only, per the standing cost-safety directive) and
scale back to zero automatically -- it does NOT run Video00 or any
CutSell editorial code. See modal_gpu_diagnostics.py for the actual
diagnostic logic (kept modal-package-free and independently testable).

Image/runtime strategy (D-043 audit): reuses
`madiator2011/better-pytorch:cuda12.4-torch2.6.0` verbatim -- the EXACT
same base image `Dockerfile.cutsell.serverless` builds FROM -- via
`modal.Image.from_registry`, plus the same `apt-get install ffmpeg`
Dockerfile step. This is a deliberate choice to avoid silently diverging
torch/CUDA/ffmpeg versions from the production RunPod image (Option A:
same base + same runtime-level installs, not a different Modal-specific
environment). The full `requirements.cutsell.worker.txt` dependency set
and `cutsell_worker` package are NOT installed here -- not needed for a
torch/CUDA/ffmpeg-only smoke test, and installing them is deferred to
whenever full Video00-on-Modal integration is separately authorized.

Invocation (this phase): the officially documented, stable `modal run`
CLI entrypoint (`modal run modal_gpu_minimal_test.py`), authenticated via
the `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` environment variables Modal's
own client reads automatically -- not Modal's Python SDK invoked
in-process from `gpu_execution_provider.py`. `ModalExecutionProvider` in
that module establishes the future in-process-invocation architecture;
it is intentionally not what this one-off smoke test uses, so this test
does not depend on guessing the exact current SDK call shape
(`Function.lookup` vs `Function.from_name` etc. have changed across Modal
SDK versions) beyond the CLI's own stable contract.

Cost safety: exactly one approved GPU type (L4), a bounded timeout (see
modal_gpu_config.DEFAULT_MODAL_TIMEOUT_S), and no explicit
`scaledown_window` override -- Modal's own default scale-to-zero behavior
already satisfies "no idle container remains" for a single ephemeral
`modal run` invocation, and omitting it avoids a possible
container_idle_timeout/scaledown_window kwarg-name mismatch across SDK
versions on this first live attempt.
"""
from __future__ import annotations

import json

import modal

from modal_gpu_config import (
    CUTSELL_BASE_IMAGE,
    DEFAULT_MODAL_TIMEOUT_S,
)
from modal_gpu_config import require_modal_gpu_type, require_modal_timeout
from modal_gpu_diagnostics import collect_gpu_diagnostics

MODAL_GPU_TYPE = "L4"

# Fail fast at import/deploy time, not silently, if this file's own
# constants ever drift from the approved pool -- the same
# "validate, don't just trust the literal" discipline as every other
# D-042/D-043 config module.
require_modal_gpu_type(MODAL_GPU_TYPE)
require_modal_timeout(DEFAULT_MODAL_TIMEOUT_S)

app = modal.App("cutsell-gpu-minimal-isolation")

image = modal.Image.from_registry(CUTSELL_BASE_IMAGE).apt_install("ffmpeg")


@app.function(gpu=MODAL_GPU_TYPE, image=image, timeout=DEFAULT_MODAL_TIMEOUT_S)
def run_minimal_gpu_check() -> dict:
    return collect_gpu_diagnostics()


@app.local_entrypoint()
def main() -> None:
    result = run_minimal_gpu_check.remote()
    print(json.dumps(result, indent=2, default=str))
