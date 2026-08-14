"""Single source of truth for the CutSell brain runtime.

The Mobile V1 / Universal Clean Cut brain is RunPod-local by design. Presence of an
OPENAI_API_KEY must never enable external model calls implicitly. All heavy analysis
runs inside the worker: Faster-Whisper, dense MediaPipe/OpenCV performance evidence,
deterministic Clean Cut, retry grouping, Best Take ranking, temporal cleanup, and
rendering.
"""
from __future__ import annotations

from dataclasses import dataclass

from .clean_cut_provider import CleanCutProvider
from .composer_provider import ComposerProvider
from .config import RuntimeConfig
from .draft_review_provider import DraftReviewProvider
from .providers import NoopSemanticProvider, SemanticProvider
from .take_grouping_provider import TakeGroupingProvider
from .take_judge_provider import TakeJudgeProvider
from .visual_analysis import VisualProvider
from .whole_video_analysis import WholeVideoProvider
from .whole_video_local import RunPodLocalWholeVideoProvider


RUNPOD_LOCAL_BACKEND = "runpod_local"


@dataclass(frozen=True)
class BrainRuntime:
    backend: str
    semantic_provider: SemanticProvider
    whole_video_provider: WholeVideoProvider
    visual_provider: VisualProvider | None
    take_grouping_provider: TakeGroupingProvider | None
    take_judge_provider: TakeJudgeProvider | None
    clean_cut_provider: CleanCutProvider | None
    composer_provider: ComposerProvider | None
    draft_review_provider: DraftReviewProvider | None

    @property
    def external_calls_enabled(self) -> bool:
        return False


def build_brain_runtime(config: RuntimeConfig) -> BrainRuntime:
    """Return the production/benchmark brain. External providers are not permitted.

    Provider ``None`` is intentional for stages that already have conservative local
    baselines in the core pipeline. Those stages remain fully executed inside RunPod;
    ``None`` means "use deterministic local implementation", not "skip the brain".
    """
    if config.brain_backend != RUNPOD_LOCAL_BACKEND:
        raise RuntimeError(
            f"unsupported CUTSELL_BRAIN_BACKEND={config.brain_backend!r}; "
            f"only {RUNPOD_LOCAL_BACKEND!r} is permitted on this branch"
        )

    return BrainRuntime(
        backend=RUNPOD_LOCAL_BACKEND,
        semantic_provider=NoopSemanticProvider(),
        whole_video_provider=RunPodLocalWholeVideoProvider(),
        visual_provider=None,              # dense MediaPipe/OpenCV path in flow_b.py
        take_grouping_provider=None,       # deterministic retry grouping baseline
        take_judge_provider=None,          # deterministic Best Take baseline
        clean_cut_provider=None,           # deterministic Clean Cut + contextual rules
        composer_provider=None,            # disabled in Universal Clean Cut
        draft_review_provider=None,        # disabled in Universal Clean Cut
    )
