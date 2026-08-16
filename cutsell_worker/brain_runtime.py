"""Single source of truth for the CutSell Flow B brain runtime.

The perception/editing backbone remains RunPod-local. Paid semantic reasoning is an
optional layer and can activate only when CUTSELL_HYBRID_LLM_ENABLED is explicitly on,
the approved Google provider is selected, and GEMINI_API_KEY is present. Merely storing
a key never enables external calls.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping

from .clean_cut_provider import CleanCutProvider
from .composer_provider import ComposerProvider
from .config import RuntimeConfig
from .draft_review_provider import DraftReviewProvider
from .hybrid_editorial import EditorialJudge
from .hybrid_google_transport import DollarBudgetLedger, GoogleGeminiTransport
from .hybrid_provider import TransportEditorialJudge
from .hybrid_provider_settings import HybridProviderSettings, load_hybrid_provider_settings
from .hybrid_take_judge import HybridTakeJudgeProvider
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
    editorial_judge: EditorialJudge | None = None
    hybrid_settings: HybridProviderSettings = HybridProviderSettings()

    @property
    def external_calls_enabled(self) -> bool:
        return self.editorial_judge is not None


def _build_editorial_judge(
    settings: HybridProviderSettings,
    values: Mapping[str, str],
) -> EditorialJudge | None:
    """Construct the approved paid judge only after every explicit gate passes."""
    if not settings.enabled or settings.provider != "google":
        return None
    api_key = str(values.get("GEMINI_API_KEY") or "").strip()
    if not api_key:
        # Missing credentials degrade to the local brain instead of breaking a job.
        return None

    # One ledger is scoped to this runtime/job. It prevents all semantic calls made
    # while processing one edit from exceeding the production LLM COGS target.
    ledger = DollarBudgetLedger(settings.max_cost_per_edit_usd)
    transport = GoogleGeminiTransport(
        api_key=api_key,
        model=settings.primary_model,
        settings=settings,
        ledger=ledger,
        escalation=False,
    )
    return TransportEditorialJudge(
        provider_name="google",
        model_name=settings.primary_model,
        transport=transport,
    )


def build_brain_runtime(
    config: RuntimeConfig,
    env: Mapping[str, str] | None = None,
) -> BrainRuntime:
    """Build local perception plus explicitly-gated Hybrid Editorial reasoning.

    The paid judge is deliberately used by the bounded-group semantic cleanup stage.
    Best Take ranking retains a zero-cost HybridTakeJudgeProvider wrapper so a group is
    never charged twice for the same semantic decision.
    """
    if config.brain_backend != RUNPOD_LOCAL_BACKEND:
        raise RuntimeError(
            f"unsupported CUTSELL_BRAIN_BACKEND={config.brain_backend!r}; "
            f"only {RUNPOD_LOCAL_BACKEND!r} is permitted on this branch"
        )

    values: Mapping[str, str] = env if env is not None else os.environ
    hybrid_settings = load_hybrid_provider_settings(dict(values))
    editorial_judge = _build_editorial_judge(hybrid_settings, values)

    return BrainRuntime(
        backend=RUNPOD_LOCAL_BACKEND,
        semantic_provider=NoopSemanticProvider(),
        whole_video_provider=RunPodLocalWholeVideoProvider(),
        visual_provider=None,              # dense MediaPipe/OpenCV path in flow_b.py
        take_grouping_provider=None,       # deterministic retry grouping + session walls
        take_judge_provider=HybridTakeJudgeProvider(editorial_judge=None),
        clean_cut_provider=None,           # deterministic Clean Cut + contextual rules
        composer_provider=None,            # disabled until Sales Funnel reactivation
        draft_review_provider=None,        # disabled in Universal Clean Cut
        editorial_judge=editorial_judge,
        hybrid_settings=hybrid_settings,
    )
