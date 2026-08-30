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
from .semantic_idea_equivalence import SemanticEquivalenceArbiter
from .semantic_idea_equivalence_google import GoogleSemanticEquivalenceArbiter
from .take_grouping_provider import TakeGroupingProvider
from .take_judge_provider import TakeJudgeProvider
from .unified_selection_google import GoogleUnifiedSelectionReasoner
from .unified_selection_reasoner import UnifiedSelectionReasoner
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
    selection_reasoner: UnifiedSelectionReasoner | None = None
    hybrid_settings: HybridProviderSettings = HybridProviderSettings()
    # Architecture rebalance Phase 0/1 rollback flag: set
    # CUTSELL_DETERMINISTIC_BEST_TAKE_AUTHORITY=0 to restore the previous
    # pure-whole-video-reasoner behavior (Unified Selection with
    # unconditional final say) unmodified during migration.
    deterministic_best_take_authority_enabled: bool = True
    # Phase 2: narrow gated semantic-equivalence arbiter used by take
    # grouping to strengthen retry-family detection before Best Take/Unified
    # Selection ever run. Independent of selection_reasoner -- it applies to
    # the legacy grouping path too, since take grouping happens upstream of
    # that branch. None whenever paid inference is off, disabled via the
    # rollback flag below, or no API key is present.
    semantic_equivalence_arbiter: SemanticEquivalenceArbiter | None = None
    # Clean Cut Core V1 (see CLAUDE.md / docs/CUTSELL_DECISIONS.md): the
    # idea-first deterministic pipeline is the default active Clean Cut path.
    # Set CUTSELL_CLEAN_CUT_CORE_V1=0 to roll back to the pre-V1 architecture
    # (whole-video Unified Selection reasoner when explicitly requested, else
    # the legacy Hybrid-vote-informed path) for comparison/regression testing.
    clean_cut_core_v1_enabled: bool = True

    @property
    def external_calls_enabled(self) -> bool:
        return self.editorial_judge is not None or self.selection_reasoner is not None


def _env_true(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _env_true_default_true(value: str | None) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _require_google_api_key(settings: HybridProviderSettings, values: Mapping[str, str]) -> str:
    if not settings.enabled or settings.provider != "google":
        return ""
    api_key = str(values.get("GEMINI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "CUTSELL_HYBRID_LLM_ENABLED=1 requires GEMINI_API_KEY; refusing silent local fallback"
        )
    return api_key


def _build_editorial_judge(
    settings: HybridProviderSettings,
    values: Mapping[str, str],
) -> EditorialJudge | None:
    """Construct the legacy bounded paid judge after every explicit gate passes."""
    if not settings.enabled or settings.provider != "google":
        return None
    api_key = _require_google_api_key(settings, values)
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


def _build_unified_selection_reasoner(
    settings: HybridProviderSettings,
    values: Mapping[str, str],
) -> UnifiedSelectionReasoner | None:
    if not settings.enabled or settings.provider != "google":
        return None
    api_key = _require_google_api_key(settings, values)
    return GoogleUnifiedSelectionReasoner(
        api_key=api_key,
        model=settings.primary_model,
        settings=settings,
        # Deliberately NOT max_cost_per_edit_usd -- that is the legacy
        # per-group Hybrid judge's COGS target, sized for many small calls.
        # Unified Selection's one whole-video call has a different cost
        # shape and its own ceiling; see hybrid_provider_settings.py.
        ledger=DollarBudgetLedger(settings.max_cost_per_unified_selection_call_usd),
    )


def _build_semantic_equivalence_arbiter(
    settings: HybridProviderSettings,
    values: Mapping[str, str],
) -> SemanticEquivalenceArbiter | None:
    if not settings.enabled or settings.provider != "google":
        return None
    api_key = _require_google_api_key(settings, values)
    return GoogleSemanticEquivalenceArbiter(
        api_key=api_key,
        model=settings.primary_model,
        settings=settings,
        # Deliberately its own ceiling -- see hybrid_provider_settings.py's
        # max_cost_per_semantic_equivalence_call_usd docstring. Neither the
        # legacy per-group Hybrid ceiling nor Unified Selection's whole-video
        # ceiling is sized for this call shape.
        ledger=DollarBudgetLedger(settings.max_cost_per_semantic_equivalence_call_usd),
    )


def build_brain_runtime(
    config: RuntimeConfig,
    env: Mapping[str, str] | None = None,
) -> BrainRuntime:
    """Build local perception plus one explicitly-gated semantic Selection authority."""
    if config.brain_backend != RUNPOD_LOCAL_BACKEND:
        raise RuntimeError(
            f"unsupported CUTSELL_BRAIN_BACKEND={config.brain_backend!r}; "
            f"only {RUNPOD_LOCAL_BACKEND!r} is permitted on this branch"
        )

    values: Mapping[str, str] = env if env is not None else os.environ
    requested_hybrid = _env_true(values.get("CUTSELL_HYBRID_LLM_ENABLED"))
    requested_unified = _env_true(values.get("CUTSELL_UNIFIED_SELECTION_REASONER"))
    requested_provider = str(values.get("CUTSELL_HYBRID_PROVIDER") or "google").strip().lower()
    if requested_hybrid and requested_provider != "google":
        raise RuntimeError(
            "CUTSELL_HYBRID_LLM_ENABLED=1 requires CUTSELL_HYBRID_PROVIDER=google; "
            f"got {requested_provider!r}. Refusing silent local fallback"
        )
    if requested_unified and not requested_hybrid:
        raise RuntimeError(
            "CUTSELL_UNIFIED_SELECTION_REASONER=1 requires CUTSELL_HYBRID_LLM_ENABLED=1"
        )

    hybrid_settings = load_hybrid_provider_settings(dict(values))
    selection_reasoner = (
        _build_unified_selection_reasoner(hybrid_settings, values)
        if requested_unified
        else None
    )
    # The pivot deliberately avoids two semantic brains fighting over the same edit.
    # When Unified Selection is active, legacy bounded Hybrid classification is OFF and
    # its former outputs remain only historical/local evidence already present in tests.
    editorial_judge = None if selection_reasoner is not None else _build_editorial_judge(hybrid_settings, values)
    deterministic_best_take_authority_enabled = _env_true_default_true(
        values.get("CUTSELL_DETERMINISTIC_BEST_TAKE_AUTHORITY")
    )
    # Phase 2 rollback flag: set CUTSELL_SEMANTIC_EQUIVALENCE_ARBITER=0 to
    # disable the semantic-equivalence arbiter without touching
    # CUTSELL_HYBRID_LLM_ENABLED (which also gates the legacy judge and
    # Unified Selection). Gated on requested_hybrid alone, not
    # requested_unified -- take grouping runs upstream of the Unified
    # Selection/legacy branch and benefits from this either way.
    semantic_equivalence_arbiter_enabled = _env_true_default_true(
        values.get("CUTSELL_SEMANTIC_EQUIVALENCE_ARBITER")
    )
    semantic_equivalence_arbiter = (
        _build_semantic_equivalence_arbiter(hybrid_settings, values)
        if requested_hybrid and semantic_equivalence_arbiter_enabled
        else None
    )
    clean_cut_core_v1_enabled = _env_true_default_true(values.get("CUTSELL_CLEAN_CUT_CORE_V1"))

    return BrainRuntime(
        backend=RUNPOD_LOCAL_BACKEND,
        semantic_provider=NoopSemanticProvider(),
        whole_video_provider=RunPodLocalWholeVideoProvider(),
        visual_provider=None,
        take_grouping_provider=None,
        take_judge_provider=HybridTakeJudgeProvider(editorial_judge=None),
        clean_cut_provider=None,
        composer_provider=None,
        draft_review_provider=None,
        editorial_judge=editorial_judge,
        selection_reasoner=selection_reasoner,
        hybrid_settings=hybrid_settings,
        deterministic_best_take_authority_enabled=deterministic_best_take_authority_enabled,
        semantic_equivalence_arbiter=semantic_equivalence_arbiter,
        clean_cut_core_v1_enabled=clean_cut_core_v1_enabled,
    )
