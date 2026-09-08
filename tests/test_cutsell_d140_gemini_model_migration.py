"""D-140: Gemini 3.6 Flash model migration (post D-139 retirement finding).

D-139 proved `gemini-2.5-flash` is retired for new users via a real Google
API 404 directing migration to `gemini-3.6-flash`. D-140 authorizes
exactly that migration target -- no other generation, no other provider.
These are the minimal targeted tests this task's own directive requires
for "exact Gemini 3.6 model/API compatibility" -- no editorial fixture
rewrite, no provider abstraction change.
"""
from __future__ import annotations

from cutsell_worker.hybrid_provider_settings import HybridProviderSettings
from cutsell_worker.multimodal_besttake_gemini import (
    REQUIRED_MODEL_ID,
    GeminiMultimodalBestTakeArbiter,
)


def test_required_model_id_migrated_to_gemini_3_6_flash():
    assert REQUIRED_MODEL_ID == "gemini-3.6-flash"


def test_required_model_id_no_longer_the_retired_generation():
    assert REQUIRED_MODEL_ID != "gemini-2.5-flash"


def test_arbiter_default_model_follows_the_migrated_constant():
    arbiter = GeminiMultimodalBestTakeArbiter(api_key="fake-key")
    assert arbiter.model == REQUIRED_MODEL_ID


def test_migration_target_matches_this_repos_own_preexisting_gemini_policy():
    # Independent corroboration (not a substitute for the real ListModels
    # check the eval workflow performs): this repo's own hybrid-brain
    # Gemini policy allowlist already defaults its escalation model to the
    # same id, confirming gemini-3.6-flash is a real, currently-served
    # model this codebase already relies on elsewhere -- not a guess.
    settings = HybridProviderSettings()
    assert settings.escalation_model == REQUIRED_MODEL_ID
