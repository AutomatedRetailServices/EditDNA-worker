from types import SimpleNamespace

import cutsell_worker.brain_runtime as br


def test_hybrid_editorial_judge_is_shared_with_best_take_provider(monkeypatch):
    sentinel_judge = object()
    settings = br.HybridProviderSettings(enabled=True, provider="google")

    monkeypatch.setattr(br, "load_hybrid_provider_settings", lambda values: settings)
    monkeypatch.setattr(br, "_build_editorial_judge", lambda loaded, values: sentinel_judge)

    runtime = br.build_brain_runtime(
        SimpleNamespace(brain_backend=br.RUNPOD_LOCAL_BACKEND),
        env={
            "CUTSELL_HYBRID_LLM_ENABLED": "1",
            "CUTSELL_HYBRID_PROVIDER": "google",
            "GEMINI_API_KEY": "test-key",
        },
    )

    assert runtime.editorial_judge is sentinel_judge
    assert runtime.take_judge_provider is not None
    assert runtime.take_judge_provider.editorial_judge is sentinel_judge
    assert runtime.external_calls_enabled is True
