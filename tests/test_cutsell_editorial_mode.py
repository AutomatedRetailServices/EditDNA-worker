from cutsell_worker.flow_b import _resolve_editorial_mode


def test_clean_cut_is_default(monkeypatch):
    monkeypatch.delenv("CUTSELL_EDITORIAL_MODE", raising=False)
    assert _resolve_editorial_mode() == "clean_cut"


def test_full_editorial_mode_remains_available(monkeypatch):
    monkeypatch.setenv("CUTSELL_EDITORIAL_MODE", "full")
    assert _resolve_editorial_mode() == "full"


def test_explicit_mode_overrides_environment(monkeypatch):
    monkeypatch.setenv("CUTSELL_EDITORIAL_MODE", "full")
    assert _resolve_editorial_mode("clean_cut") == "clean_cut"


def test_invalid_editorial_mode_is_rejected(monkeypatch):
    monkeypatch.delenv("CUTSELL_EDITORIAL_MODE", raising=False)
    try:
        _resolve_editorial_mode("sales_funnel")
    except ValueError as exc:
        assert "clean_cut or full" in str(exc)
    else:
        raise AssertionError("invalid editorial mode must fail")
