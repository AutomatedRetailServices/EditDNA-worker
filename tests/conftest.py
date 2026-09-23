"""D-269 Stage 1/32: this test suite's own explicit, bounded local/test
override for CutSell's fail-closed auth default.

`cutsell_app.auth_middleware.AuthScopeMiddleware` now requires a bearer
token and enforced ownership by DEFAULT (D-269 closed D-268's own P0
finding that a missing/mistyped `CUTSELL_AUTH_REQUIRED` silently disabled
the entire tenant-isolation layer). The pre-existing test suite has a
long-standing, legitimate convention of exercising `cutsell_app.main.app`
directly via `TestClient` without ever attaching a bearer token -- this
file is the ONE explicit, named place that keeps that convention working,
by setting the auth middleware's own dedicated local/test-only disable
flag for the whole pytest session.

Production and any real deployment never import this file, so the
fail-closed default in `auth_middleware.py` is never affected by it.
Individual tests that specifically exercise auth-REQUIRED behavior (see
`test_cutsell_clean_worker_auth.py`) set `CUTSELL_AUTH_REQUIRED=1` via
`monkeypatch`, which always wins over this file's own default -- see
`auth_middleware._auth_required`'s own precedence rule.
"""
import os

os.environ.setdefault("CUTSELL_AUTH_DISABLED_FOR_LOCAL_DEV_ONLY", "1")
