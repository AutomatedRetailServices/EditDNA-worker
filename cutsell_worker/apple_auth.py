"""Server-side Sign in with Apple identity-token verification.

Activation requires CUTSELL_APPLE_CLIENT_ID. The mobile app still needs Apple
Developer entitlement/signing before this endpoint can be used by real testers.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

APPLE_ISSUER = "https://appleid.apple.com"
APPLE_JWKS_URL = "https://appleid.apple.com/auth/keys"


@dataclass(frozen=True)
class AppleIdentity:
    subject: str
    email: str | None
    email_verified: bool | None


def verify_apple_identity_token(
    identity_token: str,
    *,
    nonce: str | None = None,
    client_id: str | None = None,
) -> AppleIdentity:
    token = str(identity_token or "").strip()
    if not token:
        raise PermissionError("missing Apple identity token")
    audience = str(client_id or os.getenv("CUTSELL_APPLE_CLIENT_ID") or "").strip()
    if not audience:
        raise RuntimeError("CUTSELL_APPLE_CLIENT_ID is not configured")

    import jwt
    from jwt import PyJWKClient

    from jwt.exceptions import PyJWKClientConnectionError, PyJWTError
    try:
        signing_key = PyJWKClient(APPLE_JWKS_URL).get_signing_key_from_jwt(token)
        claims: dict[str, Any] = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=audience,
            issuer=APPLE_ISSUER,
            options={"require": ["exp", "iat", "iss", "aud", "sub"]},
        )
    except PyJWKClientConnectionError as exc:      # Apple's keys unreachable: "try again" (503)
        raise RuntimeError("Apple sign-in keys unreachable") from exc
    except PyJWTError as exc:                        # a bad, expired or foreign token: rejected (401)
        raise PermissionError(f"Apple identity token rejected: {exc.__class__.__name__}") from exc
    if nonce is not None and str(claims.get("nonce") or "") != str(nonce):
        raise PermissionError("Apple nonce mismatch")
    subject = str(claims.get("sub") or "").strip()
    if not subject:
        raise PermissionError("Apple subject is missing")
    verified_raw = claims.get("email_verified")
    if isinstance(verified_raw, str):
        email_verified = verified_raw.lower() == "true"
    elif isinstance(verified_raw, bool):
        email_verified = verified_raw
    else:
        email_verified = None
    email = str(claims.get("email") or "").strip() or None
    return AppleIdentity(subject=subject, email=email, email_verified=email_verified)


# --- Token revoke on account deletion (Apple App Review 5.1.1(v), PO 6 Nov 2026) --------------------------------
# At sign-in the app sends the one-time authorization code; the server swaps it for Apple's refresh token and keeps
# it only to revoke it when the creator deletes her account. Needs a "Sign in with Apple" key from the Apple
# Developer account: CUTSELL_APPLE_TEAM_ID, CUTSELL_APPLE_KEY_ID, CUTSELL_APPLE_PRIVATE_KEY (the .p8 text).
# Without them every step below is skipped and says "not_configured"; sign-in itself is unchanged.

APPLE_TOKEN_URL = "https://appleid.apple.com/auth/token"
APPLE_REVOKE_URL = "https://appleid.apple.com/auth/revoke"


def revoke_configured() -> bool:
    return all(str(os.getenv(name) or "").strip() for name in
               ("CUTSELL_APPLE_CLIENT_ID", "CUTSELL_APPLE_TEAM_ID", "CUTSELL_APPLE_KEY_ID", "CUTSELL_APPLE_PRIVATE_KEY"))


def _client_secret() -> str:
    import time

    import jwt
    now = int(time.time())
    key = os.environ["CUTSELL_APPLE_PRIVATE_KEY"].replace("\\n", "\n")
    return jwt.encode({"iss": os.environ["CUTSELL_APPLE_TEAM_ID"], "iat": now, "exp": now + 300,
                       "aud": APPLE_ISSUER, "sub": os.environ["CUTSELL_APPLE_CLIENT_ID"]},
                      key, algorithm="ES256", headers={"kid": os.environ["CUTSELL_APPLE_KEY_ID"]})


def exchange_authorization_code(code: str, *, post=None) -> str | None:
    """The refresh token for this sign-in, or None (not configured, no code, or Apple said no)."""
    if not revoke_configured() or not str(code or "").strip():
        return None
    import requests
    post = post or requests.post
    response = post(APPLE_TOKEN_URL, data={"client_id": os.environ["CUTSELL_APPLE_CLIENT_ID"],
                                           "client_secret": _client_secret(), "code": code,
                                           "grant_type": "authorization_code"}, timeout=15)
    if getattr(response, "status_code", 0) != 200:
        return None
    return str((response.json() or {}).get("refresh_token") or "") or None


def revoke_refresh_token(token: str, *, post=None) -> str:
    if not revoke_configured():
        return "not_configured"
    if not token:
        return "no_token"
    import requests
    post = post or requests.post
    response = post(APPLE_REVOKE_URL, data={"client_id": os.environ["CUTSELL_APPLE_CLIENT_ID"],
                                            "client_secret": _client_secret(), "token": token,
                                            "token_type_hint": "refresh_token"}, timeout=15)
    return "revoked" if getattr(response, "status_code", 0) == 200 else f"failed_{getattr(response, 'status_code', 0)}"
