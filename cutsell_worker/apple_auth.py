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

    signing_key = PyJWKClient(APPLE_JWKS_URL).get_signing_key_from_jwt(token)
    claims: dict[str, Any] = jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256"],
        audience=audience,
        issuer=APPLE_ISSUER,
        options={"require": ["exp", "iat", "iss", "aud", "sub"]},
    )
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
