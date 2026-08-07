"""ASGI auth middleware that binds mobile requests to their bearer-session user."""
from __future__ import annotations

import json
import os
from urllib.parse import parse_qs

from cutsell_worker.auth import resolve_session

PUBLIC_PATHS = {"/v1/healthz", "/v1/auth/session"}
MAX_JSON_BODY = 8 * 1024 * 1024


async def _respond(send, status: int, detail: str):
    body = json.dumps({"detail": detail}, separators=(",", ":")).encode()
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
    })
    await send({"type": "http.response.body", "body": body})


class AuthScopeMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)
        path = str(scope.get("path") or "")
        required = str(os.environ.get("CUTSELL_AUTH_REQUIRED", "0")).strip().lower() in {"1", "true", "yes", "on"}
        if not required or path in PUBLIC_PATHS or scope.get("method") == "OPTIONS":
            return await self.app(scope, receive, send)

        headers = {bytes(k).lower(): bytes(v) for k, v in scope.get("headers") or ()}
        auth = headers.get(b"authorization", b"").decode("utf-8", errors="ignore")
        if not auth.lower().startswith("bearer "):
            return await _respond(send, 401, "bearer token required")
        token = auth.split(" ", 1)[1].strip()
        try:
            session = resolve_session(token)
        except PermissionError as exc:
            return await _respond(send, 401, str(exc))
        except RuntimeError:
            return await _respond(send, 503, "auth service unavailable")
        auth_user_id = str(session["user_id"])

        claimed_query = (parse_qs((scope.get("query_string") or b"").decode()).get("user_id") or [None])[0]
        if claimed_query is not None and str(claimed_query) != auth_user_id:
            return await _respond(send, 403, "user scope mismatch")

        messages = []
        body = b""
        more = True
        while more:
            message = await receive()
            messages.append(message)
            if message.get("type") == "http.request":
                body += message.get("body", b"")
                more = bool(message.get("more_body"))
                if len(body) > MAX_JSON_BODY:
                    return await _respond(send, 413, "request body too large")
            else:
                more = False

        content_type = headers.get(b"content-type", b"").decode("utf-8", errors="ignore").lower()
        if body and "application/json" in content_type:
            try:
                payload = json.loads(body)
            except Exception:
                payload = None
            if isinstance(payload, dict) and payload.get("user_id") is not None:
                if str(payload.get("user_id")) != auth_user_id:
                    return await _respond(send, 403, "user scope mismatch")

        state = scope.setdefault("state", {})
        state["auth_user_id"] = auth_user_id
        index = 0

        async def replay_receive():
            nonlocal index
            if index < len(messages):
                message = messages[index]
                index += 1
                return message
            return {"type": "http.request", "body": b"", "more_body": False}

        return await self.app(scope, replay_receive, send)
