"""23 Oct (PO): Apple push notifications, so "your videos are ready" reaches the iPhone even when CutSell was closed.

- The app registers its device token (POST /v1/push/register) with its language (en / es).
- The worker sends: a Single / Combine cut ready or failed (one push), a Bulk send ONCE when every video of it is done
  ("Your 10 videos are ready ✨" / "9 ready, 1 needs attention"), a Mix once.
- Configuration (Modal secret `cutsell-worker`, set by the Product Owner, never in code or chat):
  APNS_KEY_P8 (the .p8 file's text), APNS_KEY_ID, APNS_TEAM_ID; optional APNS_TOPIC (default ai.cutsell.app).
  Without them nothing is sent and the app keeps its own local notifications (push_active = false).
- Never raises: a push can never fail or change a cut.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping
from typing import Callable, Optional

from .config import load_runtime_config

TOPIC_DEFAULT = "ai.cutsell.app"
HOSTS = {"production": "https://api.push.apple.com", "sandbox": "https://api.sandbox.push.apple.com"}
CONFIGURED_KEY = "cutsell:push:configured"          # set by a worker that has the key (the API reports it to the app)
GROUP_TTL = 3 * 86_400
TOKEN_TTL = 120 * 86_400

_JWT: dict = {}


def _scope(user_id: str) -> str:
    return hashlib.sha256(str(user_id).encode()).hexdigest()[:20]


def tokens_key(user_id: str) -> str:
    return f"cutsell:push:tokens:{_scope(user_id)}"


def _redis(client=None):
    if client is not None:
        return client
    from redis import Redis
    url = load_runtime_config().redis_url
    if not url:
        raise RuntimeError("REDIS_URL is not configured")
    return Redis.from_url(url)


def _key_text() -> str:
    raw = os.environ.get("APNS_KEY_P8", "").strip()
    return raw.replace("\\n", "\n")


def configured() -> bool:
    return bool(_key_text() and os.environ.get("APNS_KEY_ID", "").strip() and os.environ.get("APNS_TEAM_ID", "").strip())


def mark_configured(client=None) -> None:
    """A worker that has the key says so (the app then leaves the "ready" notifications to the server)."""
    if not configured():
        return
    try:
        _redis(client).set(CONFIGURED_KEY, "1", ex=7 * 86_400)
    except Exception:
        pass


def active(client=None) -> bool:
    try:
        return bool(_redis(client).get(CONFIGURED_KEY))
    except Exception:
        return False


# MARK: device tokens

def register(*, user_id: str, token: str, language: str = "en", environment: str = "production", client=None) -> None:
    token = "".join(ch for ch in str(token) if ch in "0123456789abcdefABCDEF").lower()
    if not user_id or not (32 <= len(token) <= 200):
        raise ValueError("invalid device token")
    env = "sandbox" if str(environment).lower() in ("sandbox", "development") else "production"
    lang = "es" if str(language).lower().startswith("es") else "en"
    r = _redis(client)
    key = tokens_key(user_id)
    r.hset(key, token, json.dumps({"lang": lang, "env": env, "at": int(time.time())}))
    r.expire(key, TOKEN_TTL)


def unregister(*, user_id: str, token: str, client=None) -> None:
    try:
        _redis(client).hdel(tokens_key(user_id), str(token).lower())
    except Exception:
        pass


def _tokens(user_id: str, client=None) -> dict:
    out = {}
    for token, raw in (_redis(client).hgetall(tokens_key(user_id)) or {}).items():
        token = token.decode() if isinstance(token, bytes) else token
        try:
            out[token] = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        except Exception:
            out[token] = {"lang": "en", "env": "production"}
    return out


# MARK: sending

def _jwt(now: Callable[[], float] = time.time) -> str:
    """Provider token (ES256), reused up to 50 minutes (Apple allows 60)."""
    cached = _JWT.get("token")
    if cached and now() - _JWT.get("at", 0) < 50 * 60:
        return cached
    import jwt
    issued = int(now())
    token = jwt.encode({"iss": os.environ["APNS_TEAM_ID"].strip(), "iat": issued}, _key_text(), algorithm="ES256",
                       headers={"kid": os.environ["APNS_KEY_ID"].strip()})
    _JWT.update(token=token, at=issued)
    return token


def _default_post(url: str, headers: dict, body: bytes):
    import httpx
    with httpx.Client(http2=True, timeout=15) as http:
        response = http.post(url, headers=headers, content=body)
        return response.status_code, response.text


def send(user_id: str, texts: Callable[[str], tuple], *, collapse_id: str, data: Optional[dict] = None,
         client=None, post=None) -> dict:
    """One alert to every iPhone of the user. `texts(lang)` -> (title, body). Never raises."""
    if not configured():
        return {"sent": 0, "reason": "not_configured"}
    try:
        tokens = _tokens(user_id, client)
    except Exception as exc:
        return {"sent": 0, "reason": "tokens_" + exc.__class__.__name__}
    if not tokens:
        return {"sent": 0, "reason": "no_device"}
    post = post or _default_post
    sent, removed, errors = 0, 0, []
    for token, info in tokens.items():
        title, body = texts(info.get("lang", "en"))
        payload = {"aps": {"alert": {"title": title, "body": body}, "sound": "default"}, **(data or {})}
        headers = {
            "authorization": "bearer " + _jwt(),
            "apns-topic": os.environ.get("APNS_TOPIC", "").strip() or TOPIC_DEFAULT,
            "apns-push-type": "alert",
            "apns-priority": "10",
            "apns-collapse-id": collapse_id[:64],
        }
        url = f"{HOSTS.get(info.get('env'), HOSTS['production'])}/3/device/{token}"
        try:
            status, text = post(url, headers, json.dumps(payload).encode())
        except Exception as exc:
            errors.append(exc.__class__.__name__)
            continue
        if status == 200:
            sent += 1
        elif status == 410 or (status == 400 and "BadDeviceToken" in str(text)):
            unregister(user_id=user_id, token=token, client=client)     # the app was removed / the token changed
            removed += 1
        else:
            errors.append(f"{status}:{str(text)[:60]}")
    return {"sent": sent, "removed": removed, "errors": errors[:5]}


# MARK: the words (EN / ES, the same as the app's)

def _videos_ready(n: int, lang: str) -> str:
    if lang == "es":
        return "Tu video está listo ✨" if n == 1 else f"Tus {n} videos están listos ✨"
    return "Your video is ready ✨" if n == 1 else f"Your {n} videos are ready ✨"


def group_title(ready: int, failed: int, lang: str) -> str:
    if failed == 0:
        return _videos_ready(ready, lang)
    if lang == "es":
        attention = "1 necesita atención" if failed == 1 else f"{failed} necesitan atención"
        if ready == 0:
            return "1 video necesita atención" if failed == 1 else f"{failed} videos necesitan atención"
        return ("1 listo" if ready == 1 else f"{ready} listos") + ", " + attention
    attention = "1 needs attention" if failed == 1 else f"{failed} need attention"
    if ready == 0:
        return "1 video needs attention" if failed == 1 else f"{failed} videos need attention"
    return ("1 ready" if ready == 1 else f"{ready} ready") + ", " + attention


def _them(lang: str) -> str:
    return "Abre CutSell para revisarlos y exportarlos." if lang == "es" else "Open CutSell to review and export them."


# MARK: events from the worker

def _bulk(sources) -> tuple[str, int]:
    for source in sources or []:
        meta = getattr(source, "metadata", None)
        if meta is None and isinstance(source, dict):
            meta = source.get("metadata")
        if isinstance(meta, Mapping):
            try:
                size = int(str(meta.get("bulk_size") or "0"))
            except ValueError:
                size = 0
            if size > 1 and meta.get("bulk_id"):
                return str(meta["bulk_id"]), size
    return "", 0


def cut_finished(*, user_id: str, project_id: str, sources, ok: bool, error: str = "", client=None, post=None) -> dict:
    """A cut is done. Single / Combine: one push. Bulk: counted; ONE push when the whole send is done."""
    try:
        bulk_id, size = _bulk(sources)
        if not bulk_id:
            if ok:
                return send(user_id, lambda lang: (("Tu corte está listo", "Abre CutSell para revisarlo y exportarlo.")
                                                    if lang == "es" else
                                                    ("Your cut is ready", "Open CutSell to review and export it.")),
                            collapse_id="cut-" + project_id, data={"project_id": project_id, "kind": "draft_ready"},
                            client=client, post=post)
            def failed_text(lang):
                if error == "no_speech":
                    return ("CutSell", "No pudimos escuchar a nadie hablando en este video" if lang == "es"
                            else "We couldn't hear anyone talking in this video")
                return ("CutSell", "Este corte necesita otro intento." if lang == "es" else "This cut needs another try.")
            return send(user_id, failed_text, collapse_id="cut-" + project_id,
                        data={"project_id": project_id, "kind": "processing_failed"}, client=client, post=post)
        r = _redis(client)
        key = f"cutsell:push:bulk:{_scope(user_id)}:{bulk_id}"
        # Each video counts once (a retried job does not count twice; a retry that succeeds moves it to ready).
        r.hset(key, project_id, "ready" if ok else "failed")
        r.hset(key, "_size", str(size))
        r.expire(key, GROUP_TTL)
        states = {(k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
                  for k, v in (r.hgetall(key) or {}).items()}
        done = {k: v for k, v in states.items() if not k.startswith("_")}
        if len(done) < size:
            return {"sent": 0, "reason": "bulk_waiting", "done": len(done), "size": size}
        if not r.set(key + ":told", "1", nx=True, ex=GROUP_TTL):
            return {"sent": 0, "reason": "bulk_already_told"}
        ready = sum(1 for v in done.values() if v == "ready")
        failed = len(done) - ready
        return send(user_id, lambda lang: (group_title(ready, failed, lang), _them(lang)),
                    collapse_id="bulk-" + bulk_id, data={"bulk_id": bulk_id, "kind": "bulk_ready"}, client=client, post=post)
    except Exception as exc:
        return {"sent": 0, "reason": "error_" + exc.__class__.__name__}


def mix_finished(*, user_id: str, mix_id: str, videos: int, project_id: str = "", client=None, post=None) -> dict:
    try:
        return send(user_id, lambda lang: (_videos_ready(videos, lang), _them(lang)), collapse_id="mix-" + mix_id,
                    data={"mix_id": mix_id, "project_id": project_id, "kind": "mix_ready"}, client=client, post=post)
    except Exception as exc:
        return {"sent": 0, "reason": "error_" + exc.__class__.__name__}
