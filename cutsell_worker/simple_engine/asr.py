"""Deepgram Nova-3 word timestamps, with the exact request the engine was calibrated on
(punctuation + filler words kept, no smart formatting). Returns plain dict words {"w","s","e"}."""
from __future__ import annotations

import os
import subprocess
import time

import requests

API_URL = "https://api.deepgram.com/v1/listen"
# Brand and shop words creators say often; Deepgram keyterm prompting (Nova-3) spells them right ("Shein", not "en por").
KEYTERMS = ("Shein", "TikTok Shop", "TikTok", "Temu", "Amazon", "leggings", "jumpsuit", "gym", "top", "short", "size",
            "bra", "pads", "link")


def extract_audio(video_path: str, audio_path: str) -> str:
    """Mono 16 kHz MP3: what Deepgram hears and what the silence refine measures."""
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", str(audio_path)],
        check=True, capture_output=True, timeout=600,
    )
    return str(audio_path)


MAX_ATTEMPTS = 3
RETRY_WAIT_SEC = (2.0, 6.0)


def read_timeout_for(audio_path: str) -> float:
    """Seconds to wait for Deepgram's answer: about 60 s + 1.5 x the audio length (64 kbps MP3), 90-600 s.
    22 Oct (production): one 88 s cut waited the old flat 600 s for an answer that never came, then failed."""
    try:
        seconds = os.path.getsize(audio_path) * 8 / 64000
    except OSError:
        seconds = 300.0
    return float(min(600.0, max(90.0, 60.0 + 1.5 * seconds)))


def _post_with_retries(audio_path: str, params, key: str, *, post=None, sleep=time.sleep) -> dict:
    """Up to 3 tries on a timeout, a dropped connection, 429 or 5xx (a new upload each time); other errors raise."""
    post = post or requests.post
    timeout = (20, read_timeout_for(audio_path))
    last: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            with open(audio_path, "rb") as handle:
                response = post(API_URL, params=params, data=handle, timeout=timeout,
                                headers={"Authorization": "Token " + key, "Content-Type": "audio/mpeg"})
            if response.status_code == 200:
                return response.json()
            if response.status_code != 429 and response.status_code < 500:
                raise RuntimeError(f"Deepgram HTTP {response.status_code}")
            last = RuntimeError(f"Deepgram HTTP {response.status_code}")
        except (requests.Timeout, requests.ConnectionError) as exc:
            last = exc
        if attempt < MAX_ATTEMPTS - 1:
            sleep(RETRY_WAIT_SEC[min(attempt, len(RETRY_WAIT_SEC) - 1)])
    assert last is not None
    raise last


def transcribe_words(audio_path: str, *, language_hint: str | None = None) -> dict:
    """-> {"language", "duration", "words": [{"w","s","e"}]}"""
    key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
    if not key:
        raise RuntimeError("DEEPGRAM_API_KEY missing")
    params = [("model", "nova-3"), ("punctuate", "true"), ("filler_words", "true"), ("smart_format", "false")]
    if language_hint in ("en", "es"):
        params.append(("language", language_hint))
    else:
        params.append(("detect_language", "true"))
    params += [("keyterm", term) for term in KEYTERMS]
    payload = _post_with_retries(audio_path, params, key)
    channel = payload["results"]["channels"][0]
    alternative = channel["alternatives"][0]
    words = [
        {"w": str(item.get("punctuated_word") or item["word"]), "s": round(float(item["start"]), 3), "e": round(float(item["end"]), 3)}
        for item in alternative.get("words", [])
    ]
    return {
        "language": channel.get("detected_language") or language_hint,
        "duration": float(payload.get("metadata", {}).get("duration") or 0.0),
        "words": words,
    }
