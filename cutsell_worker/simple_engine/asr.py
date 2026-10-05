"""Deepgram Nova-3 word timestamps, with the exact request the engine was calibrated on
(punctuation + filler words kept, no smart formatting). Returns plain dict words {"w","s","e"}."""
from __future__ import annotations

import os
import subprocess

import requests

API_URL = "https://api.deepgram.com/v1/listen"


def extract_audio(video_path: str, audio_path: str) -> str:
    """Mono 16 kHz MP3: what Deepgram hears and what the silence refine measures."""
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", str(audio_path)],
        check=True, capture_output=True, timeout=600,
    )
    return str(audio_path)


def transcribe_words(audio_path: str, *, language_hint: str | None = None) -> dict:
    """-> {"language", "duration", "words": [{"w","s","e"}]}"""
    key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
    if not key:
        raise RuntimeError("DEEPGRAM_API_KEY missing")
    params = {"model": "nova-3", "punctuate": "true", "filler_words": "true", "smart_format": "false"}
    if language_hint in ("en", "es"):
        params["language"] = language_hint
    else:
        params["detect_language"] = "true"
    with open(audio_path, "rb") as handle:
        response = requests.post(API_URL, params=params, data=handle, timeout=(20, 600),
                                 headers={"Authorization": "Token " + key, "Content-Type": "audio/mpeg"})
    if response.status_code != 200:
        raise RuntimeError(f"Deepgram HTTP {response.status_code}")
    payload = response.json()
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
