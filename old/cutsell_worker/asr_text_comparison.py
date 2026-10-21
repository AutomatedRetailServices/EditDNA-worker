"""Opt-in ASR text comparison. Never supplies edit boundaries or calls a provider implicitly.

The original media remains local to this invocation. Only a manually requested
comparison sends its extracted audio to one fixed provider endpoint. A provider
failure is explicit; it never changes the authoritative Whisper transcript.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import time

_TEXT_PROVIDERS = ("gpt-transcribe", "gpt-4o-transcribe", "deepgram-nova-3-multi")
_MAX_OPENAI_BYTES = 25 * 1024 * 1024


def compare_text_candidate(source_path: str, provider: str, *, language_hint: str | None = None) -> dict:
    """Return the candidate's unaligned text for evaluation, never for editing."""
    if provider not in _TEXT_PROVIDERS:
        raise ValueError("unsupported ASR text comparison provider")
    import requests

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="cutsell-asr-compare-") as directory:
        audio = Path(directory) / "audio.mp3"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", source_path, "-vn", "-ac", "1", "-ar", "16000",
             "-b:a", "64k", str(audio)],
            check=True, capture_output=True, timeout=120,
        )
        if not audio.exists() or not audio.stat().st_size:
            raise RuntimeError("ASR comparison audio extraction produced no audio")
        if provider in {"gpt-transcribe", "gpt-4o-transcribe"}:
            key = os.environ.get("OPENAI_API_KEY", "").strip()
            if not key:
                raise RuntimeError("OPENAI_API_KEY is required for the requested ASR comparison")
            if audio.stat().st_size > _MAX_OPENAI_BYTES:
                raise ValueError("ASR comparison audio exceeds the provider's 25 MB upload limit")
            data = {"model": provider}
            # GPT-Transcribe uses plural languages[], not the legacy language
            # field. With no hint, keep automatic detection for both models.
            if provider == "gpt-transcribe" and language_hint:
                data["languages[]"] = language_hint
            with audio.open("rb") as source:
                response = requests.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {key}"},
                    files={"file": ("audio.mp3", source, "audio/mpeg")},
                    data=data,
                    timeout=300,
                )
            response.raise_for_status()
            text = response.json().get("text")
        else:
            key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
            if not key:
                raise RuntimeError("DEEPGRAM_API_KEY is required for the requested ASR comparison")
            with audio.open("rb") as source:
                response = requests.post(
                    "https://api.deepgram.com/v1/listen",
                    params={"model": "nova-3", "language": "multi"},
                    headers={"Authorization": f"Token {key}", "Content-Type": "audio/mpeg"},
                    data=source, timeout=300,
                )
            response.raise_for_status()
            payload = response.json()
            channels = (payload.get("results") or {}).get("channels") or ()
            alternatives = (channels[0].get("alternatives") or ()) if channels else ()
            text = alternatives[0].get("transcript") if alternatives else None
    if not isinstance(text, str) or not text.strip():
        raise ValueError("ASR comparison provider returned an empty transcript")
    return {
        "provider": provider,
        "language_hint": language_hint,
        "text": text.strip(),
        "elapsed_sec": round(time.monotonic() - started, 3),
        "has_word_timestamps": False,
        "selection_authority": False,
    }
