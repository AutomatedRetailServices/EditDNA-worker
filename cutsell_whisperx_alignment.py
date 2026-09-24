"""Isolated WhisperX 3.8.6 process; called by the RAW experimental ASR.

This module intentionally lives outside the clean worker dependency graph.
Only forced alignment is used: no Whisper transcription or diarization.
"""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import sys
import time


def main(manifest_path: str, output_path: str) -> None:
    import torch
    import whisperx

    if not torch.cuda.is_available():
        raise RuntimeError("WhisperX benchmark requires the approved GPU")
    began = time.monotonic()
    manifest = Path(manifest_path)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    models = {}
    results = []
    for chunk in data["chunks"]:
        language = chunk["language"]
        if not chunk["text"]:
            results.append({"index": chunk["index"], "segments": [], "alignment_model": None})
            continue
        if language not in {"en", "es"}:
            raise ValueError("Unqualified alignment language")
        if language not in models:
            models[language] = whisperx.load_align_model(language, "cuda")
        model, metadata = models[language]
        audio = whisperx.load_audio(str(manifest.parent / f"chunk-{chunk['index']:04d}.wav"))
        # "ignore" leaves missing evidence missing: no nearest-word fill.
        result = whisperx.align([{"start": 0.0, "end": len(audio) / 16000,
                                  "text": chunk["text"]}], model, metadata, audio, "cuda",
                                 interpolate_method="ignore", return_char_alignments=False)
        from whisperx.alignment import DEFAULT_ALIGN_MODELS_TORCH
        results.append({"index": chunk["index"], "segments": result["segments"],
                        "alignment_model": DEFAULT_ALIGN_MODELS_TORCH[language]})
    runtime = {"packages": {p: importlib.metadata.version(p) for p in
                           ("whisperx", "torch", "torchaudio", "transformers", "numpy")},
               "gpu": torch.cuda.get_device_name(0), "interpolation": "ignore",
               "elapsed_sec": round(time.monotonic() - began, 3)}
    Path(output_path).write_text(json.dumps({"chunks": results, "runtime": runtime},
                                           allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
