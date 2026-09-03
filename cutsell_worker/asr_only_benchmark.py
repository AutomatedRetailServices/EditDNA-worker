"""D-053 Section 5: isolated ASR-only benchmark harness.

Runs ONLY the ASR stage (download the same Video00 source, transcribe,
build CanonicalASREvidence) and stops -- no AttemptReconstructor, no
Gemini/hybrid editorial, no Unified Realization Resolver, no Render/QC.
This makes one ASR determinism test materially cheaper and faster than a
full Video00 RAW (which additionally runs local-performance/vision
analysis, clean_cut, hybrid editorial, resolver, and render/QC).

Also carries the D-053 Section 1 "audit the actual Modal ASR runtime"
payload: installed faster-whisper/ctranslate2/torch versions, GPU model,
Python version, CUDA runtime, and the ACTUAL installed
``WhisperModel.transcribe`` signature (via ``inspect.signature`` on the
live import) -- ground truth from the exact running container, never
assumed from documentation for a different version.
"""
from __future__ import annotations

import inspect
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from .asr import FasterWhisperASR, load_asr_provider_from_env
from .canonical_asr_evidence import build_canonical_asr_evidence, normalize_transcript_segments
from .config import load_runtime_config
from .contracts import ProcessingRequest, SourceAsset
from .media_probe import probe_media
from .source_identity import stable_source_id
from .storage import download_source


def _safe_id(value: str, fallback: str) -> str:
    raw = str(value or fallback).strip()
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in raw)[:100]


def _nvidia_smi_field(field: str) -> str | None:
    """Best-effort single-field query via the ``nvidia-smi`` CLI -- never
    torch. ``cutsell_worker`` is a deliberately torch-free package (see
    tests/test_cutsell_clean_worker_dependency_boundary.py); GPU/CUDA-
    runtime facts are read from the driver's own CLI tool instead, exactly
    like a human operator would ground-truth them on the box."""
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return lines[0] if lines else None
    except Exception:  # noqa: BLE001 -- best-effort, no nvidia-smi binary is a valid (non-GPU) outcome
        return None


def _nvidia_smi_driver_cuda_version() -> str | None:
    """The CUDA runtime version nvidia-smi's own header reports (its
    driver's max-supported CUDA version) -- not queryable via
    ``--query-gpu``, only via the plain-text banner."""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10, check=True)
        match = re.search(r"CUDA Version:\s*([\d.]+)", result.stdout)
        return match.group(1) if match else None
    except Exception:  # noqa: BLE001
        return None


def collect_asr_runtime_audit() -> dict[str, Any]:
    """D-053 Section 1: ground-truth the actual installed ASR runtime --
    never a documentation guess for a different version. Every field is
    best-effort (wrapped so a missing/renamed attribute in a future
    library version degrades to ``None``/an error string rather than
    crashing the whole harness).

    Deliberately never imports ``torch`` -- ``cutsell_worker`` is a clean,
    torch-free package (faster-whisper's own dependency is ctranslate2, not
    torch; see requirements.cutsell.worker.txt and
    test_cutsell_clean_worker_dependency_boundary.py). GPU model and the
    CUDA runtime version are read from the ``nvidia-smi`` CLI instead --
    the same ground-truth-from-the-actual-box discipline this section
    requires, without reintroducing the legacy heavy ML stack this worker
    was deliberately built to avoid.
    """
    audit: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
    }
    try:
        import faster_whisper
        audit["faster_whisper_version"] = getattr(faster_whisper, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        audit["faster_whisper_version"] = f"import_error:{exc}"
    try:
        import ctranslate2
        audit["ctranslate2_version"] = getattr(ctranslate2, "__version__", "unknown")
        try:
            audit["ctranslate2_supported_compute_types_cuda"] = list(
                ctranslate2.get_supported_compute_types("cuda")
            )
        except Exception as exc:  # noqa: BLE001
            audit["ctranslate2_supported_compute_types_cuda"] = f"query_error:{exc}"
        try:
            audit["ctranslate2_cuda_device_count"] = ctranslate2.get_cuda_device_count()
        except Exception as exc:  # noqa: BLE001
            audit["ctranslate2_cuda_device_count"] = f"query_error:{exc}"
    except Exception as exc:  # noqa: BLE001
        audit["ctranslate2_version"] = f"import_error:{exc}"
    audit["gpu_name"] = _nvidia_smi_field("name")
    audit["nvidia_driver_cuda_version"] = _nvidia_smi_driver_cuda_version()
    audit["cuda_available"] = audit["gpu_name"] is not None
    try:
        from faster_whisper import WhisperModel
        signature = inspect.signature(WhisperModel.transcribe)
        audit["transcribe_signature"] = {
            name: (repr(param.default) if param.default is not inspect.Parameter.empty else "REQUIRED")
            for name, param in signature.parameters.items()
            if name != "self"
        }
    except Exception as exc:  # noqa: BLE001
        audit["transcribe_signature"] = f"introspection_error:{exc}"
    return audit


def run_asr_only_benchmark(payload: Mapping[str, Any]) -> dict[str, Any]:
    """D-053 Section 5. Payload shape mirrors run_op()'s own focused
    payload: ``{"source_key": ..., "benchmark_id": ..., "language_hint":
    optional}``. Returns a plain-JSON-native dict -- never a torch-typed
    value -- so it is safe to return directly from a Modal function body
    (same discipline modal_video00_full_benchmark.py already applies)."""
    source_key = str(payload.get("source_key") or "").strip()
    if not source_key:
        raise ValueError("source_key is required")
    safe_id = _safe_id(payload.get("benchmark_id"), "asr-only")
    language_hint = payload.get("language_hint")

    config = load_runtime_config()
    if not config.s3_bucket:
        raise RuntimeError("S3_BUCKET is required")

    asr_provider: FasterWhisperASR = load_asr_provider_from_env(model_name=config.asr_model)
    source_id = stable_source_id(safe_id, 0, Path(source_key).name)
    source = SourceAsset(
        source_asset_id=source_id,
        project_id=safe_id,
        user_id="asr-only-benchmark",
        original_name=Path(source_key).name,
        source_order=0,
        duration_sec=0.0,
        uri=f"s3://{config.s3_bucket}/{source_key}",
    )
    request = ProcessingRequest(project_id=safe_id, user_id="asr-only-benchmark", sources=(source,), language_hint=language_hint)

    started = time.monotonic()
    import tempfile

    with tempfile.TemporaryDirectory(prefix="cutsell-asr-only-") as directory:
        local_path = download_source(source.uri, str(Path(directory) / source.original_name))
        duration_sec = float(probe_media(local_path).duration_sec)
        transcript = asr_provider.transcribe(local_path, source_asset_id=source_id, language_hint=request.language_hint)
    elapsed_sec = round(time.monotonic() - started, 3)

    config_fingerprint = asr_provider.config_fingerprint(language_hint=request.language_hint)
    evidence = build_canonical_asr_evidence(
        transcript,
        source_asset_id=source_id,
        language=request.language_hint,
        asr_model=asr_provider.model_name,
        asr_config_fingerprint=config_fingerprint.fingerprint(),
    )
    normalized_segments = normalize_transcript_segments(transcript)

    return {
        "ok": True,
        "benchmark_id": safe_id,
        "source_key": source_key,
        "source_duration_sec": round(duration_sec, 3),
        "elapsed_sec": elapsed_sec,
        "raw_segment_count": len(transcript),
        "raw_transcript": [
            {"start": round(seg.start, 3), "end": round(seg.end, 3), "text": seg.text}
            for seg in transcript
        ],
        "normalized_segment_count": len(normalized_segments),
        "normalized_segments": [
            {"start": round(seg.start, 3), "end": round(seg.end, 3), "text": seg.text}
            for seg in normalized_segments
        ],
        "normalized_word_count": len(evidence.normalized_words),
        "normalized_word_sequence": [word.text for word in evidence.normalized_words],
        "word_timestamps": [
            {"text": word.text, "start": round(word.start, 3), "end": round(word.end, 3)}
            for word in evidence.normalized_words
        ],
        "evidence_hash": evidence.evidence_hash,
        "asr_config_fingerprint": config_fingerprint.fingerprint(),
        "asr_config": {
            "model_name": asr_provider.model_name,
            "device": asr_provider.device,
            "compute_type": asr_provider.compute_type,
            "temperature_ladder": list(asr_provider.temperature_ladder),
            "sampling_fallback_enabled": asr_provider.sampling_fallback_enabled,
            "condition_on_previous_text": asr_provider.condition_on_previous_text,
            "vad_parameters": list(asr_provider.vad_parameters),
        },
        "runtime_audit": collect_asr_runtime_audit(),
    }
