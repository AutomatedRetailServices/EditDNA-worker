"""Experimental Medium text with isolated WhisperX word alignment, no fallback."""
from dataclasses import dataclass, field
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from .contracts import TranscriptSegment
from .gpt_whisperx_asr import ALIGNER_PYTHON, ALIGNER_SCRIPT, AlignmentEvidenceError, checked_segments

PROVIDER = "faster-whisper-medium-whisperx"

def _drop_degenerate_duplicate_tail(segments):
    """Discard only a physically impossible repeated ASR tail at the source edge."""
    kept = []
    dropped = []
    for index, segment in enumerate(segments):
        normalized = " ".join(segment.text.casefold().split())
        matching_previous = next((
            previous for previous in reversed(kept)
            if " ".join(previous.text.casefold().split()) == normalized
        ), None)
        if (matching_previous is not None
                and index == len(segments) - 1
                and 0 < segment.end - segment.start <= 0.05
                and 0 <= segment.start - matching_previous.end <= 15.0
                and len(segment.text.split()) >= 3):
            dropped.append({"start": segment.start, "end": segment.end,
                            "reason": "degenerate_duplicate_tail"})
            continue
        kept.append(segment)
    return tuple(kept), dropped



@dataclass(frozen=True)
class AlignmentFingerprint:
    decode: str
    language: str | None

    def fingerprint(self):
        spec = {"provider": PROVIDER, "decode": self.decode, "language": self.language,
                "whisperx": "3.8.6", "policy": "strict-segment-word-coverage-v3-terminal-duplicate-tail",
                "interpolation": "ignore", "audio": "pcm_s16le-mono-16000"}
        return "asrcfg_" + hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class MediumWhisperXASR:
    base: object
    model_name: str = field(default="medium+whisperx-3.8.6", init=False)
    last_audit: dict = field(default_factory=dict, init=False)
    _source_cache: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if self.base.model_name != "medium":
            raise ValueError("Medium/WhisperX requires the medium decoder")

    def config_fingerprint(self, *, language_hint=None):
        return AlignmentFingerprint(self.base.config_fingerprint(language_hint=language_hint).fingerprint(), language_hint)

    def transcribe(self, path, *, source_asset_id, language_hint=None):
        if language_hint is not None and language_hint not in {"en", "es"}:
            raise ValueError("Qualified alignment supports en/es only")
        if not Path(ALIGNER_PYTHON).is_file() or not Path(ALIGNER_SCRIPT).is_file():
            raise RuntimeError("The isolated WhisperX environment is missing")
        digest = hashlib.sha256()
        with open(path, "rb") as src:
            for block in iter(lambda: src.read(1024 * 1024), b""):
                digest.update(block)
        key = (digest.hexdigest(), source_asset_id, self.config_fingerprint(language_hint=language_hint).fingerprint())
        if key in self._source_cache:
            result, audit = self._source_cache[key]
            audit["cache_hit_count"] += 1
            self.last_audit = copy.deepcopy(audit)
            return result
        began = time.monotonic()
        self.last_audit = {"provider": PROVIDER, "model": "medium", "status": "running",
                           "source_media_sha256": key[0], "config_fingerprint": key[2],
                           "cache_hit_count": 0, "fallback": None, "timestamp_interpolation": False,
                           "confidence_semantics": "CTC alignment score; not lexical confidence", "chunks": []}
        try:
            original = self.base.transcribe(path, source_asset_id=source_asset_id, language_hint=language_hint)
            original, dropped = _drop_degenerate_duplicate_tail(original)
            self.last_audit["discarded_degenerate_asr_tails"] = dropped
            language = language_hint or getattr(self.base, "last_detected_language", None)
            if original and language not in {"en", "es"}:
                raise AlignmentEvidenceError("No qualified detected alignment language")
            self.last_audit["detected_language"] = language
            with tempfile.TemporaryDirectory(prefix="cutsell-medium-whisperx-") as temp:
                work = Path(temp)
                for i, segment in enumerate(original):
                    if not (0 <= segment.start < segment.end):
                        raise AlignmentEvidenceError("Invalid Medium segment window")
                    chunk = {"index": i, "start": segment.start, "end": segment.end,
                             "text": " ".join(segment.text.split()), "language": language}
                    self.last_audit["chunks"].append(chunk)
                    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", path,
                                    "-ss", str(segment.start), "-t", str(segment.end-segment.start),
                                    "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "-y",
                                    str(work / f"chunk-{i:04d}.wav")], check=True, capture_output=True, timeout=120)
                (work / "input.json").write_text(json.dumps({"chunks": self.last_audit["chunks"]}))
                env = {k: v for k, v in os.environ.items() if k in {
                    "PATH", "HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES",
                    "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "HF_HOME", "TORCH_HOME", "NLTK_DATA", "TMPDIR"}}
                proc = subprocess.run([ALIGNER_PYTHON, ALIGNER_SCRIPT, str(work / "input.json"),
                                       str(work / "output.json")], env=env, capture_output=True, text=True, timeout=900)
                if proc.returncode:
                    raise AlignmentEvidenceError("WhisperX sidecar failed: " + proc.stderr[-4000:])
                aligned = json.loads((work / "output.json").read_text())
                if len(aligned["chunks"]) != len(original):
                    raise AlignmentEvidenceError("WhisperX changed segment count")
                self.last_audit["alignment_runtime"] = aligned["runtime"]
                result = []
                for segment, chunk, aligned_chunk in zip(original, self.last_audit["chunks"], aligned["chunks"]):
                    if aligned_chunk["index"] != chunk["index"]:
                        raise AlignmentEvidenceError("WhisperX changed segment order")
                    checked = checked_segments(chunk, aligned_chunk, source_asset_id)
                    words = tuple(w for s in checked for w in s.words)
                    if not words:
                        raise AlignmentEvidenceError("WhisperX lost a nonempty segment")
                    if result and words[0].start < result[-1].end - 0.002:
                        raise AlignmentEvidenceError("WhisperX overlapped adjacent segments")
                    # Preserve Medium segment boundaries in the text structure;
                    # only start/end/word evidence changes, no sentence re-chunking.
                    result.append(TranscriptSegment(source_asset_id, words[0].start, words[-1].end,
                                                    segment.text, words))
                    chunk.update(alignment_model=aligned_chunk.get("alignment_model"), aligned_word_count=len(words))
            output = tuple(result)
            self.last_audit.update(status="passed", word_count=sum(len(s.words) for s in output),
                                   segment_count=len(output), elapsed_sec=round(time.monotonic()-began, 3))
            self._source_cache[key] = (output, copy.deepcopy(self.last_audit))
            return output
        except Exception as exc:
            self.last_audit.update(status="failed", error=str(exc))
            print("MEDIUM_WHISPERX_ASR_AUDIT=" + json.dumps(self.last_audit, ensure_ascii=False), flush=True)
            raise
