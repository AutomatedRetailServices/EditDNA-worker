# worker/sentence_boundary.py
from __future__ import annotations
import os, re, subprocess, json, tempfile
from dataclasses import dataclass
from typing import List, Tuple

SILENCE_DB   = float(os.getenv("MICRO_SILENCE_DB", "-30"))   # threshold in dB
SILENCE_MINS = float(os.getenv("MICRO_SILENCE_MIN", "0.25")) # min silence length (sec)

_SENT_SPLIT = re.compile(r'([.!?]+)(\s+|$)')  # lightweight sentence splitter

@dataclass
class Take:
    id: str
    start: float
    end: float
    text: str

def _ffprobe_duration(path: str) -> float:
    cmd = ["ffprobe","-v","error","-show_entries","format=duration",
           "-of","default=nokey=1:noprint_wrappers=1", path]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    try:
        return float(out.stdout.strip())
    except Exception:
        return 0.0

def _silences(audio_path: str, seg_start: float, seg_end: float) -> List[Tuple[float,float]]:
    """
    Returns list of (silence_start, silence_end) (absolute seconds in the original audio)
    within [seg_start, seg_end] using ffmpeg silencedetect.
    """
    dur = seg_end - seg_start
    if dur <= 0: return []
    # run silencedetect on the sub-window
    cmd = [
        "ffmpeg","-hide_banner","-loglevel","error",
        "-ss", f"{seg_start:.3f}",
        "-t", f"{dur:.3f}",
        "-i", audio_path,
        "-af", f"silencedetect=noise={SILENCE_DB}dB:d={SILENCE_MINS}",
        "-f","null","-"
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    sil = []
    cur_start = None
    for line in p.stderr.splitlines():
        # examples:
        # [silencedetect @ ...] silence_start: 1.344
        # [silencedetect @ ...] silence_end: 1.696 | silence_duration: 0.352
        if "silence_start:" in line:
            try:
                cur_start = seg_start + float(line.split("silence_start:")[1].strip().split()[0])
            except Exception:
                cur_start = None
        elif "silence_end:" in line and cur_start is not None:
            try:
                t = float(line.split("silence_end:")[1].strip().split()[0])
                sil.append((cur_start, seg_start + t))
            except Exception:
                pass
            cur_start = None
    return sil

def _split_sentences(text: str) -> List[str]:
    if not text: return []
    parts, last = [], 0
    for m in _SENT_SPLIT.finditer(text):
        end = m.end()
        parts.append(text[last:end].strip())
        last = end
    if last < len(text):
        tail = text[last:].strip()
        if tail: parts.append(tail)
    # collapse very short tails into previous
    out = []
    for s in parts:
        if len(s.split()) <= 2 and out:
            out[-1] = (out[-1] + " " + s).strip()
        else:
            out.append(s)
    return [s for s in out if s]

def micro_segment(audio_wav_path: str, take: Take) -> List[Take]:
    """
    Split a Take into sentence-level micro-takes.
    Uses: ffmpeg silencedetect for pause boundaries + punctuation for text boundaries.
    Time alignment: distribute text sentences across (pauses+1) bins proportionally.
    """
    start, end = float(take.start), float(take.end)
    text = take.text or ""
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [take]

    pauses = _silences(audio_wav_path, start, end)  # absolute times
    # keep only pauses strictly inside (not at edges)
    pauses = [(a,b) for (a,b) in pauses if (a - start) > 0.05 and (end - b) > 0.05]
    # turn into cut points (single instants) at pause midpoint
    cuts = [ (a+b)/2.0 for (a,b) in pauses ]
    # always include segment edges as bins
    bins = [start] + cuts + [end]
    # build time windows
    windows = list(zip(bins[:-1], bins[1:]))
    if len(windows) < 1: 
        return [take]

    # distribute sentences over windows by relative length
    lengths = [max(1, len(s)) for s in sentences]
    total = sum(lengths)
    target_counts = max(1, len(windows))
    # simple greedy proportional assignment
    sent_per_win = []
    acc = 0.0
    idx = 0
    for w in range(target_counts):
        quota = round((w+1)*len(sentences)/target_counts) - round(w*len(sentences)/target_counts)
        sent_per_win.append(quota)
    # enforce sum exact
    delta = len(sentences) - sum(sent_per_win)
    if delta != 0:
        sent_per_win[-1] += delta

    # build micro takes
    out: List[Take] = []
    si = 0
    for wi, (ws, we) in enumerate(windows):
        k = sent_per_win[wi]
        k = max(1, min(k, len(sentences)-si)) if wi == len(windows)-1 else max(0, min(k, len(sentences)-si))
        if k == 0:
            continue
        chunk_text = " ".join(sentences[si:si+k]).strip()
        si += k
        out.append(Take(
            id=f"{take.id}_s{wi+1:02d}",
            start=ws, end=we,
            text=chunk_text
        ))
    # safety: if we lost text (edge cases), fallback to original take
    if not out or sum(len(t.text) for t in out) < max(1, int(0.6*len(text))):
        return [take]
    return out
