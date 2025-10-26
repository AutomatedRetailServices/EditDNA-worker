# worker/sentence_boundary.py — micro-cut: silence + punctuation splitter
from __future__ import annotations
import os, re, shlex, tempfile, subprocess
from typing import List, Tuple

FFMPEG = os.getenv("FFMPEG_BIN", "/usr/bin/ffmpeg")
RETRY_TOKENS = re.compile(
    r"\b(uh|um|uhm|wait|hold on|let me start again|start over|sorry|i mean|actually|no no|take two|redo)\b",
    re.I,
)
FILLERS = {"uh","um","uhm","like","so","okay","ok","sorry"}

# Tunables (safe defaults)
SILENCE_DB = float(os.getenv("SB_NOISE_DB", "-30"))     # silence detect threshold
SILENCE_D  = float(os.getenv("SB_MIN_SILENCE", "0.20")) # min silence length (sec)
MIN_SENT_S = float(os.getenv("SB_MIN_SENTENCE", "0.80"))# drop segments shorter than this
JOIN_UNDER = float(os.getenv("SB_JOIN_UNDER", "1.00"))  # if < this, merge with neighbor
MAX_GAP_S  = float(os.getenv("SB_MAX_GAP", "0.20"))     # treat tiny gaps as zero

def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

def _silences_on_window(video_path: str, win: Tuple[float,float]) -> List[Tuple[float,float]]:
    """Return list of (silence_start, silence_end) relative to win[0]."""
    ss, ee = win
    if ee <= ss: return []
    # Run ffmpeg silencedetect over the sub-window; parse stderr
    cmd = [
        FFMPEG, "-hide_banner", "-nostats",
        "-ss", f"{ss:.3f}", "-to", f"{ee:.3f}", "-i", video_path,
        "-af", f"silencedetect=noise={SILENCE_DB}dB:d={SILENCE_D}",
        "-f", "null", "-"
    ]
    p = _run(cmd)
    lines = p.stderr.decode("utf-8", "ignore").splitlines()
    s_starts, s_ends = [], []
    for ln in lines:
        if "silence_start:" in ln:
            try:
                s_starts.append(float(ln.split("silence_start:")[1].strip()))
            except: pass
        elif "silence_end:" in ln and "silence_duration:" in ln:
            try:
                s_ends.append(float(ln.split("silence_end:")[1].split("|")[0].strip()))
            except: pass

    # Pair up starts/ends; normalize to window start
    pairs: List[Tuple[float,float]] = []
    i=j=0
    while i < len(s_starts) or j < len(s_ends):
        if i < len(s_starts) and (j >= len(s_ends) or s_starts[i] < s_ends[j]):
            # open segment until next end or window end
            st = s_starts[i]; i += 1
            en = s_ends[j] if j < len(s_ends) else (ee-ss)
            pairs.append((max(0.0, st), max(0.0, en)))
            if j < len(s_ends): j += 1
        else:
            # end without explicit start (rare): ignore
            j += 1
    # De-noise tiny gaps
    out=[]
    for st,en in pairs:
        if en-st >= SILENCE_D - 1e-6:
            out.append((st, en))
    return out

def _split_text_punct(txt: str) -> List[str]:
    # Split at ., !, ?, and strong commas + dashes when surrounded by spaces
    chunks = re.split(r"(?<=[\.\!\?])\s+|(?<=\s[,-]\s)", (txt or "").strip())
    chunks = [c.strip() for c in chunks if c and c.strip()]
    return chunks or ([txt.strip()] if txt.strip() else [])

def _align_text_to_times(chunks: List[str], ss: float, ee: float, cuts: List[float]) -> List[Tuple[float,float,str]]:
    """
    Distribute text chunks to time bins defined by cuts (relative to window).
    If no cuts, put all in one bin.
    """
    if ee <= ss: return []
    win_dur = ee - ss
    if win_dur <= 0.0: return []
    # Build bins between cuts
    times = [0.0] + sorted([c for c in cuts if 0.0 < c < win_dur]) + [win_dur]
    bins = [(times[i], times[i+1]) for i in range(len(times)-1)]

    if not bins: bins = [(0.0, win_dur)]
    if not chunks:
        return [(ss + a, ss + b, "") for (a,b) in bins]

    # Simple proportional assignment: spread chunks across bins in order
    out: List[Tuple[float,float,str]] = []
    ci = 0
    for b in bins:
        if ci >= len(chunks): break
        a,bend = b
        out.append((ss + a, ss + bend, chunks[ci]))
        ci += 1
    # any leftover chunks? merge into last bin text
    if ci < len(chunks) and out:
        a,bend,tx = out[-1]
        rest = " ".join(chunks[ci:])
        out[-1] = (a,bend, (tx + " " + rest).strip())
    return out

def _is_retry_or_filler(tx: str) -> bool:
    if RETRY_TOKENS.search(tx or ""): return True
    words = (tx or "").split()
    if not words: return True
    fillers = sum(1 for w in words if w.lower().strip(",.!?") in FILLERS)
    rate = fillers / max(1, len(words))
    return rate > float(os.getenv("SEM_FILLER_MAX_RATE", "0.08"))

def _merge_short_neighbors(spans: List[Tuple[float,float,str]]) -> List[Tuple[float,float,str]]:
    if not spans: return spans
    out = [spans[0]]
    for s,e,tx in spans[1:]:
        ps,pe,ptx = out[-1]
        if (e - s) < JOIN_UNDER or (s - pe) <= MAX_GAP_S:
            out[-1] = (ps, e, (ptx + " " + tx).strip())
        else:
            out.append((s,e,tx))
    return out

def micro_split_and_clean(video_path: str, take_window: Tuple[float,float], text: str) -> List[Tuple[float,float,str]]:
    """
    Input:
      video_path: full video path
      take_window: (start,end) seconds for the take
      text: ASR text for this take
    Output:
      list of (abs_start, abs_end, text) after micro cuts
    """
    ss, ee = take_window
    if ee <= ss:
        return []
    # 1) find silences in this window
    silences = _silences_on_window(video_path, take_window)  # relative to ss
    cut_points = sorted({round(s,3) for pair in silences for s in pair})  # start & end of each silence as potential cuts

    # 2) split text
    chunks = _split_text_punct(text)

    # 3) align text chunks to time bins (between silence cut points)
    spans = _align_text_to_times(chunks, ss, ee, cut_points)

    # 4) drop retries/fillers + too-short
    spans = [(a,b,tx) for (a,b,tx) in spans if (b-a) >= MIN_SENT_S and not _is_retry_or_filler(tx)]

    # 5) merge any tiny leftovers/adjacent bins to avoid awkward jump cuts
    spans = _merge_short_neighbors(spans)

    return spans
