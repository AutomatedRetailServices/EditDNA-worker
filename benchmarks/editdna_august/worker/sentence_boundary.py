# /workspace/editdna/sentence_boundary.py
"""
Lightweight sentence boundary detection with optional spaCy.
Used by upstream pipelines to split ASR text into stable sentences.
"""

from __future__ import annotations
import re
from typing import List, Dict, Any

_SENT_END = re.compile(r'([.!?])+(\s+|$)')

def split_sentences_simple(text: str) -> List[str]:
    if not text:
        return []
    out = []
    last = 0
    for m in _SENT_END.finditer(text):
        end = m.end()
        chunk = text[last:end].strip()
        if chunk:
            out.append(chunk)
        last = end
    tail = text[last:].strip()
    if tail:
        out.append(tail)
    # merge too-short fragments
    merged = []
    buf = ""
    for s in out:
        if len(s.split()) < 3:
            buf = (buf + " " + s).strip()
        else:
            if buf:
                merged.append(buf)
                buf = ""
            merged.append(s)
    if buf:
        merged.append(buf)
    return merged

def split_sentences_spacy(text: str) -> List[str]:
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            # fall back to rule-based if model not present
            return split_sentences_simple(text)
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    except Exception:
        return split_sentences_simple(text)

def split_sentences(text: str) -> List[str]:
    # try spaCy, fallback to regex
    return split_sentences_spacy(text)

def map_asr_to_sentences(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Given whisper-like segments [{start,end,text}], return sentence-level
    timed chunks by distributing each segment's time evenly across its sentences.
    """
    takes: List[Dict[str,Any]] = []
    idx = 1
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end   = float(seg.get("end", 0.0))
        text  = (seg.get("text") or "").strip()
        if not text:
            # produce a tiny bin if necessary
            takes.append({"id": f"S{idx:04d}", "start": start, "end": end, "text": ""})
            idx += 1
            continue
        sents = split_sentences(text)
        if not sents:
            takes.append({"id": f"S{idx:04d}", "start": start, "end": end, "text": text})
            idx += 1
            continue
        total = max(0.001, end - start)
        per = total / len(sents)
        t0 = start
        for s in sents:
            t1 = min(end, t0 + per)
            takes.append({"id": f"S{idx:04d}", "start": t0, "end": t1, "text": s})
            idx += 1
            t0 = t1
    return takes
