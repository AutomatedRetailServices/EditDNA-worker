import os
import time
import uuid
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import re
import boto3

# ================= ENV =================

def _env_str(k: str, d: str) -> str:
    v = (os.getenv(k) or "").split("#")[0].strip()
    return v or d

def _env_float(k: str, d: float) -> float:
    v = (os.getenv(k) or "").split("#")[0].strip().split()[:1]
    try:
        return float(v[0]) if v else d
    except Exception:
        return d

FFMPEG_BIN  = _env_str("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE_BIN = _env_str("FFPROBE_BIN", "/usr/bin/ffprobe")
S3_BUCKET   = _env_str("S3_BUCKET", "")
S3_PREFIX   = _env_str("S3_PREFIX", "editdna/outputs")
AWS_REGION  = _env_str("AWS_REGION", "us-east-1")
S3_ACL      = _env_str("S3_ACL", "public-read")

MAX_DURATION_SEC = _env_float("MAX_DURATION_SEC", 120.0)
MAX_TAKE_SEC     = _env_float("MAX_TAKE_SEC", 20.0)
MIN_TAKE_SEC     = _env_float("MIN_TAKE_SEC", 2.0)

# stuff we always want to drop
BAD_PHRASES = [
    "wait", "hold on", "lemme start again", "let me start again",
    "start over", "no no", "redo", "i mean", "actually", "sorry",
]
FILLERS = {"uh","um","like","so","okay"}

CTA_FLUFF = [
    "click the link",
    "get yours today",
    "go ahead and click",
    "go ahead and grab",
    "i left it down below",
    "i left it for you down below",
]

# =============== MODEL ===============

@dataclass
class Take:
    id: str
    start: float
    end: float
    text: str

    @property
    def dur(self) -> float:
        return self.end - self.start

# =============== helpers ===============

def _run(cmd: List[str]) -> Tuple[int,str,str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out.strip(), err.strip()

def _tmpfile(suffix: str=".mp4") -> str:
    fd, p = tempfile.mkstemp(suffix=suffix); os.close(fd); return p

def _download_to_tmp(url: str) -> str:
    local_path = _tmpfile(".mp4")
    code, out, err = _run(["curl","-L","-o",local_path,url])
    if code != 0:
        raise RuntimeError(f"curl failed {code}: {err}")
    return local_path

def _ffprobe_duration(path: str) -> float:
    code, out, err = _run([
        FFPROBE_BIN,
        "-v","error",
        "-show_entries","format=duration",
        "-of","default=nokey=1:noprint_wrappers=1",
        path
    ])
    if code != 0: return 0.0
    try: return float(out.strip())
    except: return 0.0

# =============== S3 ===============

def _upload_to_s3(local_path: str, s3_prefix: Optional[str]=None) -> Dict[str,str]:
    s3 = boto3.client("s3", region_name=AWS_REGION)
    if not S3_BUCKET: raise RuntimeError("S3_BUCKET not set")
    prefix = (s3_prefix or S3_PREFIX).rstrip("/")
    key = f"{prefix}/{uuid.uuid4().hex}_{int(time.time())}.mp4"
    with open(local_path,"rb") as fh:
        s3.upload_fileobj(fh, S3_BUCKET, key, ExtraArgs={"ACL": S3_ACL, "ContentType":"video/mp4"})
    return {
        "s3_key": key,
        "s3_url": f"s3://{S3_BUCKET}/{key}",
        "https_url": f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}",
    }

# =============== ffmpeg export ===============

def _export_concat(src: str, takes: List[Take]) -> str:
    if not takes:
        takes = [Take(id="FALLBACK", start=0.0, end=min(5.0, MAX_DURATION_SEC), text="")]
    parts = []
    listfile = _tmpfile(".txt")
    for idx, t in enumerate(takes, start=1):
        part = _tmpfile(f".part{idx:02d}.mp4"); parts.append(part)
        _run([
            FFMPEG_BIN,"-y",
            "-ss",f"{t.start:.3f}",
            "-i",src,
            "-t",f"{t.dur:.3f}",
            "-c:v","libx264","-preset","fast","-crf","23",
            "-pix_fmt","yuv420p","-g","48",
            "-c:a","aac","-b:a","128k",
            part
        ])
    with open(listfile,"w") as f:
        for p in parts: f.write(f"file '{p}'\n")
    final = _tmpfile(".mp4")
    _run([
        FFMPEG_BIN,"-y",
        "-f","concat","-safe","0",
        "-i",listfile,
        "-c:v","libx264","-preset","fast","-crf","23",
        "-pix_fmt","yuv420p","-g","48",
        "-c:a","aac","-b:a","128k",
        final
    ])
    return final

# =============== ASR load ===============

def _load_asr_segments(src: str) -> Optional[List[Dict[str,Any]]]:
    try:
        from worker.asr import transcribe_segments
    except Exception:
        return None
    try:
        segs = transcribe_segments(src)
    except Exception:
        return None
    if not segs: return None
    if "temp placeholder" in (segs[0].get("text") or "").lower():
        return None
    return segs

def _segments_to_takes_asr(segs: List[Dict[str,Any]]) -> List[Take]:
    takes: List[Take] = []
    for i, seg in enumerate(segs, start=1):
        txt = (seg.get("text") or "").strip()
        if not txt: continue
        s = float(seg.get("start", 0.0))
        e = float(seg.get("end", s))
        # split long ones
        while (e - s) > MAX_TAKE_SEC:
            takes.append(Take(id=f"ASR{i:04d}_{len(takes)+1:02d}", start=s, end=s+MAX_TAKE_SEC, text=txt))
            s += MAX_TAKE_SEC
        if (e - s) >= MIN_TAKE_SEC:
            takes.append(Take(id=f"ASR{i:04d}", start=s, end=e, text=txt))
    return takes

# =============== semantic-ish utils ===============

def _is_retry_or_trash(txt: str) -> bool:
    low = txt.lower().strip()
    if not low: return True
    for p in BAD_PHRASES:
        if p in low: return True
    # high filler rate
    words = [w.strip(",.?!") for w in low.split()]
    if not words: return True
    filler = sum(1 for w in words if w in FILLERS)
    if filler / max(1,len(words)) > 0.45:
        return True
    return False

def _dedupe_takes(takes: List[Take]) -> List[Take]:
    out = []
    seen = set()
    for t in takes:
        norm = "".join(c.lower() for c in t.text if (c.isalnum() or c.isspace())).strip()
        if not norm: continue
        if norm in seen: continue
        seen.add(norm)
        out.append(t)
    return out

def _merge_adjacent(takes: List[Take], max_gap: float = 1.0, max_chain: int = 3) -> List[Take]:
    if not takes: return []
    takes = sorted(takes, key=lambda x: x.start)
    merged = []
    i = 0
    while i < len(takes):
        chain = [takes[i]]; j = i
        while (j+1) < len(takes) and len(chain) < max_chain:
            a = chain[-1]; b = takes[j+1]
            if (b.start - a.end) > max_gap: break
            chain.append(b); j += 1
        first, last = chain[0], chain[-1]
        merged.append(Take(
            id=f"{first.id}_to_{last.id}",
            start=first.start,
            end=last.end,
            text=" ".join(c.text for c in chain),
        ))
        i = j + 1
    return merged

def _extract_topic_words(all_text: str, top_k: int = 8) -> List[str]:
    # very simple: count non-stopwords and take top-k
    stops = {"the","and","or","a","an","it","is","this","that","to","for","of","on","with","in","at","you","your","i"}
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]+", all_text.lower())
    freq = {}
    for w in words:
        if w in stops: continue
        freq[w] = freq.get(w, 0) + 1
    # sort by freq
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for (w,_) in sorted_words[:top_k]]

def _score_take_by_topics(t: Take, topics: List[str]) -> float:
    low = t.text.lower()
    score = 0.0
    for tok in topics:
        if tok in low:
            score += 1.0
    return score

def _trim_repeated_ngrams(txt: str, n: int = 4) -> str:
    words = txt.split()
    if len(words) <= n*2:
        return txt
    seen = {}
    for i in range(0, len(words)-n+1):
        key = " ".join(w.lower() for w in words[i:i+n])
        if key in seen:
            # cut at second occurrence
            return " ".join(words[:i]).rstrip(" ,.;")
        else:
            seen[key] = i
    return txt

def _trim_cta_fluff(txt: str) -> str:
    low = txt.lower()
    for p in CTA_FLUFF:
        idx = low.find(p)
        if idx != -1:
            return txt[:idx].rstrip(" ,.;")
    return txt

def _clean_take_text(txt: str) -> str:
    txt = _trim_repeated_ngrams(txt, n=4)
    txt = _trim_cta_fluff(txt)
    return txt

# =============== fallback ===============

def _time_based_takes(vid_dur: float) -> List[Take]:
    takes = []
    t = 0.0
    idx = 1
    while t < vid_dur:
        end = min(t + MAX_TAKE_SEC, vid_dur)
        if (end - t) >= MIN_TAKE_SEC:
            takes.append(Take(id=f"SEG{idx:04d}", start=t, end=end,
                              text=f"Auto segment {idx} ({t:.1f}s–{end:.1f}s)"))
            idx += 1
        t = end
    return takes

# =============== public entry ===============

def run_pipeline(
    *,
    session_id: str,
    file_urls: List[str],
    portrait: bool,
    funnel_counts,
    max_duration: float,
    s3_prefix: Optional[str] = None,
    **kwargs,
) -> Dict[str,Any]:
    if not file_urls:
        return {"ok": False, "error": "no input files"}

    src = _download_to_tmp(file_urls[0])
    real_dur = _ffprobe_duration(src)
    cap = float(max_duration or MAX_DURATION_SEC)
    if real_dur > 0:
        cap = min(cap, real_dur)

    segs = _load_asr_segments(src)
    if segs is not None:
        # collect all text to guess topic
        all_txt = " ".join((s.get("text") or "") for s in segs)
        topics = _extract_topic_words(all_txt)
        # segments -> takes
        takes = _segments_to_takes_asr(segs)
        # drop obvious retries
        takes = [t for t in takes if not _is_retry_or_trash(t.text)]
        # merge
        takes = _merge_adjacent(takes)
        # score by topic
        scored = []
        for t in takes:
            clean_txt = _clean_take_text(t.text)
            sc = _score_take_by_topics(Take(t.id, t.start, t.end, clean_txt), topics)
            scored.append((sc, Take(t.id, t.start, t.end, clean_txt)))
        # keep only takes that talk about topic OR are the first/last
        scored.sort(key=lambda x: x[1].start)
        filtered: List[Take] = []
        for idx, (sc, tk) in enumerate(scored):
            if sc >= 1.0 or idx == 0 or idx == len(scored)-1:
                filtered.append(tk)
        # cap duration
        story: List[Take] = []
        total = 0.0
        for t in filtered:
            if total + t.dur > cap:
                break
            story.append(t)
            total += t.dur
        used_asr = True
    else:
        story = _time_based_takes(cap)
        used_asr = False

    final_path = _export_concat(src, story)
    up = _upload_to_s3(final_path, s3_prefix=s3_prefix)

    def _trim(txt: str, n: int = 220) -> str:
        return txt if len(txt) <= n else txt[:n].rstrip() + "..."

    clips_block = [{
        "id": t.id,
        "slot": "STORY",
        "start": t.start,
        "end": t.end,
        "score": 2.5,
        "face_q": 1.0,
        "scene_q": 1.0,
        "vtx_sim": 0.0,
        "chain_ids": [t.id],
        "text": _trim(t.text),
    } for t in story]

    slots_block: Dict[str, List[Dict[str,Any]]] = {"HOOK": [], "PROBLEM": [], "FEATURE": [], "PROOF": [], "CTA": []}
    if story:
        slots_block["HOOK"].append({
            "id": story[0].id,
            "start": story[0].start,
            "end": story[0].end,
            "text": _trim(story[0].text),
            "meta": {"slot": "HOOK", "score": 2.5, "chain_ids": [story[0].id]},
            "face_q": 1.0, "scene_q": 1.0, "vtx_sim": 0.0,
            "has_product": False, "ocr_hit": 0,
        })
    if len(story) > 2:
        for mid in story[1:-1]:
            slots_block["FEATURE"].append({
                "id": mid.id,
                "start": mid.start,
                "end": mid.end,
                "text": _trim(mid.text),
                "meta": {"slot": "FEATURE", "score": 2.0, "chain_ids": [mid.id]},
                "face_q": 1.0, "scene_q": 1.0, "vtx_sim": 0.0,
                "has_product": False, "ocr_hit": 0,
            })
    if len(story) >= 2:
        last = story[-1]
        slots_block["CTA"].append({
            "id": last.id,
            "start": last.start,
            "end": last.end,
            "text": _trim(last.text),
            "meta": {"slot": "CTA", "score": 2.0, "chain_ids": [last.id]},
            "face_q": 1.0, "scene_q": 1.0, "vtx_sim": 0.0,
            "has_product": False, "ocr_hit": 0,
        })

    return {
        "ok": True,
        "session_id": session_id,
        "input_local": src,
        "duration_sec": _ffprobe_duration(final_path),
        "s3_key": up["s3_key"],
        "s3_url": up["s3_url"],
        "https_url": up["https_url"],
        "clips": clips_block,
        "slots": slots_block,
        "asr": used_asr,
        "semantic": used_asr,
        "vision": False,
    }
