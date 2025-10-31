#!/usr/bin/env python3
# pipeline.py
#
# MODE B: take 1 raw long video, build ONE cleaned ad:
# HOOK -> PROBLEM -> FEATURE -> PROOF -> CTA
#
# - chunk audio into speaking segments
# - throw away obvious "wait / restart / blooper"
# - merge consecutive takes into longer chains (don't cut to 2s)
# - tag each take with slot (HOOK / PROBLEM / FEATURE / PROOF / CTA)
# - score, pick best per slot
# - stitch in funnel order
# - hard cap final length with MAX_DURATION_SEC
#
# Requirements:
#   - ffmpeg/ffprobe via env FFMPEG_BIN, FFPROBE_BIN
#   - whisper (if ASR_ENABLED=1)
#   - sentence-transformers (semantic tagging)
#   - boto3 (S3 upload)
#
# NOTE: We do not rely on FUNNEL_COUNTS from the request anymore.
# We force 1 HOOK, 1 PROBLEM, 1 FEATURE, 1 PROOF, 1 CTA if available.

import os, subprocess, uuid, json, tempfile, shutil, math, time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

# ------------------ ENV / CONSTANTS ------------------

FFMPEG_BIN   = os.getenv("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE_BIN  = os.getenv("FFPROBE_BIN", "/usr/bin/ffprobe")

ASR_ENABLED  = os.getenv("ASR_ENABLED", "1") == "1"
ASR_MODEL    = os.getenv("ASR_MODEL_SIZE", "small")  # "tiny","base","small"...
ASR_LANG     = os.getenv("ASR_LANGUAGE", "en")

MAX_TAKE_SEC      = float(os.getenv("MAX_TAKE_SEC", "220"))
MIN_TAKE_SEC      = float(os.getenv("MIN_TAKE_SEC", "2.0"))
MAX_DURATION_SEC  = float(os.getenv("MAX_DURATION_SEC", "220"))
FALLBACK_MIN_SEC  = float(os.getenv("FALLBACK_MIN_SEC", "60"))

SEM_DUP_THRESHOLD = float(os.getenv("SEM_DUP_THRESHOLD", "0.88"))
SEM_MERGE_SIM     = float(os.getenv("SEM_MERGE_SIM", "0.70"))
VIZ_MERGE_SIM     = float(os.getenv("VIZ_MERGE_SIM", "0.70"))
MERGE_MAX_CHAIN   = int(os.getenv("MERGE_MAX_CHAIN", "999"))

W_SEM   = float(os.getenv("W_SEM",  "1.2"))
W_FACE  = float(os.getenv("W_FACE", "0.8"))
W_SCENE = float(os.getenv("W_SCENE","0.5"))
W_VTX   = float(os.getenv("W_VTX",  "0.8"))
W_PROD  = float(os.getenv("W_PROD", "0.9"))
W_OCR   = float(os.getenv("W_OCR",  "0.6"))

# slot constraints (a clip may be rejected for a slot if it doesn't satisfy required flags)
REQ_PRODUCT = set((os.getenv("SLOT_REQUIRE_PRODUCT","") or "").split(",")) - {""}
REQ_CTA_OCR = set((os.getenv("SLOT_REQUIRE_OCR_CTA","") or "").split(",")) - {""}

# filler / retry detection
FILLER_WORDS = set([w.strip().lower() for w in os.getenv(
    "SEM_FILLER_LIST","um,uh,like,so,okay,wait,hold on,let me start again"
).split(",") if w.strip()])

SEM_FILLER_MAX_RATE = float(os.getenv("SEM_FILLER_MAX_RATE","0.08"))

# ------------------ UTIL ------------------

def _run(cmd: List[str]) -> Tuple[int,str,str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out,err = p.communicate()
    return p.returncode,out,err

def _probe_duration(path:str)->float:
    code,out,err = _run([
        FFPROBE_BIN,"-v","error","-show_entries","format=duration",
        "-of","default=nokey=1:noprint_wrappers=1", path
    ])
    if code!=0: return 0.0
    try:
        return float(out.strip())
    except:
        return 0.0

# ------------------ DATA MODEL ------------------

@dataclass
class Take:
    id: str
    start: float
    end: float
    text: str = ""
    face_q: float = 1.0
    scene_q: float = 1.0
    vtx_sim: float = 0.0
    has_product: bool = False
    ocr_hit: int = 0
    slot_hint: Optional[str] = None
    meta: Dict[str,Any] = field(default_factory=dict)

# ------------------ STEP 1: ASR ------------------

def asr_segments(video_path:str) -> List[Take]:
    """
    Run whisper (or fallback) to get transcript segments with timestamps.
    Returns list[Take] with start,end,text.
    """
    # For now we assume whisperx-ish output has "segments".
    # We'll do a super simple local whisper usage.
    # If ASR_ENABLED=0 => we just create one giant take covering whole video,
    # text="" so we can still export raw visuals.
    dur = _probe_duration(video_path)

    if not ASR_ENABLED:
        return [Take(id="T0001", start=0.0, end=min(dur,MAX_TAKE_SEC), text="")]

    try:
        import whisper
    except Exception:
        # fallback no-ASR
        return [Take(id="T0001", start=0.0, end=min(dur,MAX_TAKE_SEC), text="")]

    model = whisper.load_model(ASR_MODEL)
    result = model.transcribe(video_path, language=ASR_LANG)
    segs = []
    idx = 1
    for seg in result.get("segments", []):
        s = float(seg.get("start",0.0))
        e = float(seg.get("end",0.0))
        tx = seg.get("text","").strip()
        if e<=s: continue
        t = Take(
            id=f"T{idx:04d}",
            start=s,
            end=min(e, s+MAX_TAKE_SEC),
            text=tx
        )
        segs.append(t)
        idx+=1
    return segs

# ------------------ STEP 2: CLEAN RETRIES ------------------

def _is_retry_or_garbage(text:str)->bool:
    # kill obvious retries / bloopers like "wait wait no no let me start over"
    if not text: return False
    low = text.lower()
    retry_words = [
        "wait", "hold on", "let me start again", "let me start over",
        "no no no", "restart", "okay okay okay", "that was bad",
        "i can't talk", "cut that", "delete that"
    ]
    if any(w in low for w in retry_words):
        return True

    words = low.split()
    if not words: return False
    fillers = sum(1 for w in words if w in FILLER_WORDS)
    rate = fillers / max(1,len(words))
    if rate > SEM_FILLER_MAX_RATE:
        return True

    return False

def drop_retries(takes:List[Take])->List[Take]:
    cleaned=[]
    for t in takes:
        if _is_retry_or_garbage(t.text):
            continue
        # also drop tiny stutters under MIN_TAKE_SEC
        dur = t.end - t.start
        if dur < MIN_TAKE_SEC:
            continue
        cleaned.append(t)
    return cleaned

# ------------------ STEP 3: MERGE CHAINS ------------------

def _text_sim(a:str,b:str, cache=None)->float:
    """
    cosine similarity using sentence-transformers,
    fallback tf-idf if sentence-transformers missing.
    """
    if cache is None: cache={}
    key=(a,b)
    if key in cache: return cache[key]

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        if "embedder" not in cache:
            cache["embedder"]=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        emb=cache["embedder"]
        va,vb = emb.encode([a,b], normalize_embeddings=True)
        sim=float((va*vb).sum())
    except Exception:
        # tf-idf fallback
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(min_df=1).fit([a,b])
        Xa = vec.transform([a])
        Xb = vec.transform([b])
        sim=float(cosine_similarity(Xa,Xb)[0,0])

    cache[key]=sim
    return sim

def _can_merge(a:Take,b:Take,cache=None)->bool:
    # semantic continuity
    s = _text_sim(a.text,b.text,cache=cache)
    if s < SEM_MERGE_SIM:
        return False
    # visual continuity proxy: we don't have real face/scene tracking in this cut,
    # but we approximate with scene_q similarity:
    vscore = min(a.scene_q, b.scene_q)
    if vscore < VIZ_MERGE_SIM:
        return False
    # simple temporal sanity: they should be close in time (<=2s gap)
    if b.start - a.end > 2.0:
        return False
    return True

def merge_chains(takes:List[Take])->List[Take]:
    """
    Walk through takes in time order and merge consecutive ones that are
    essentially the same scene/same idea so we don't cut every 2s.
    """
    takes = sorted(takes, key=lambda t:(t.start,t.end))
    out=[]
    cache={}
    i=0
    while i < len(takes):
        chain=[takes[i]]
        j=i
        while (j+1 < len(takes)
               and len(chain)<MERGE_MAX_CHAIN
               and _can_merge(chain[-1], takes[j+1], cache=cache)):
            chain.append(takes[j+1])
            j+=1
        # merge chain -> first item expanded
        merged=chain[0]
        merged.end = chain[-1].end
        merged.text=" ".join([c.text for c in chain]).strip()
        merged.meta["chain_ids"]=[c.id for c in chain]
        out.append(merged)
        i=j+1
    return out

# ------------------ STEP 4: TAG SLOTS ------------------

SLOT_KEYWORDS={
    "HOOK":    ["stop scrolling","you need to hear this","listen","if you","is your","nobody tells you"],
    "PROBLEM": ["problem","struggle","tired of","embarrassing","i hate","i used to","odor","pain","dry"],
    "FEATURE": ["made with","ingredient","formula","this bag","it has","comes with","packed with","each gummy"],
    "PROOF":   ["i use","i've been using","it works","results","reviews","testimonials","i get compliments"],
    "CTA":     ["link","grab yours","get yours today","shop now","i left it below","click","use code"]
}

def guess_slot(t:Take)->str:
    txt=t.text.lower()
    # direct keyword pass
    for slot,keys in SLOT_KEYWORDS.items():
        for k in keys:
            if k in txt:
                return slot
    # fallback heuristics
    # short hype opener -> HOOK
    if len(txt.split()) < 25:
        return "HOOK"
    # long descriptive spec -> FEATURE
    if "it has" in txt or "it's made" in txt or "each gummy" in txt:
        return "FEATURE"
    return "PROOF"

def tag_all(takes:List[Take])->List[Take]:
    for t in takes:
        t.slot_hint = guess_slot(t)
    return takes

# ------------------ STEP 5: SCORE + PICK BEST FUNNEL ------------------

def _slot_blocked(slot:str, t:Take)->bool:
    # If slot requires product-on-screen and we didn't mark it:
    if slot in REQ_PRODUCT and not t.has_product:
        return True
    # If CTA requires OCR hit and we didn't hit:
    if slot in REQ_CTA_OCR and t.ocr_hit < 1:
        return True
    return False

def score_take(t:Take,slot:str)->float:
    # base semantic quality
    sem_score = 1.0
    if _is_retry_or_garbage(t.text):
        sem_score -= 0.5
    # combine weights
    return (
        W_SEM   * sem_score +
        W_FACE  * float(t.face_q) +
        W_SCENE * float(t.scene_q) +
        W_PROD  * (1.0 if t.has_product else 0.0) +
        W_OCR   * min(1.0, float(t.ocr_hit)) +
        W_VTX   * float(t.vtx_sim)
    )

def pick_best_by_slot(takes:List[Take])->Dict[str,Take]:
    # choose ONE best HOOK, PROBLEM, FEATURE, PROOF, CTA (if exists)
    wanted_slots=["HOOK","PROBLEM","FEATURE","PROOF","CTA"]
    best={}
    for slot in wanted_slots:
        pool=[t for t in takes if t.slot_hint==slot and not _slot_blocked(slot,t)]
        if not pool:
            continue
        ranked=sorted(pool, key=lambda x: score_take(x,slot), reverse=True)
        best[slot]=ranked[0]
    return best

# ------------------ STEP 6: FALLBACK IF WE'RE TOO SHORT ------------------

def ensure_minimum_runtime(best_map:Dict[str,Take], merged_takes:List[Take])->List[Take]:
    """
    We expect ~1 clip per slot.
    If final runtime < FALLBACK_MIN_SEC and we actually have long talk,
    just return the longest continuous merged segment (so user at least gets content).
    """
    chosen=[best_map[s] for s in ["HOOK","PROBLEM","FEATURE","PROOF","CTA"] if s in best_map]
    if not chosen:
        # nothing tagged at all? give longest merged take.
        if not merged_takes:
            return []
        longest=max(merged_takes, key=lambda t:(t.end-t.start))
        return [longest]

    total = sum((t.end-t.start) for t in chosen)
    if total < FALLBACK_MIN_SEC:
        # grab longest merged take from full list
        longest=max(merged_takes, key=lambda t:(t.end-t.start))
        # if longest is already inside chosen, keep chosen
        if all(longest is not c for c in chosen):
            # prepend longest so user gets context
            chosen=[longest]+chosen
    return chosen

# ------------------ STEP 7: BUILD FINAL ORDER + CUT WITH FFMPEG ------------------

def build_funnel_order(best_map:Dict[str,Take])->List[Take]:
    order=[]
    for slot in ["HOOK","PROBLEM","FEATURE","PROOF","CTA"]:
        if slot in best_map:
            order.append(best_map[slot])
    return order

def _ffmpeg_cut(in_path:str, start:float, end:float, out_path:str):
    dur = max(0.01, end-start)
    cmd=[
        FFMPEG_BIN,"-y",
        "-ss",f"{start:.3f}",
        "-i",in_path,
        "-t",f"{dur:.3f}",
        "-c:v","libx264","-preset","fast","-crf","23",
        "-pix_fmt","yuv420p","-g","48",
        "-c:a","aac","-b:a","128k",
        out_path
    ]
    code,out,err=_run(cmd)
    if code!=0:
        raise RuntimeError(f"ffmpeg cut failed {code}: {err}")

def _ffmpeg_concat(list_path:str, out_path:str):
    cmd=[
        FFMPEG_BIN,"-y",
        "-f","concat","-safe","0",
        "-i",list_path,
        "-c:v","libx264","-preset","fast","-crf","23",
        "-pix_fmt","yuv420p","-g","48",
        "-c:a","aac","-b:a","128k",
        out_path
    ]
    code,out,err=_run(cmd)
    if code!=0:
        raise RuntimeError(f"ffmpeg concat failed {code}: {err}")

def stitch_video(video_path:str, takes:List[Take], max_len_sec:float)->Tuple[str,float,List[Dict[str,Any]]]:
    """
    Export ffmpeg parts for each chosen take in funnel order,
    stop once we hit max_len_sec.
    Return (final_mp4_path, final_duration, debug_clips).
    """
    workdir=tempfile.mkdtemp(prefix="ed_")
    parts=[]
    manifest_lines=[]
    used=[]
    running=0.0
    idx=1
    for t in takes:
        seg_len = t.end - t.start
        if seg_len<=0: continue
        if running >= max_len_sec:
            break
        # if clip would overflow cap, trim tail
        take_dur = min(seg_len, max_len_sec-running)
        seg_out = os.path.join(workdir, f"part{idx:02d}.mp4")
        _ffmpeg_cut(video_path, t.start, t.start+take_dur, seg_out)
        manifest_lines.append(f"file '{seg_out}'\n")
        parts.append(seg_out)
        used.append({
            "id": t.id,
            "slot": t.slot_hint,
            "start": t.start,
            "end": t.start+take_dur,
            "score": score_take(t, t.slot_hint or "HOOK"),
            "text": t.text
        })
        running += take_dur
        idx+=1

    concat_txt=os.path.join(workdir,"concat.txt")
    with open(concat_txt,"w") as f:
        f.writelines(manifest_lines)

    final_path=os.path.join(workdir,f"{uuid.uuid4().hex}.mp4")
    _ffmpeg_concat(concat_txt, final_path)

    final_dur=_probe_duration(final_path)
    return final_path, final_dur, used

# ------------------ MAIN PIPE ------------------

def run_pipeline(local_path:str, session_id:str="session")->Dict[str,Any]:
    """
    This is what tasks.job_render() will call.
    Returns a dict that later gets uploaded to S3 and returned to FastAPI.
    """

    # 1. ASR
    segs = asr_segments(local_path)

    # 2. drop retries / stutters
    segs2 = drop_retries(segs)

    # 3. merge chains for continuity
    merged = merge_chains(segs2)

    # 4. slot tagging
    tagged = tag_all(merged)

    # 5. pick best per slot
    best_map = pick_best_by_slot(tagged)

    # 6. fallback to guarantee we don't just send a 2s nothing
    chosen_for_export = ensure_minimum_runtime(best_map, merged)

    # 7. build funnel order (HOOK→PROBLEM→FEATURE→PROOF→CTA)
    ordered = build_funnel_order(best_map)
    if not ordered:
        ordered = chosen_for_export  # extreme fallback

    # 8. stitch video
    final_path, final_dur, used_clips = stitch_video(
        local_path,
        ordered,
        MAX_DURATION_SEC
    )

    # response structure expected by /jobs/<id> in your FastAPI
    result = {
        "ok": True,
        "input_local": local_path,
        "duration_sec": round(final_dur,3),
        "clips": [
            {
                "id": c["id"],
                "slot": c["slot"],
                "start": round(c["start"],2),
                "end": round(c["end"],2),
                "score": round(c["score"],2),
                "text": c["text"],
            }
            for c in used_clips
        ],
        "slots": {
            "HOOK":   [ {"id":t.id,"start":t.start,"end":t.end,"text":t.text} for t in tagged if t.slot_hint=="HOOK" ],
            "PROBLEM":[ {"id":t.id,"start":t.start,"end":t.end,"text":t.text} for t in tagged if t.slot_hint=="PROBLEM" ],
            "FEATURE":[ {"id":t.id,"start":t.start,"end":t.end,"text":t.text} for t in tagged if t.slot_hint=="FEATURE" ],
            "PROOF":  [ {"id":t.id,"start":t.start,"end":t.end,"text":t.text} for t in tagged if t.slot_hint=="PROOF" ],
            "CTA":    [ {"id":t.id,"start":t.start,"end":t.end,"text":t.text} for t in tagged if t.slot_hint=="CTA" ],
        },
        "semantic": True,
        "vision": False,
        "asr": ASR_ENABLED,
    }

    return result
