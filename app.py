# app.py — EditDNA.ai MVP
# v1.6.3  (manifest uses analysis; fallback = real duration; TalkingHead targets)

import os
import json
import time
import uuid
import shutil
import pathlib
import subprocess
from typing import List, Dict, Any, Optional, Tuple

import boto3
import requests
import redis
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel
from rq import Queue
from rq.job import Job
from rq import Retry

# ------------------------------------------------------------------------------
# ENV & clients
# ------------------------------------------------------------------------------

REDIS_URL        = os.getenv("REDIS_URL", "")
AWS_ACCESS_KEY   = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_KEY   = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION       = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET        = os.getenv("S3_BUCKET", "")
S3_PUBLIC_BASE   = os.getenv("S3_PUBLIC_BASE") or os.getenv("S3_PUBLIC_BUCKET")
S3_URL_MODE      = os.getenv("S3_URL_MODE", "auto").lower()  # auto|public|presigned

if not REDIS_URL:
    raise RuntimeError("Missing REDIS_URL")
if not AWS_ACCESS_KEY or not AWS_SECRET_KEY or not S3_BUCKET:
    print("[WARN] AWS/S3 env vars are not fully set. Uploads may fail.")

r_conn = redis.from_url(REDIS_URL, decode_responses=False)
q = Queue("default", connection=r_conn)

_s3 = boto3.client(
    "s3",
    region_name=AWS_REGION or "us-east-1",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)

TMP_ROOT = pathlib.Path("/tmp/s2c_sessions")
TMP_ROOT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# Speech-adaptive config (env tunable)
# ------------------------------------------------------------------------------

SILENCE_DB      = float(os.getenv("SILENCE_DB", "-35"))
SILENCE_MIN_D   = float(os.getenv("SILENCE_MIN_D", "0.28"))
UTT_MIN_D       = float(os.getenv("UTT_MIN_D", "0.60"))
UTT_MAX_D       = float(os.getenv("UTT_MAX_D", "14.0"))
UTT_JOIN_GAP_D  = float(os.getenv("UTT_JOIN_GAP_D", "0.25"))
UTT_PAD_D       = float(os.getenv("UTT_PAD_D", "0.06"))

# ------------------------------------------------------------------------------
# Presets (TalkingHead now has real targets)
# ------------------------------------------------------------------------------

DEFAULT_PRESETS = {
    # ≈27s total; adaptive (never cuts mid-word)
    "TalkingHead_ShockHook": [
        {"name": "Hook",         "target": 6.0, "styles": ["Talking Head", "Shock Hook"]},
        {"name": "Demo Step 1",  "target": 8.0, "styles": ["Talking Head"]},
        {"name": "Demo Step 2",  "target": 8.0, "styles": ["Talking Head"]},
        {"name": "CTA",          "target": 5.0, "styles": ["Talking Head"]},
    ],
    "ASMR_Product": [
        {"name": "Hook-Foley",   "target": 3.5, "styles": ["ASMR"], "asmr": True},
        {"name": "Prep",         "target": 7.0, "styles": ["ASMR"], "asmr": True},
        {"name": "Money Shot",   "target": 5.0, "styles": ["ASMR"], "asmr": True},
        {"name": "CTA",          "target": 3.0, "styles": ["ASMR"], "asmr": True},
    ],
}
PRESETS_PATH = os.getenv("PRESETS_PATH")

def _load_presets() -> Dict[str, Any]:
    if PRESETS_PATH and os.path.exists(PRESETS_PATH):
        try:
            with open(PRESETS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load presets from {PRESETS_PATH}: {e}")
    return DEFAULT_PRESETS

# ------------------------------------------------------------------------------
# Redis helpers
# ------------------------------------------------------------------------------

def _jset(key: str, obj: Any, ex: Optional[int] = None):
    r_conn.set(key, json.dumps(obj).encode("utf-8"), ex=ex)

def _jget(key: str) -> Optional[Any]:
    raw = r_conn.get(key)
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None

def _session_key(session_id: str): return f"session:{session_id}:files"
def _analysis_key(session_id: str): return f"session:{session_id}:analysis"
def _manifest_key(session_id: str): return f"session:{session_id}:manifest"
def _jobs_key(job_id: str): return f"job:{job_id}"
def _safe_name(name: str): return "".join(c for c in name if c.isalnum() or c in ("-", "_", ".", " ")).strip().replace(" ", "_")

# ------------------------------------------------------------------------------
# IO / ffmpeg helpers
# ------------------------------------------------------------------------------

def _download_to(tmp_dir: pathlib.Path, url: str, out_name: str) -> pathlib.Path:
    out_path = tmp_dir / out_name
    with requests.get(url, stream=True, timeout=60) as res:
        res.raise_for_status()
        with open(out_path, "wb") as f:
            shutil.copyfileobj(res.raw, f)
    return out_path

def _probe_duration(path: pathlib.Path) -> float:
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8").strip()
        return float(out)
    except Exception:
        return 0.0

def _detect_utterances(path: pathlib.Path) -> List[Dict[str,float]]:
    # Parse ffmpeg silencedetect logs to get non-silence (speech) ranges
    cmd = ["ffmpeg", "-i", str(path), "-af",
           f"silencedetect=noise={SILENCE_DB}dB:d={SILENCE_MIN_D}",
           "-f", "null", "-"]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
        txt = proc.stderr
    except Exception:
        return []
    sil_starts, sil_ends = [], []
    for line in txt.splitlines():
        line = line.strip().lower()
        if "silence_start:" in line:
            try: sil_starts.append(float(line.split("silence_start:")[1].strip()))
            except: pass
        elif "silence_end:" in line and "|" in line:
            try: sil_ends.append(float(line.split("silence_end:")[1].split("|")[0].strip()))
            except: pass
    dur = _probe_duration(path)
    if dur <= 0:
        return []
    # Build non-silence ranges
    ranges, last_sil_end = [], 0.0
    events = sorted([(t,"start") for t in sil_starts]+[(t,"end") for t in sil_ends], key=lambda x:x[0])
    for t, kind in events:
        if kind=="start":
            if t>last_sil_end: ranges.append((last_sil_end,t))
        else:
            last_sil_end=t
    if last_sil_end<dur: ranges.append((last_sil_end,dur))
    # Merge/clip/join; pad edges a little
    merged=[]
    for s,e in ranges:
        s=max(0.0,s-UTT_PAD_D); e=min(dur,e+UTT_PAD_D)
        if e-s<UTT_MIN_D: continue
        if e-s>UTT_MAX_D: e=s+UTT_MAX_D
        if merged and s-merged[-1]["end"]<=UTT_JOIN_GAP_D: merged[-1]["end"]=e
        else: merged.append({"start":float(s),"end":float(e)})
    return merged

def _ffmpeg_trim(src: pathlib.Path, dst: pathlib.Path, start: float, end: float, scale: int, fps: int):
    dur=max(0.0,end-start)
    if dur<=0: raise RuntimeError("Invalid segment duration")
    vf=f"scale=-2:{scale}"
    cmd=["ffmpeg","-y","-ss",f"{start:.3f}","-i",str(src),"-t",f"{dur:.3f}","-r",str(fps),"-vf",vf,
         "-c:v","libx264","-preset","veryfast","-crf","23","-c:a","aac","-b:a","128k",str(dst)]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def _ffmpeg_concat(tsv: List[pathlib.Path], out_path: pathlib.Path):
    list_file=out_path.parent/f"concat_{uuid.uuid4().hex}.txt"
    try:
        with open(list_file,"w") as f:
            for p in tsv: f.write(f"file '{p.as_posix()}'\n")
        cmd=["ffmpeg","-y","-f","concat","-safe","0","-i",str(list_file),"-c","copy",str(out_path)]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    finally:
        if list_file.exists(): list_file.unlink(missing_ok=True)

def _upload_final_to_s3(session_id: str, filename: str, local_path: pathlib.Path) -> str:
    key=f"sessions/{session_id}/{filename}"
    _s3.upload_file(str(local_path),S3_BUCKET,key,ExtraArgs={})
    if S3_URL_MODE=="presigned":
        return _s3.generate_presigned_url("get_object",Params={"Bucket":S3_BUCKET,"Key":key},ExpiresIn=900)
    base=S3_PUBLIC_BASE
    if base: return f"{base.rstrip('/')}/{key}"
    return _s3.generate_presigned_url("get_object",Params={"Bucket":S3_BUCKET,"Key":key},ExpiresIn=900)

# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------

class ProcessUrlsIn(BaseModel):
    urls: List[str]
    tone: Optional[str] = None
    product_link: Optional[str] = None
    features_csv: Optional[str] = None

class AutoManifestIn(BaseModel):
    session_id: str
    filename: str = "final.mp4"
    fps: int = 30
    scale: int = 720
    max_total_sec: int = 12
    max_segments_per_file: int = 1

class AnalyzeIn(BaseModel):
    session_id: str
    script_text: Optional[str] = None  # future: compare transcription to this

class ChooseBestIn(BaseModel):
    session_id: str
    target_sec: int = 12
    max_segments_per_file: int = 1
    fps: int = 30
    scale: int = 720

class BuildManifestIn(BaseModel):
    session_id: str
    preset_key: str = "TalkingHead_ShockHook"
    filename: str = "final.mp4"
    fps: int = 30
    scale: int = 720
    segments: Optional[List[Dict[str, Any]]] = None

class StitchIn(BaseModel):
    session_id: str
    filename: str = "final.mp4"

class StitchAsyncIn(BaseModel):
    session_id: str
    filename: str = "final.mp4"
    fps: int = 30
    scale: int = 720
    manifest: Dict[str, Any]

# ------------------------------------------------------------------------------
# Core jobs
# ------------------------------------------------------------------------------

def stitch_core(session_id:str, filename:str, manifest:Dict[str,Any], fps:int, scale:int)->Dict[str,Any]:
    files=_jget(_session_key(session_id)) or {}
    if not files: raise RuntimeError("session not found")
    work_dir=TMP_ROOT/session_id; work_dir.mkdir(parents=True,exist_ok=True)
    cache_dir=work_dir/"cache"; cache_dir.mkdir(exist_ok=True)
    tmp_segments=[]
    try:
        for idx,seg in enumerate(manifest.get("segments",[])):
            file_id=seg["file_id"]; start=float(seg.get("start",0)); end=float(seg.get("end",start+1))
            src_url=files.get(file_id)
            if not src_url: raise RuntimeError(f"file_id not found: {file_id}")
            cache_name=f"{file_id}.src"; cache_path=cache_dir/cache_name
            if not cache_path.exists(): _download_to(cache_dir,src_url,cache_name)
            out_piece=work_dir/f"piece_{idx:03d}.mp4"
            _ffmpeg_trim(cache_path,out_piece,start,end,scale,fps)
            tmp_segments.append(out_piece)
        safe_name=_safe_name(filename or "final.mp4"); final_path=work_dir/safe_name
        _ffmpeg_concat(tmp_segments,final_path)
        url=_upload_final_to_s3(session_id,safe_name,final_path)
        return {"ok":True,"public_url":url}
    finally:
        for p in tmp_segments: p.unlink(missing_ok=True)

def analyze_core_from_session(session_id:str, script_text:Optional[str]=None)->Dict[str,Any]:
    files=_jget(_session_key(session_id)) or {}
    if not files: raise RuntimeError("session not found")
    work_dir=TMP_ROOT/session_id; work_dir.mkdir(parents=True,exist_ok=True)
    cache_dir=work_dir/"cache"; cache_dir.mkdir(exist_ok=True)
    results={}
    for fid,url in files.items():
        cache_name=f"{fid}.src"; src_path=cache_dir/cache_name
        if not src_path.exists(): _download_to(cache_dir,url,cache_name)
        dur=_probe_duration(src_path); utts=_detect_utterances(src_path)
        score=0.8 if dur>=1.5 else 0.4
        results[fid]={"duration":dur,"score":score,"utterances":utts}
    _jset(_analysis_key(session_id),{"results":results},ex=86400)
    return {"ok":True,"analyzed":len(results)}

# ------------------------------------------------------------------------------
# Manifest builders
# ------------------------------------------------------------------------------

def _load_analysis_for_choose(session_id:str)->Dict[str,Dict[str,Any]]:
    analysis=_jget(_analysis_key(session_id)) or {}
    results=analysis.get("results") or {}
    return {fid:{
        "score":float(info.get("score",0.5)),
        "duration":float(info.get("duration",2.0)),
        "utterances":info.get("utterances") or []
    } for fid,info in results.items()}

def _choose_adaptive_segments(session_id:str,preset_key:str)->Tuple[List[Dict[str,Any]],Dict[str,Any]]:
    """
    Speech-adaptive: use utterances to reach each slot's target length.
    If no utterances for a file, fallback to that file's real duration.
    """
    presets=_load_presets()
    if preset_key not in presets: raise HTTPException(400,f"Unknown preset {preset_key}")
    slots=presets[preset_key]
    files=_jget(_session_key(session_id)) or {}
    if not files: raise HTTPException(404,"session not found")

    analysis=_load_analysis_for_choose(session_id)
    # Rank files: better score + longer duration first
    ranked=sorted(analysis.items(), key=lambda kv:(kv[1]["score"], kv[1]["duration"]), reverse=True)
    file_ptrs={fid:0 for fid in analysis.keys()}

    selection={}; segs=[]
    for slot in slots:
        target=float(slot.get("target",0.0))
        chosen=None

        # Pass 1: try to assemble from utterances
        for fid,meta in ranked:
            utts=meta["utterances"]; p=file_ptrs[fid]
            if not utts or p>=len(utts): continue
            picks=[]; total=0.0
            while p<len(utts):
                u=utts[p]; ulen=float(max(0.0,u["end"]-u["start"]))
                if ulen<=0.01: p+=1; continue
                if target>0 and total+ulen<=target+0.75:
                    picks.append(u); total+=ulen; p+=1
                    if total>=max(0.5,target-0.25): break
                else:
                    if not picks: picks.append(u); total+=ulen; p+=1
                    break
            file_ptrs[fid]=p
            if picks:
                s=float(picks[0]["start"]); e=float(picks[-1]["end"])
                chosen={"file_id":fid,"start":s,"end":e}
                break

        # Pass 2: fallback to full file duration if no utterances used
        if not chosen and ranked:
            fid,meta=ranked[0]
            dur=float(meta.get("duration",2.0))
            if dur<=0.05: dur=2.0
            # If the slot has a target, cap at min(dur, target+0.75) but never below 2.0
            wanted = target+0.75 if target>0 else min(dur, UTT_MAX_D)
            end = float(min(dur, max(2.0, wanted)))
            chosen={"file_id":fid,"start":0.0,"end":end}

        segs.append(chosen); selection[slot["name"]]=chosen
    return segs,selection

# --- Legacy simple choose_best (kept for old clients) --------------------------

def choose_best_core(session_id: str, target_sec: int, max_segments_per_file: int, fps: int, scale: int) -> Dict[str, Any]:
    files = _jget(_session_key(session_id)) or {}
    if not files:
        raise RuntimeError("session not found")
    analysis = _jget(_analysis_key(session_id)) or {}
    scored = []
    for fid, url in files.items():
        info = (analysis.get("results") or {}).get(fid) or {}
        score = float(info.get("score", 0.5))
        dur   = float(info.get("duration", 2.0))
        scored.append((fid, score, dur))
    scored.sort(key=lambda t: t[1], reverse=True)
    remaining = float(target_sec)
    segments = []
    for fid, _score, dur in scored:
        if remaining <= 0:
            break
        count = max_segments_per_file
        while count > 0 and remaining > 0:
            piece = min(2.4, remaining, max(0.4, dur))
            segments.append({"file_id": fid, "start": 0.0, "end": float(min(piece, dur))})
            remaining -= float(piece)
            count -= 1
    manifest = {"segments": segments, "fps": fps, "scale": scale}
    _jset(_manifest_key(session_id), manifest, ex=60*60*24)
    return {"ok": True, "manifest": manifest}

# ------------------------------------------------------------------------------
# FastAPI endpoints
# ------------------------------------------------------------------------------

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True, "service": "editdna-worker", "version": "1.6.3", "redis": True}

# Session: store URLs -----------------------------------------------------------

@app.post("/process_urls")
def process_urls(inp: ProcessUrlsIn):
    if not inp.urls:
        raise HTTPException(400, "urls required")
    session_id = uuid.uuid4().hex
    files_map = {uuid.uuid4().hex[:8]: url.strip() for url in inp.urls}
    _jset(_session_key(session_id), files_map, ex=60*60*24)
    return {
        "ok": True,
        "session_id": session_id,
        "files": [{"file_id": fid, "source": "url"} for fid in files_map.keys()]
    }

# Legacy Auto manifest (equal splitter) ----------------------------------------

@app.post("/automanifest")
def automanifest(inp: AutoManifestIn):
    files = _jget(_session_key(inp.session_id)) or {}
    if not files:
        raise HTTPException(404, "session not found")
    per = max(0.4, float(inp.max_total_sec) / max(1, len(files)))
    segments = []
    for fid in list(files.keys()):
        for _ in range(inp.max_segments_per_file):
            if len(segments) * per >= inp.max_total_sec:
                break
            segments.append({"file_id": fid, "start": 0.0, "end": float(min(2.4, per))})
    manifest = {"segments": segments, "fps": inp.fps, "scale": inp.scale}
    _jset(_manifest_key(inp.session_id), manifest, ex=60*60*24)
    return {"ok": True, "session_id": inp.session_id, "filename": inp.filename, "manifest": manifest}

# Analyze (build durations + utterances) ---------------------------------------

@app.post("/analyze")
def analyze(inp: AnalyzeIn):
    job: Job = q.enqueue(
        analyze_core_from_session,
        inp.session_id,
        inp.script_text,
        job_timeout=60*30,
        retry=Retry(max=3, interval=[30, 90, 180]),
    )
    _jset(_jobs_key(job.id), {"type": "analyze", "session_id": inp.session_id, "created_at": int(time.time())}, ex=60*60*12)
    return {"ok": True, "job_id": job.id}

# Legacy choose_best ------------------------------------------------------------

@app.post("/choose_best")
def choose_best(inp: ChooseBestIn):
    job: Job = q.enqueue(
        choose_best_core,
        inp.session_id, inp.target_sec, inp.max_segments_per_file, inp.fps, inp.scale,
        job_timeout=60*30,
        retry=Retry(max=3, interval=[30, 90, 180]),
    )
    _jset(_jobs_key(job.id), {"type": "choose_best", "session_id": inp.session_id, "created_at": int(time.time())}, ex=60*60*12)
    return {"ok": True, "job_id": job.id}

# New speech-adaptive manifest (with real-duration fallback) --------------------

@app.post("/manifest")
def manifest(inp: BuildManifestIn):
    files=_jget(_session_key(inp.session_id)) or {}
    if not files: raise HTTPException(404,"session not found")

    if inp.segments and len(inp.segments)>0:
        segs=[{"file_id":s["file_id"],"start":float(s.get("start",0.0)),"end":float(s.get("end",1.0))} for s in inp.segments]
        debug={"mode":"segments_supplied"}
    else:
        # Build adaptively; if no utterances, fallback to full-file durations
        segs, sel = _choose_adaptive_segments(inp.session_id, inp.preset_key)
        debug={"mode":"adaptive","selection":sel}

    # Safety: clamp any end < start or tiny segments
    fixed=[]
    for s in segs:
        st=float(s.get("start",0.0)); en=float(s.get("end",0.0))
        if en-st < 0.25: en = st + 0.25
        fixed.append({"file_id":s["file_id"],"start":st,"end":en})

    manifest={"segments":fixed,"fps":inp.fps,"scale":inp.scale,"filename":inp.filename,"preset_key":inp.preset_key}
    _jset(_manifest_key(inp.session_id),manifest,ex=86400)
    return {"ok":True,"session_id":inp.session_id,"manifest":manifest,"debug":debug}

# Stitch (uses last saved manifest) --------------------------------------------

@app.post("/stitch")
def stitch(inp: StitchIn):
    manifest=_jget(_manifest_key(inp.session_id))
    if not manifest: raise HTTPException(400,"No manifest stored for this session. Call /manifest (or /automanifest/choose_best) first.")
    filename=_safe_name(inp.filename or manifest.get("filename") or "final.mp4")
    fps=int(manifest.get("fps",30)); scale=int(manifest.get("scale",720))
    job: Job = q.enqueue(
        stitch_core, inp.session_id, filename, manifest, fps, scale,
        job_timeout=60*60,
        retry=Retry(max=3, interval=[30, 90, 180]),
    )
    _jset(_jobs_key(job.id),{"type":"stitch","session_id":inp.session_id,"filename":filename,"manifest":manifest,"created_at":int(time.time())},ex=43200)
    return {"ok":True,"job_id":job.id,"session_id":inp.session_id}

# Stitch with provided manifest (async) -----------------------------------------

@app.post("/stitch_async")
def stitch_async(inp: StitchAsyncIn):
    manifest=inp.manifest or _jget(_manifest_key(inp.session_id))
    if not manifest: raise HTTPException(400,"manifest required (or call /manifest /automanifest /choose_best first)")
    job: Job = q.enqueue(
        stitch_core, inp.session_id, _safe_name(inp.filename), manifest,
        int(manifest.get("fps", inp.fps)), int(manifest.get("scale", inp.scale)),
        job_timeout=60*60,
        retry=Retry(max=3, interval=[30, 90, 180]),
    )
    _jset(_jobs_key(job.id),{"type":"stitch","session_id":inp.session_id,"filename":inp.filename,"manifest":manifest,"created_at":int(time.time())},ex=43200)
    return {"ok":True,"job_id":job.id}

# Jobs --------------------------------------------------------------------------

@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    try:
        job = Job.fetch(job_id, connection=r_conn)
    except Exception:
        meta = _jget(_jobs_key(job_id)) or {}
        return {"job_id": job_id, "status": "queued", **({"meta": meta} if meta else {})}
    status = job.get_status()
    resp: Dict[str, Any] = {"job_id": job_id, "status": status}
    if status == "finished":
        resp["result"] = job.result
    elif status == "failed":
        resp["result"] = None
        tb = getattr(job, "exc_info", None)
        if tb:
            resp["error"] = tb.splitlines()[-1] if isinstance(tb, str) else "failed"
    return resp

# Download (if file still on disk) ---------------------------------------------

@app.get("/download/{session_id}/{filename}")
def download_local(session_id: str, filename: str):
    path = (TMP_ROOT / session_id / _safe_name(filename)).resolve()
    if not path.exists():
        raise HTTPException(404, "file not found")
    def _iter():
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk
    return StreamingResponse(_iter(), media_type="video/mp4")

@app.get("/")
def root():
    return PlainTextResponse("Not Found", status_code=404)
