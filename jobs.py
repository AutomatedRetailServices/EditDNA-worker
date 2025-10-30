"""
jobs.py
Full funnel pipeline:
- ASR (optional)
- sentence boundary / microcuts (optional)
- semantic tagging (optional)
- funnel assembly
- ffmpeg subclips + concat
- S3 upload
Returns dict { ok, https_url, clips, slots, ... }

This file MUST define run_pipeline() or job_render() so tasks.py can call us.
"""

from __future__ import annotations
import os, uuid, time, tempfile, subprocess, math
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

########################################
# SAFE ENV HELPERS (no crashes on junk)
########################################

def _env_float(name: str, default_val: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default_val
    # Try first token only (ignore inline comments)
    token = raw.strip().split()[0]
    try:
        return float(token)
    except Exception:
        return default_val

def _env_int(name: str, default_val: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default_val
    token = raw.strip().split()[0]
    try:
        return int(token)
    except Exception:
        return default_val

def _env_str(name: str, default_val: str) -> str:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default_val
    return raw.strip()

########################################
# ENV DEFAULTS
########################################

FFMPEG_BIN   = _env_str("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE_BIN  = _env_str("FFPROBE_BIN", "/usr/bin/ffprobe")

BIN_SEC          = _env_float("BIN_SEC",          1.0)
MIN_TAKE_SEC     = _env_float("MIN_TAKE_SEC",     2.0)
MAX_TAKE_SEC     = _env_float("MAX_TAKE_SEC",     20.0)
MAX_DURATION_SEC = _env_float("MAX_DURATION_SEC", 120.0)
FALLBACK_MIN_SEC = _env_float("FALLBACK_MIN_SEC", 60.0)

ASR_ENABLED   = (_env_str("ASR_ENABLED", "1") == "1")
ASR_MODEL     = _env_str("ASR_MODEL_SIZE", "tiny")
ASR_LANG      = _env_str("ASR_LANGUAGE", "en")
ASR_DEVICE    = _env_str("ASR_DEVICE", "cpu")

SEMANTICS_ENABLED = (_env_str("SEMANTICS_ENABLED", "0") == "1")

MICRO_CUT         = (_env_str("MICRO_CUT", "0") == "1")
MICRO_SILENCE_DB  = _env_float("MICRO_SILENCE_DB",  -30.0)
MICRO_SILENCE_MIN = _env_float("MICRO_SILENCE_MIN", 0.25)

FUNNEL_COUNTS = _env_str("FUNNEL_COUNTS", "1,3,3,3,1")

S3_BUCKET  = _env_str("S3_BUCKET", "")
S3_REGION  = _env_str("AWS_REGION", "us-east-1")
S3_PREFIX  = _env_str("S3_PREFIX", "editdna/outputs").strip("/")
S3_ACL     = _env_str("S3_ACL", "public-read")
PRESIGN_EXPIRES = _env_int("PRESIGN_EXPIRES", 100000)

########################################
# LIGHT DATA CLASS FOR TAKES
########################################

@dataclass
class BasicTake:
    id: str
    start: float
    end: float
    text: str = ""
    face_q: float = 1.0
    scene_q: float = 1.0
    vtx_sim: float = 0.0
    has_product: bool = False
    ocr_hit: int = 0

########################################
# LITTLE SHELL HELPERS
########################################

def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

def _ffprobe_duration(path: str) -> float:
    try:
        out = _run([
            FFPROBE_BIN,
            "-v","error",
            "-show_entries","format=duration",
            "-of","default=nokey=1:noprint_wrappers=1",
            path
        ])
        return float(out.stdout.decode().strip())
    except Exception:
        return 0.0

def _ffmpeg_subclip(src: str, dst: str, ss: float, ee: float):
    cut_dur = max(0.01, ee-ss)
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-ss", f"{ss:.3f}",
        "-i", src,
        "-t", f"{cut_dur:.3f}",
        "-c:v","libx264",
        "-preset","fast",
        "-crf","23",
        "-pix_fmt","yuv420p",
        "-g","48",
        "-c:a","aac",
        "-b:a","128k",
        dst
    ]
    _run(cmd)

def _ffmpeg_concat(list_file: str, out_file: str):
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-f","concat",
        "-safe","0",
        "-i", list_file,
        "-c:v","libx264",
        "-preset","fast",
        "-crf","23",
        "-pix_fmt","yuv420p",
        "-g","48",
        "-c:a","aac",
        "-b:a","128k",
        out_file
    ]
    _run(cmd)

########################################
# OPTIONAL IMPORTS, LOADED *INSIDE* FUNCTIONS
########################################

def _maybe_import_sentence_boundary():
    try:
        from worker.sentence_boundary import split_by_sentence
        return split_by_sentence
    except Exception:
        return None

def _maybe_import_semantic():
    try:
        from worker.semantic_visual_pass import (
            Take,
            dedup_takes,
            tag_slot,
            score_take,
            stitch_chain,
        )
        return (Take, dedup_takes, tag_slot, score_take, stitch_chain)
    except Exception:
        return (None, None, None, None, None)

########################################
# ASR STEP
########################################

def _whisper_asr(local_path: str) -> List[Dict[str, Any]]:
    """
    returns list of {id,start,end,text}
    """
    import whisper
    model = whisper.load_model(ASR_MODEL, device=ASR_DEVICE)
    result = model.transcribe(local_path, language=ASR_LANG)

    segs = []
    for i, seg in enumerate(result.get("segments", []), 1):
        segs.append({
            "id": f"T{i:04d}",
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": seg.get("text","").strip()
        })
    return segs

########################################
# BUILD TAKES
########################################

def _build_takes(local_path: str) -> List[BasicTake]:
    takes: List[BasicTake] = []

    if ASR_ENABLED:
        segs = _whisper_asr(local_path)

        split_by_sentence = _maybe_import_sentence_boundary()
        if split_by_sentence:
            segs = split_by_sentence(
                segs,
                min_take_sec=MIN_TAKE_SEC,
                max_take_sec=MAX_TAKE_SEC,
            )

        tmp_list = []
        for seg in segs:
            st = float(seg["start"])
            en = float(seg["end"])
            dur = en - st
            if dur < MIN_TAKE_SEC:
                en = st + MIN_TAKE_SEC
            if (en - st) > MAX_TAKE_SEC:
                en = st + MAX_TAKE_SEC

            tmp_list.append(
                BasicTake(
                    id=seg["id"],
                    start=st,
                    end=en,
                    text=seg.get("text",""),
                )
            )
        takes = tmp_list

    else:
        # fallback: dumb bins
        total = _ffprobe_duration(local_path)
        if total <= 0:
            total = FALLBACK_MIN_SEC
        step = max(MIN_TAKE_SEC, BIN_SEC)

        cur = 0.0
        idx = 1
        while cur < total:
            st = cur
            en = min(total, st + step)
            takes.append(
                BasicTake(
                    id=f"T{idx:04d}",
                    start=st,
                    end=en,
                    text="",
                )
            )
            cur = en
            idx += 1

    return takes

########################################
# SEMANTIC TAGGING + SLOTS
########################################

SLOT_ORDER = ["HOOK","PROBLEM","FEATURE","PROOF","CTA"]

def _tag_semantic(takes: List[BasicTake]) -> (List[dict], Dict[str, List[dict]]):
    """
    returns (clips, slots)
    """
    Take, dedup_takes, tag_slot, score_take, stitch_chain = _maybe_import_semantic()

    # if we cannot import semantic module, fallback: just HOOK
    if Take is None:
        clips = []
        slots: Dict[str, List[dict]] = {s: [] for s in SLOT_ORDER}
        for t in takes:
            clip = {
                "id": t.id,
                "slot": "HOOK",
                "start": t.start,
                "end": t.end,
                "score": 2.5,
                "face_q": t.face_q,
                "scene_q": t.scene_q,
                "vtx_sim": t.vtx_sim,
                "chain_ids": []
            }
            clips.append(clip)

            slots["HOOK"].append({
                "id": t.id,
                "start": t.start,
                "end": t.end,
                "text": t.text,
                "meta": {"slot":"HOOK","score":2.5},
                "face_q": t.face_q,
                "scene_q": t.scene_q,
                "vtx_sim": t.vtx_sim,
                "has_product": t.has_product,
                "ocr_hit": t.ocr_hit
            })
        return clips, slots

    # else full semantic path
    sem_takes = [Take(
        id=t.id,
        start=t.start,
        end=t.end,
        text=t.text,
        face_q=t.face_q,
        scene_q=t.scene_q,
        vtx_sim=t.vtx_sim,
        has_product=t.has_product,
        ocr_hit=t.ocr_hit,
        meta={}
    ) for t in takes]

    if callable(dedup_takes):
        sem_takes = dedup_takes(sem_takes)

    if callable(stitch_chain):
        sem_takes = stitch_chain(sem_takes)

    clips: List[dict] = []
    slots: Dict[str, List[dict]] = {s: [] for s in SLOT_ORDER}

    for tt in sem_takes:
        slot_guess = "HOOK"
        if callable(tag_slot):
            try:
                slot_guess = tag_slot(tt, None) or "HOOK"
            except Exception:
                slot_guess = "HOOK"

        sc_val = 2.5
        if callable(score_take):
            try:
                sc_val = score_take(tt, slot_guess)
            except Exception:
                sc_val = 2.5

        clip_entry = {
            "id": tt.id,
            "slot": slot_guess,
            "start": tt.start,
            "end": tt.end,
            "score": sc_val,
            "face_q": getattr(tt,"face_q",1.0),
            "scene_q": getattr(tt,"scene_q",1.0),
            "vtx_sim": getattr(tt,"vtx_sim",0.0),
            "chain_ids": getattr(tt,"meta",{}).get("chain_ids",[])
        }
        clips.append(clip_entry)

        slots[slot_guess].append({
            "id": tt.id,
            "start": tt.start,
            "end": tt.end,
            "text": tt.text,
            "meta": {"slot":slot_guess,"score":sc_val},
            "face_q": getattr(tt,"face_q",1.0),
            "scene_q": getattr(tt,"scene_q",1.0),
            "vtx_sim": getattr(tt,"vtx_sim",0.0),
            "has_product": getattr(tt,"has_product",False),
            "ocr_hit": getattr(tt,"ocr_hit",0)
        })

    return clips, slots

########################################
# FUNNEL ASSEMBLY
########################################

def _parse_funnel(raw: str) -> Dict[str, Optional[int]]:
    # "1,3,3,3,1" -> {"HOOK":1,"PROBLEM":3,"FEATURE":3,"PROOF":3,"CTA":1}
    parts = [p.strip() for p in raw.split(",")]
    while len(parts) < 5:
        parts.append("0")

    def _cap(x: str) -> Optional[int]:
        # "0" or negative or "" => unlimited
        if x == "" or x.startswith("0") or x.startswith("-"):
            return None
        try:
            n = int(x)
            if n <= 0:
                return None
            return n
        except:
            return None

    nums = list(map(_cap, parts[:5]))
    return dict(zip(SLOT_ORDER, nums))

def _choose_plan(slots: Dict[str, List[dict]], max_total_sec: float) -> List[Tuple[float,float,str,str]]:
    caps = _parse_funnel(FUNNEL_COUNTS)

    chosen: List[Tuple[float,float,str,str]] = []
    used_ids = set()
    total = 0.0

    for slot in SLOT_ORDER:
        limit = caps[slot]  # None = unlimited
        count_slot = 0
        for c in slots.get(slot, []):
            if limit is not None and count_slot >= limit:
                break

            dur = max(0.01, float(c["end"]) - float(c["start"]))
            if total + dur > max_total_sec:
                break

            cid = c["id"]
            if cid in used_ids:
                continue

            chosen.append((float(c["start"]), float(c["end"]), cid, slot))
            used_ids.add(cid)
            total += dur
            count_slot += 1

    # fallback: nothing picked -> pick single longest from any slot
    if not chosen:
        longest = None
        for slot in SLOT_ORDER:
            for c in slots.get(slot, []):
                d = float(c["end"]) - float(c["start"])
                if longest is None or d > longest[0]:
                    longest = (d, c, slot)
        if longest:
            _, c, slot = longest
            chosen = [(float(c["start"]), float(c["end"]), c["id"], slot)]

    return chosen

########################################
# S3 UPLOAD
########################################

def _upload_s3(local_path: str) -> (str,str,str):
    """
    returns (key, s3_url, https_url)
    """
    import boto3
    if not S3_BUCKET:
        # no bucket? return file:// fallback
        base = os.path.basename(local_path)
        key = f"{S3_PREFIX}/{base}"
        return (key, f"s3://(no-bucket)/{key}", f"file://{local_path}")

    s3 = boto3.client("s3", region_name=S3_REGION)
    obj_name = f"{uuid.uuid4().hex}_{int(time.time())}.mp4"
    key = f"{S3_PREFIX}/{obj_name}"

    s3.upload_file(
        local_path,
        S3_BUCKET,
        key,
        ExtraArgs={
            "ACL": S3_ACL,
            "ContentType": "video/mp4",
        },
    )

    https_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{key}"
    s3_url   = f"s3://{S3_BUCKET}/{key}"
    return (key, s3_url, https_url)

########################################
# DOWNLOAD REMOTE FILE IF NEEDED
########################################

def _download_if_http(src: str) -> str:
    import requests, tempfile, os
    if src.startswith("http://") or src.startswith("https://"):
        r = requests.get(src, stream=True, timeout=60)
        r.raise_for_status()
        fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
        with os.fdopen(fd, "wb") as w:
            for chunk in r.iter_content(1024*1024):
                if chunk:
                    w.write(chunk)
        return tmp_path
    return src

########################################
# RENDER FUNNEL VIDEO
########################################

def _render_video(local_video: str) -> Dict[str, Any]:
    # 1. build takes
    takes = _build_takes(local_video)

    # 2. semantic tagging
    clips, slots = _tag_semantic(takes)

    # 3. choose funnel sequence
    plan = _choose_plan(slots, MAX_DURATION_SEC)

    # 4. cut & concat
    tmpdir = tempfile.mkdtemp(prefix="ed_")
    part_paths: List[str] = []

    for i, (ss, ee, cid, slot) in enumerate(plan, start=1):
        out_part = os.path.join(tmpdir, f"part{i:02d}.mp4")
        _ffmpeg_subclip(local_video, out_part, ss, ee)
        part_paths.append(out_part)

    # fallback: if somehow empty
    if not part_paths:
        out_part = os.path.join(tmpdir, "part01.mp4")
        _ffmpeg_subclip(local_video, out_part, 0.0, min(1.0, _ffprobe_duration(local_video)))
        part_paths = [out_part]

    concat_list = os.path.join(tmpdir, "concat.txt")
    with open(concat_list,"w") as f:
        for p in part_paths:
            f.write(f"file '{p}'\n")

    final_path = os.path.join(tmpdir, f"final_{uuid.uuid4().hex}.mp4")
    _ffmpeg_concat(concat_list, final_path)

    # 5. upload
    key, s3_url, https_url = _upload_s3(final_path)
    out_dur = _ffprobe_duration(final_path)

    return {
        "ok": True,
        "input_local": local_video,
        "duration_sec": round(out_dur,3),
        "s3_key": key,
        "s3_url": s3_url,
        "https_url": https_url,
        "clips": clips,
        "slots": slots,
        "semantic": SEMANTICS_ENABLED,
        "vision": False,
        "asr": ASR_ENABLED,
    }

########################################
# PUBLIC ENTRYPOINTS
########################################

def run_pipeline(local_path: Optional[str] = None, payload: Optional[Dict[str,Any]] = None) -> Dict[str,Any]:
    """
    Main function tasks.job_render will call.
    We ALSO honor runtime overrides in payload["options"].
    """
    payload = payload or {}
    options = payload.get("options") or {}

    # override MAX_DURATION_SEC / FUNNEL_COUNTS at request time
    global MAX_DURATION_SEC, FUNNEL_COUNTS
    if "MAX_DURATION_SEC" in options:
        try:
            MAX_DURATION_SEC = float(str(options["MAX_DURATION_SEC"]).strip().split()[0])
        except Exception:
            pass
    if "FUNNEL_COUNTS" in options:
        FUNNEL_COUNTS = str(options["FUNNEL_COUNTS"]).strip()

    # resolve input file
    if not local_path:
        files = payload.get("files") or []
        if files:
            first = files[0]
            if isinstance(first, dict) and "url" in first:
                first = first["url"]
            local_path = _download_if_http(str(first))

    if not local_path:
        return {"ok": False, "error": "No input provided."}

    return _render_video(local_path)

def job_render(payload: Dict[str,Any]) -> Dict[str,Any]:
    """
    Fallback signature, if tasks.py decides to call jobs.job_render directly.
    Just forwards to run_pipeline.
    """
    if not isinstance(payload, dict):
        payload = {}
    local_path = payload.get("local_path")
    return run_pipeline(local_path=local_path, payload=payload)
