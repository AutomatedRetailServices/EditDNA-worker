# editdna/jobs.py
# Full, self-contained pipeline used by tasks.job_render
# - ASR (optional, Whisper) -> takes
# - Sentence/micro-silence boundary cleanup (optional)
# - Semantic pass (optional) -> tagging & stitching
# - Funnel assembly by FUNNEL_COUNTS
# - ffmpeg subclip + concat
# - S3 upload with boto3
# - Returns JSON compatible with your web API

from __future__ import annotations
import os, io, json, math, tempfile, subprocess, shlex, uuid, time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

# =============== ENV ===============
FFMPEG_BIN   = os.getenv("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE_BIN  = os.getenv("FFPROBE_BIN", "/usr/bin/ffprobe")

def _get_float(name: str, default_str: str) -> float:
    try:
        return float(os.getenv(name, default_str).strip())
    except Exception:
        # if env var is dirty (has comments, spaces, whatever), try first token
        raw = os.getenv(name, default_str)
        if isinstance(raw, str):
            first_token = raw.strip().split()[0]
            return float(first_token)
        return float(default_str)

def _get_int(name: str, default_val: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default_val
    try:
        return int(raw.strip().split()[0])
    except Exception:
        return default_val

BIN_SEC          = _get_float("BIN_SEC",          "1.0")
MIN_TAKE_SEC     = _get_float("MIN_TAKE_SEC",     "2.0")
MAX_TAKE_SEC     = _get_float("MAX_TAKE_SEC",     "20")
MAX_DURATION_SEC = _get_float("MAX_DURATION_SEC", "120")
FALLBACK_MIN_SEC = _get_float("FALLBACK_MIN_SEC", "60")

ASR_ENABLED   = os.getenv("ASR_ENABLED", "1").strip() == "1"
ASR_MODEL     = os.getenv("ASR_MODEL_SIZE", "tiny").strip()
ASR_LANG      = os.getenv("ASR_LANGUAGE", "en").strip()
ASR_DEVICE    = os.getenv("ASR_DEVICE", "cpu").strip()

SEM_ENABLED   = os.getenv("SEMANTICS_ENABLED", "0").strip() == "1"

# sentence/micro settings
MICRO_CUT         = os.getenv("MICRO_CUT", "0").strip() == "1"
MICRO_SILENCE_DB  = _get_float("MICRO_SILENCE_DB",  "-30")
MICRO_SILENCE_MIN = _get_float("MICRO_SILENCE_MIN", "0.25")

# scoring/requirements used by semantic pass (if present)
SLOT_REQUIRE_PRODUCT = set((os.getenv("SLOT_REQUIRE_PRODUCT", "") or "").split(",")) - {""}
SLOT_REQUIRE_OCR_CTA = set((os.getenv("SLOT_REQUIRE_OCR_CTA", "") or "").split(",")) - {""}

# Funnel layout: "H,P,FEAT,PROOF,CTA"
FUNNEL_COUNTS = os.getenv("FUNNEL_COUNTS", "1,3,3,3,1").strip()

# S3
S3_BUCKET  = (os.getenv("S3_BUCKET") or "").strip()
S3_REGION  = os.getenv("AWS_REGION", "us-east-1").strip()
S3_PREFIX  = (os.getenv("S3_PREFIX") or "editdna/outputs").strip().strip("/")
S3_ACL     = os.getenv("S3_ACL", "public-read").strip()
PRESIGN_EXPIRES = _get_int("PRESIGN_EXPIRES", 100000)

# =============== Optional imports ===============
_Take = None
def _import_semantic():
    """
    Try to import semantic_visual_pass pieces.
    If not available, we just fall back to dummy scoring/tagging.
    """
    global _Take
    try:
        from worker.semantic_visual_pass import (
            Take,
            dedup_takes,
            tag_slot,
            score_take,
            stitch_chain,
        )
        _Take = Take
        return Take, dedup_takes, tag_slot, score_take, stitch_chain
    except Exception:
        _Take = None
        return None, None, None, None, None

def _import_sentence_boundary():
    """
    Try to import sentence_boundary.split_by_sentence (optional).
    """
    try:
        from worker.sentence_boundary import split_by_sentence
        return split_by_sentence
    except Exception:
        return None

Take, dedup_takes, tag_slot, score_take, stitch_chain = _import_semantic()
split_by_sentence = _import_sentence_boundary()

# =============== Data model (fallback Take) ===============
@dataclass
class _FallbackTake:
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
    meta: Dict[str, Any] = None

    def to_take(self):
        if Take is None:
            return self
        return Take(
            id=self.id,
            start=self.start,
            end=self.end,
            text=self.text,
            face_q=self.face_q,
            scene_q=self.scene_q,
            vtx_sim=self.vtx_sim,
            has_product=self.has_product,
            ocr_hit=self.ocr_hit,
            slot_hint=self.slot_hint,
            meta=self.meta or {},
        )

# =============== Utilities ===============
def _run(cmd: List[str], check=True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )

def _ffprobe_duration(path: str) -> float:
    try:
        proc = _run(
            [
                FFPROBE_BIN,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                path,
            ]
        )
        return float(proc.stdout.decode().strip())
    except Exception:
        return 0.0

def _ffmpeg_subclip(src: str, dst: str, ss: float, ee: float):
    dur = max(0.01, ee - ss)
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-ss",
        f"{ss:.3f}",
        "-i",
        src,
        "-t",
        f"{dur:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-g",
        "48",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        dst,
    ]
    _run(cmd)

def _ffmpeg_concat(parts_list_path: str, out_path: str):
    _run(
        [
            FFMPEG_BIN,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            parts_list_path,
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "48",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            out_path,
        ]
    )

# =============== ASR ===============
def _do_whisper_asr(local_path: str) -> List[Dict[str, Any]]:
    """
    Returns [{'start': float, 'end': float, 'text': str}, ...]
    """
    import whisper  # openai-whisper
    model = whisper.load_model(ASR_MODEL, device=ASR_DEVICE)
    result = model.transcribe(local_path, language=ASR_LANG)
    segs = []
    for i, s in enumerate(result.get("segments", []), 1):
        segs.append(
            {
                "id": f"T{i:04d}",
                "start": float(s["start"]),
                "end": float(s["end"]),
                "text": s.get("text", "").strip(),
            }
        )
    return segs

# =============== Takes building ===============
def _build_takes(local_path: str) -> List[_FallbackTake]:
    takes: List[_FallbackTake] = []
    if ASR_ENABLED:
        segs = _do_whisper_asr(local_path)

        # sentence boundary refinement (optional)
        if split_by_sentence is not None:
            segs = split_by_sentence(
                segs,
                min_take_sec=MIN_TAKE_SEC,
                max_take_sec=MAX_TAKE_SEC,
            )

        # min/max clamp
        out = []
        for s in segs:
            st = float(s["start"])
            en = float(s["end"])
            if en - st < MIN_TAKE_SEC:
                en = st + MIN_TAKE_SEC
            if en - st > MAX_TAKE_SEC:
                en = st + MAX_TAKE_SEC
            out.append(
                _FallbackTake(
                    id=s["id"],
                    start=st,
                    end=en,
                    text=s.get("text", ""),
                    meta={"src": "asr"},
                )
            )
        takes = out
    else:
        # fallback: time bins
        dur = _ffprobe_duration(local_path) or FALLBACK_MIN_SEC
        step = max(MIN_TAKE_SEC, BIN_SEC)
        cursor = 0.0
        idx = 1
        while cursor < dur:
            st = cursor
            en = min(dur, st + step)
            takes.append(
                _FallbackTake(
                    id=f"T{idx:04d}",
                    start=st,
                    end=en,
                    text="",
                    meta={"src": "bins"},
                )
            )
            cursor = en
            idx += 1
    return takes

# =============== Semantic tagging (optional) ===============
SLOT_ORDER = ["HOOK", "PROBLEM", "FEATURE", "PROOF", "CTA"]

def _tag_and_stitch(takes: List[_FallbackTake]) -> Tuple[List[dict], Dict[str, List[dict]]]:
    """
    Returns (clips_for_response, slots_dict_for_response)
    """
    # If semantic module is present:
    if Take is not None:
        sem_takes = [t.to_take() for t in takes]

        if callable(dedup_takes):
            sem_takes = dedup_takes(sem_takes)

        if callable(stitch_chain):
            sem_takes = stitch_chain(sem_takes)

        clips = []
        slots: Dict[str, List[dict]] = {k: [] for k in SLOT_ORDER}

        for t in sem_takes:
            slot = "HOOK"
            if callable(tag_slot):
                slot = tag_slot(t, None) or "HOOK"

            sc = 2.5
            if callable(score_take):
                sc = score_take(t, slot)

            clips.append(
                {
                    "id": t.id,
                    "slot": slot,
                    "start": t.start,
                    "end": t.end,
                    "score": sc,
                    "face_q": getattr(t, "face_q", 1.0),
                    "scene_q": getattr(t, "scene_q", 1.0),
                    "vtx_sim": getattr(t, "vtx_sim", 0.0),
                    "chain_ids": getattr(t, "meta", {}).get("chain_ids", []),
                }
            )

            slots[slot].append(
                {
                    "id": t.id,
                    "start": t.start,
                    "end": t.end,
                    "text": t.text,
                    "meta": {"slot": slot, "score": sc},
                    "face_q": getattr(t, "face_q", 1.0),
                    "scene_q": getattr(t, "scene_q", 1.0),
                    "vtx_sim": getattr(t, "vtx_sim", 0.0),
                    "has_product": getattr(t, "has_product", False),
                    "ocr_hit": getattr(t, "ocr_hit", 0),
                }
            )

        return clips, slots

    # No semantic module → treat all as HOOK
    clips = []
    slots = {k: [] for k in SLOT_ORDER}
    for t in takes:
        slot = "HOOK"
        sc = 2.5
        clips.append(
            {
                "id": t.id,
                "slot": slot,
                "start": t.start,
                "end": t.end,
                "score": sc,
                "face_q": 1.0,
                "scene_q": 1.0,
                "vtx_sim": 0.0,
                "chain_ids": [],
            }
        )
        slots[slot].append(
            {
                "id": t.id,
                "start": t.start,
                "end": t.end,
                "text": t.text,
                "meta": {"slot": slot, "score": sc},
                "face_q": 1.0,
                "scene_q": 1.0,
                "vtx_sim": 0.0,
                "has_product": False,
                "ocr_hit": 0,
            }
        )

    return clips, slots

# =============== Funnel assembly ===============
def _parse_funnel_counts(raw: str) -> Dict[str, Optional[int]]:
    # "1,3,3,3,1" → dict
    xs = [x.strip() for x in raw.split(",")]
    while len(xs) < 5:
        xs.append("0")

    def cap(v: str):
        # "0" means "unlimited" in our world
        if v in ("", "0", "00", "000"):
            return None
        try:
            n = int(v)
            return None if n < 0 else n
        except Exception:
            return None

    caps = list(map(cap, xs[:5]))
    return dict(zip(SLOT_ORDER, caps))

def _assemble_funnel(
    slots: Dict[str, List[dict]],
    max_total: float,
) -> List[Tuple[float, float, str, str]]:
    """
    Pick clips in slot order (HOOK→PROBLEM→FEATURE→PROOF→CTA)
    respecting slot caps and max_total seconds.
    Returns list of (start, end, clip_id, slot)
    """
    caps = _parse_funnel_counts(FUNNEL_COUNTS)

    plan: List[Tuple[float, float, str, str]] = []
    used = set()
    total = 0.0

    for slot in SLOT_ORDER:
        cap = caps[slot]  # None = unlimited
        picks = 0
        for c in slots.get(slot, []):
            if cap is not None and picks >= cap:
                break

            dur = max(0.01, float(c["end"]) - float(c["start"]))
            if total + dur > max_total:
                break

            cid = c["id"]
            if cid in used:
                continue

            used.add(cid)
            plan.append((float(c["start"]), float(c["end"]), cid, slot))
            total += dur
            picks += 1

    # fallback: if plan is empty, pick the single longest HOOK/whatever
    if not plan:
        longest = None
        for slot in SLOT_ORDER:
            for c in slots.get(slot, []):
                d = float(c["end"]) - float(c["start"])
                if longest is None or d > longest[0]:
                    longest = (d, c)
        if longest:
            c = longest[1]
            plan = [
                (
                    float(c["start"]),
                    float(c["end"]),
                    c["id"],
                    "HOOK",
                )
            ]

    return plan

# =============== S3 upload ===============
def _s3_upload(local_path: str) -> Tuple[str, str, str]:
    """
    Uploads to s3://{bucket}/{prefix}/{file}
    Returns (key, s3_url, https_url)
    """
    import boto3
    import pathlib

    if not S3_BUCKET:
        # no bucket configured → just return local info
        name = pathlib.Path(local_path).name
        key = f"{S3_PREFIX}/{name}"
        return key, f"s3://(no-bucket)/{key}", f"file://{local_path}"

    s3 = boto3.client("s3", region_name=S3_REGION)

    base = f"{uuid.uuid4().hex}_{int(time.time())}.mp4"
    key = f"{S3_PREFIX}/{base}"

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
    s3_url = f"s3://{S3_BUCKET}/{key}"
    return key, s3_url, https_url

# =============== Main render ===============
def render_funnel(local_video: str) -> Tuple[str, List[dict], Dict[str, List[dict]]]:
    """
    Returns (out_path, clips_for_response, slots_for_response)
    """
    takes = _build_takes(local_video)

    clips, slots = _tag_and_stitch(takes)

    # pick clips for funnel
    plan = _assemble_funnel(slots, max_total=MAX_DURATION_SEC)

    # cut subclips
    tmpdir = tempfile.mkdtemp(prefix="ed_")
    parts: List[str] = []

    for i, (ss, ee, cid, slot) in enumerate(plan, 1):
        outp = os.path.join(tmpdir, f"part{i:02d}.mp4")
        _ffmpeg_subclip(local_video, outp, ss, ee)
        parts.append(outp)

    # if nothing, force 1s clip from start
    if not parts:
        outp = os.path.join(tmpdir, "part01.mp4")
        _ffmpeg_subclip(local_video, outp, 0.0, min(1.0, _ffprobe_duration(local_video)))
        parts = [outp]

    # concat list file
    concat_list = os.path.join(tmpdir, "concat.txt")
    with open(concat_list, "w") as f:
        for p in parts:
            f.write(f"file '{p}'\n")

    out_final = os.path.join(tmpdir, f"final_{uuid.uuid4().hex}.mp4")
    _ffmpeg_concat(concat_list, out_final)

    return out_final, clips, slots

# =============== Download helper ===============
def _download_to_tmp(url: str) -> str:
    import requests
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=".mp4")
    with os.fdopen(fd, "wb") as w:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                w.write(chunk)
    return path

# =============== Public API ===============
def run_pipeline(local_path: Optional[str] = None, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Main callable used by the worker if tasks.job_render forwards here.
    payload (optional) can override env via payload["options"].
    """
    payload = payload or {}
    options = payload.get("options") or {}

    # allow request-time overrides
    global MAX_DURATION_SEC, FUNNEL_COUNTS
    if "MAX_DURATION_SEC" in options:
        try:
            MAX_DURATION_SEC = float(str(options["MAX_DURATION_SEC"]).strip().split()[0])
        except Exception:
            pass
    if "FUNNEL_COUNTS" in options:
        FUNNEL_COUNTS = str(options["FUNNEL_COUNTS"]).strip()

    # resolve local input
    input_local = local_path
    if not input_local:
        files = payload.get("files") or []
        if files:
            src = files[0]
            if isinstance(src, dict) and "url" in src:
                src = src["url"]
            if isinstance(src, str) and src.startswith("http"):
                input_local = _download_to_tmp(src)
            else:
                input_local = src

    if not input_local:
        return {"ok": False, "error": "No input file provided."}

    # render pipeline
    out_path, clips, slots = render_funnel(input_local)

    # upload
    key, s3_url, https_url = _s3_upload(out_path)
    dur = _ffprobe_duration(out_path)

    return {
        "ok": True,
        "input_local": input_local,
        "duration_sec": round(dur, 3),
        "s3_key": key,
        "s3_url": s3_url,
        "https_url": https_url,
        "clips": clips,
        "slots": slots,
        "semantic": bool(SEM_ENABLED and Take is not None),
        "vision": False,
        "asr": bool(ASR_ENABLED),
    }

# --- Adapter for the worker entrypoint ---
def job_render(payload=None):
    """
    RQ calls this (string 'tasks.job_render' in Redis),
    our tasks.py forwards to us.
    """
    if not isinstance(payload, dict):
        payload = {}
    local_path = payload.get("local_path")
    if not local_path:
        files = payload.get("files") or []
        if files:
            src = files[0]
            if isinstance(src, dict) and "url" in src:
                src = src["url"]
            if isinstance(src, str) and src.startswith("http"):
                local_path = _download_to_tmp(src)
            else:
                local_path = src
    return run_pipeline(local_path=local_path, payload=payload)
