import os
import uuid
import time
import tempfile
import subprocess
import json
from typing import List, Dict, Any, Tuple

from .semantic_visual_pass import semantic_visual_pass
from .s3_utils import download_url_to_tempfile


def _ffmpeg_extract_segment(src_path: str, start_sec: float, end_sec: float, out_path: str):
    """
    Cut [start_sec, end_sec] from src_path into out_path using ffmpeg.
    """
    dur = max(0.0, end_sec - start_sec)
    cmd = [
        os.getenv("FFMPEG_BIN", "/usr/bin/ffmpeg"),
        "-y",
        "-ss", f"{start_sec:.3f}",
        "-i", src_path,
        "-t", f"{dur:.3f}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-g", "48",
        "-c:a", "aac",
        "-b:a", "128k",
        out_path,
    ]
    subprocess.run(cmd, check=True)


def _ffmpeg_concat(parts_list_file: str, out_path: str):
    """
    Concat parts listed in parts_list_file (ffmpeg concat demuxer style).
    """
    cmd = [
        os.getenv("FFMPEG_BIN", "/usr/bin/ffmpeg"),
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", parts_list_file,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-g", "48",
        "-c:a", "aac",
        "-b:a", "128k",
        out_path,
    ]
    subprocess.run(cmd, check=True)


def _ffprobe_duration(path: str) -> float:
    """
    Return duration in seconds using ffprobe.
    """
    cmd = [
        os.getenv("FFPROBE_BIN", "/usr/bin/ffprobe"),
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nokey=1:noprint_wrappers=1",
        path,
    ]
    out = subprocess.check_output(cmd).decode("utf-8", "replace").strip()
    try:
        return float(out)
    except:
        return 0.0


def _pick_funnel_chunks(
    takes: List[Dict[str, Any]],
    funnel_counts: str,
    max_duration: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """
    takes = list of { "id","slot","start","end","text", ... }
    funnel_counts = "1,3,3,3,1" (HOOK,PROBLEM,FEATURE,PROOF,CTA)
    max_duration = hard cap final runtime in seconds

    We keep reading takes in order and greedily fill slots until we hit:
    - the requested count per slot
    - or the max_duration cap
    """

    # parse requested counts
    # default array length 5 -> [HOOK, PROBLEM, FEATURE, PROOF, CTA]
    default_counts = [1, 3, 3, 3, 1]
    parts = [p.strip() for p in funnel_counts.split(",") if p.strip()]
    want = []
    for i in range(5):
        try:
            want.append(int(parts[i]))
        except Exception:
            want.append(default_counts[i])

    SLOT_ORDER = ["HOOK", "PROBLEM", "FEATURE", "PROOF", "CTA"]
    needed_by_slot = {slot: want[idx] for idx, slot in enumerate(SLOT_ORDER)}

    used: List[Dict[str, Any]] = []
    per_slot: Dict[str, List[Dict[str, Any]]] = {s: [] for s in SLOT_ORDER}

    total_runtime = 0.0

    for tk in takes:
        slot = tk.get("slot", "HOOK").upper()
        if slot not in per_slot:
            slot = "HOOK"

        # skip if that slot already satisfied
        if needed_by_slot.get(slot, 0) <= 0:
            continue

        seg_len = float(tk["end"]) - float(tk["start"])
        if seg_len <= 0:
            continue

        # if adding this would explode max_duration, stop
        if total_runtime + seg_len > max_duration:
            break

        # accept
        used.append(tk)
        per_slot[slot].append(tk)
        needed_by_slot[slot] -= 1
        total_runtime += seg_len

        # small safety: if we filled all slots fully, we can stop
        if all(v <= 0 for v in needed_by_slot.values()):
            break

    # if we ended up with literally nothing (all filtered or too short),
    # fallback: grab at least something from the first take(s)
    if not used and takes:
        first = takes[0]
        used = [first]
        per_slot = {s: [] for s in SLOT_ORDER}
        per_slot[first.get("slot", "HOOK").upper()] = [first]

    return used, per_slot


def _stitch_parts(src_path: str, chosen: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Given chosen segments, actually cut them and concat them.
    Return (final_video_path, clips_meta_list)
    """
    tmpdir = tempfile.mkdtemp(prefix="ed_")
    part_files = []
    clip_meta_out = []

    for idx, seg in enumerate(chosen, start=1):
        part_path = os.path.join(tmpdir, f"part{idx:02d}.mp4")
        _ffmpeg_extract_segment(
            src_path=src_path,
            start_sec=seg["start"],
            end_sec=seg["end"],
            out_path=part_path,
        )
        part_files.append(part_path)
        clip_meta_out.append({
            "id": seg.get("id", f"T{idx:04d}"),
            "slot": seg.get("slot", "HOOK"),
            "start": seg["start"],
            "end": seg["end"],
            "score": seg.get("score", 2.5),
            "face_q": seg.get("face_q", 1.0),
            "scene_q": seg.get("scene_q", 1.0),
            "vtx_sim": seg.get("vtx_sim", 0.0),
            "chain_ids": seg.get("chain_ids", []),
        })

    # write concat list
    concat_list_path = os.path.join(tmpdir, "concat.txt")
    with open(concat_list_path, "w") as fh:
        for p in part_files:
            fh.write(f"file '{p}'\n")

    final_path = os.path.join(
        tmpdir,
        f"ed_{uuid.uuid4().hex[:32]}.mp4"
    )
    _ffmpeg_concat(concat_list_path, final_path)

    return final_path, clip_meta_out


def run_pipeline(
    session_id: str,
    file_urls: List[str],
    portrait: bool,
    funnel_counts: str,
    max_duration: int,
    bin_sec: float,
    min_take_sec: float,
    max_take_sec: float,
    veto_min_score: float,
    sem_merge_sim: float,
    viz_merge_sim: float,
    merge_max_chain: int,
    filler_tokens: List[str],
    filler_max_rate: float,
    micro_cut: bool,
    micro_silence_db: float,
    micro_silence_min: float,
    slot_require_product: List[str],
    slot_require_ocr_cta: str,
    fallback_min_sec: int,
) -> Dict[str, Any]:
    """
    Main pipeline.
    - Download the FIRST file URL only (MVP).
    - semantic_visual_pass() = YOUR intelligence pass (ASR, scoring, merge).
    - pick funnel.
    - ffmpeg stitch.
    """

    # 1) Download source video locally
    # (your s3_utils already has download_url_to_tempfile like we used before)
    src_local = download_url_to_tempfile(file_urls[0])  # returns /tmp/tmpabcd1234.mp4

    # 2) Run semantic / ASR / merge pass.
    # IMPORTANT:
    # semantic_visual_pass() is YOUR existing logic that was already working in the pod logs:
    # "🧠 [semantic_visual_pass] Semantic pipeline active."
    #
    # It MUST return a dict shaped like:
    # {
    #   "takes": [
    #       {
    #         "id": "T0001",
    #         "slot": "HOOK",
    #         "start": 0.00,
    #         "end": 2.96,
    #         "text": "...",
    #         "score": 2.5,
    #         "face_q": 1.0,
    #         "scene_q": 1.0,
    #         "vtx_sim": 0.0,
    #         "chain_ids": []
    #       },
    #       ...
    #   ]
    # }
    #
    sem_out = semantic_visual_pass(
        video_path=src_local,
        portrait=portrait,
        bin_sec=bin_sec,
        min_take_sec=min_take_sec,
        max_take_sec=max_take_sec,
        veto_min_score=veto_min_score,
        sem_merge_sim=sem_merge_sim,
        viz_merge_sim=viz_merge_sim,
        merge_max_chain=merge_max_chain,
        filler_tokens=filler_tokens,
        filler_max_rate=filler_max_rate,
        micro_cut=micro_cut,
        micro_silence_db=micro_silence_db,
        micro_silence_min=micro_silence_min,
        slot_require_product=slot_require_product,
        slot_require_ocr_cta=slot_require_ocr_cta,
        fallback_min_sec=fallback_min_sec,
    )

    takes = sem_out.get("takes", [])

    # 3) Pick final funnel slots with time budget
    chosen_segments, slot_map = _pick_funnel_chunks(
        takes=takes,
        funnel_counts=funnel_counts,
        max_duration=max_duration,
    )

    # 4) Actually stitch those chosen segments into a new mp4
    final_local, clips_meta = _stitch_parts(src_local, chosen_segments)

    # 5) Measure duration
    final_dur = _ffprobe_duration(final_local)

    # Reformat slots to match how you’ve been returning them:
    # { "HOOK":[...], "PROBLEM":[...], "FEATURE":[...], "PROOF":[...], "CTA":[...] }
    slots_struct: Dict[str, List[Dict[str, Any]]] = {
        "HOOK": [],
        "PROBLEM": [],
        "FEATURE": [],
        "PROOF": [],
        "CTA": [],
    }
    for slot_name, seg_list in slot_map.items():
        norm_slot = slot_name.upper()
        if norm_slot not in slots_struct:
            slots_struct[norm_slot] = []
        for seg in seg_list:
            slots_struct[norm_slot].append({
                "id": seg.get("id", ""),
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "text": seg.get("text", ""),
                "meta": {
                    "slot": seg.get("slot", ""),
                    "score": seg.get("score", 2.5),
                },
                "face_q": seg.get("face_q", 1.0),
                "scene_q": seg.get("scene_q", 1.0),
                "vtx_sim": seg.get("vtx_sim", 0.0),
                "has_product": seg.get("has_product", False),
                "ocr_hit": seg.get("ocr_hit", 0),
            })

    return {
        "ok": True,
        "input_local": src_local,
        "final_local": final_local,
        "duration_sec": final_dur,
        "clips": clips_meta,
        "slots": slots_struct,
    }
