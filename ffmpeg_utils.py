import os
import subprocess
import tempfile
from typing import List, Dict, Any
from dataclasses import dataclass

FFMPEG_BIN  = os.getenv("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "/usr/bin/ffprobe")


def _run(cmd: List[str]) -> None:
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    out, err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {err.strip()}")


def tmp_path(suffix: str = ".mp4") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


def cut_and_concat_story(raw_video_path: str, story_takes: List[Dict[str, Any]]) -> str:
    """
    story_takes = list of takes from semantic_visual_pass() final story
    Each take needs: start, end
    We:
      1. cut each take
      2. concat them with ffmpeg
    Return path to combined mp4
    """
    part_paths = []
    concat_file = tmp_path(suffix=".txt")

    # make each part
    for idx, t in enumerate(story_takes, start=1):
        start = float(t["start"])
        end   = float(t["end"])
        dur   = end - start

        part_path = tmp_path(suffix=f".part{idx:02d}.mp4")
        part_paths.append(part_path)

        cmd = [
            FFMPEG_BIN, "-y",
            "-ss", f"{start:.3f}",
            "-i", raw_video_path,
            "-t", f"{dur:.3f}",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-g", "48",
            "-c:a", "aac",
            "-b:a", "128k",
            part_path
        ]
        _run(cmd)

    # write concat list
    with open(concat_file, "w") as f:
        for p in part_paths:
            f.write(f"file '{p}'\n")

    final_path = tmp_path(suffix=".mp4")
    cmd2 = [
        FFMPEG_BIN, "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_file,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-g", "48",
        "-c:a", "aac",
        "-b:a", "128k",
        final_path
    ]
    _run(cmd2)

    return final_path


def burn_captions_srt(input_video: str, slots_dict: Dict[str, List[Dict[str, Any]]]) -> str:
    """
    Build temporary .srt from slots_dict["HOOK"] text chunks,
    then overlay using ffmpeg subtitles filter.
    """
    hook_list = slots_dict.get("HOOK", [])

    def fmt_ts(sec: float) -> str:
        ms = int(round((sec - int(sec)) * 1000))
        s = int(sec)
        hh = s // 3600
        mm = (s % 3600) // 60
        ss = s % 60
        return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

    # generate SRT
    srt_path = tmp_path(suffix=".srt")
    lines = []
    for i, seg in enumerate(hook_list, start=1):
        start = float(seg["start"])
        end   = float(seg["end"])
        text  = str(seg.get("text", "")).strip() or "."
        lines.append(str(i))
        lines.append(f"{fmt_ts(start)} --> {fmt_ts(end)}")
        lines.append(text.replace("\n"," "))
        lines.append("")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    out_path = tmp_path(suffix=".mp4")
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", input_video,
        "-vf", f"subtitles={srt_path}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-g", "48",
        "-c:a", "aac",
        "-b:a", "128k",
        out_path
    ]
    _run(cmd)
    return out_path


def probe_duration_sec(video_path: str) -> float:
    """
    Use ffprobe to get final video duration.
    """
    p = subprocess.Popen(
        [
            FFPROBE_BIN,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=nokey=1:noprint_wrappers=1",
            video_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = p.communicate()
    if p.returncode != 0:
        return 0.0
    try:
        return float(out.strip())
    except:
        return 0.0
