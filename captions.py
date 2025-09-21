# captions.py — generate and burn captions (.srt -> hard subtitles)

import os
import subprocess
from typing import List, Tuple

FFMPEG = os.getenv("FFMPEG_BIN", "ffmpeg")

def write_srt(segments: List[Tuple[float, float, str]], out_path: str):
    """
    segments: list of (start, end, text)
    Writes a UTF-8 .srt file.
    """
    def format_time(t: float) -> str:
        if t < 0:
            t = 0.0
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int(round((t - int(t)) * 1000))
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    with open(out_path, "w", encoding="utf-8") as f:
        for i, (s, e, txt) in enumerate(segments, 1):
            text = (txt or "").strip()
            if not text:
                continue
            f.write(f"{i}\n{format_time(s)} --> {format_time(e)}\n{text}\n\n")

def burn_captions(video_in: str, srt_path: str, video_out: str):
    """
    Burns captions using ffmpeg's subtitles filter (libass).
    Style: readable white with thin black outline.
    """
    style = "FontName=Arial,FontSize=36,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,BorderStyle=3,Outline=2,Shadow=0"
    cmd = [
        FFMPEG, "-y",
        "-i", video_in,
        "-vf", f"subtitles={srt_path}:force_style='{style}'",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        video_out,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Caption burn failed:\n{proc.stdout}")
