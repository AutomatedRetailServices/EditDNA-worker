# --- add near top ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os, subprocess, tempfile, uuid, json

FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")

class ProcIn(BaseModel):
    input_url: str
    mode: Optional[str] = "best"
    portrait: Optional[bool] = True
    # trim controls
    start: Optional[float] = 0.0       # seconds to seek into source
    max_duration: Optional[float] = 20 # seconds to keep
    # audio controls: "original" (default) | "silent" | "mute"
    audio: Optional[str] = "original"
    # s3 output prefix (optional passthrough if your code uploads after)
    output_prefix: Optional[str] = "editdna/test"
    # choose encoder: libx264 | libopenh264 (default = libx264)
    encoder: Optional[str] = "libx264"

@app.post("/process")
def process_video(p: ProcIn):
    # tmp files
    tmp_dir = "/root/tmp"
    out_dir = "/root/proxies"
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    src_name = f"src_{os.path.basename(p.input_url).split('?')[0]}"
    tmp_in = os.path.join(tmp_dir, src_name if "." in src_name else src_name + ".mp4")
    out_path = os.path.join(out_dir, f"proxy_{uuid.uuid4().hex}.mp4")

    # 1) download
    try:
        # curl is installed in your template now; if not, use python requests instead
        subprocess.run(["curl", "-L", "-o", tmp_in, p.input_url], check=True)
    except Exception as e:
        raise HTTPException(400, f"download failed: {e}")

    # 2) build ffmpeg command
    vf = "scale=1080:trunc(ow/a/2)*2:force_original_aspect_ratio=decrease"
    if p.portrait:
        vf += ",pad=1080:1920:(1080-iw)/2:(1920-ih)/2"
    vf += ",fps=24"

    # choose encoder safely
    vcodec = "libx264" if p.encoder not in ("libx264", "libopenh264") else p.encoder
    vargs = [FFMPEG, "-y"]

    # seek/trim
    start = max(0.0, float(p.start or 0))
    dur = max(0.1, float(p.max_duration or 20))
    if start > 0:
        vargs += ["-ss", str(start)]
    vargs += ["-i", tmp_in]

    # audio mapping
    audio_mode = (p.audio or "original").lower()
    if audio_mode == "original":
        # try to map original audio; if the source has none, ffmpeg will fail mapping 0:a:0
        # fallback: re-run without audio on failure
        map_audio = True
    elif audio_mode == "silent":
        vargs += ["-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=48000"]
        map_audio = "silent"
    else:  # mute
        map_audio = False

    # maps
    if map_audio is True:
        vargs += ["-map", "0:v:0", "-map", "0:a:0"]
    elif map_audio == "silent":
        vargs += ["-map", "0:v:0", "-map", "1:a:0"]
    else:
        vargs += ["-map", "0:v:0"]

    # filters + codecs
    vargs += [
        "-t", str(dur),
        "-vf", vf,
        "-c:v", vcodec,
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
    ]
    if map_audio:
        vargs += ["-c:a", "aac", "-ar", "48000", "-ac", "1", "-b:a", "128k"]
    vargs += ["-movflags", "+faststart", out_path]

    # 3) run ffmpeg (with graceful fallback if original audio missing)
    try:
        subprocess.run(vargs, check=True)
    except subprocess.CalledProcessError as e:
        if map_audio is True:
            # retry without audio if the source had no audio stream
            vargs2 = [arg for arg in vargs if arg not in ["-map","0:a:0","-c:a","aac","-ar","48000","-ac","1","-b:a","128k"]]
            # remove the specific items (list cleanup)
            cleaned = []
            skip = 0
            i = 0
            while i < len(vargs2):
                if i+1 < len(vargs2) and vargs2[i] == "-map" and vargs2[i+1] == "0:a:0":
                    i += 2
                    continue
                if vargs2[i] in ["-c:a","-ar","-ac","-b:a"]:
                    i += 2
                    continue
                cleaned.append(vargs2[i]); i += 1
            subprocess.run(cleaned, check=True)
        else:
            raise HTTPException(500, f"FFmpeg failed: {e}")

    # 4) (optional) upload to S3 – if your code already does this elsewhere, keep that.
    resp = {
        "ok": True,
        "session_id": "session",
        "inputs": [p.input_url],
        "encoder": vcodec,
        "output_path": out_path,
        # if you also upload: include s3_bucket, key, url
    }
    return resp
