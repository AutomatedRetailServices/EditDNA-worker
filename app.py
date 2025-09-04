import os, tempfile, subprocess
from fastapi import HTTPException

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{p.stderr.decode()}")

def transcribe_in_chunks(client, video_bytes: bytes, filename: str) -> str:
    # Save upload to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_in:
        tmp_in.write(video_bytes)
        in_path = tmp_in.name

    try:
        with tempfile.TemporaryDirectory() as td:
            # Extract 16k mono wav for stable STT
            wav_path = os.path.join(td, "audio.wav")
            run(["ffmpeg", "-y", "-i", in_path, "-ar", "16000", "-ac", "1", wav_path])

            # Get duration (seconds)
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", wav_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            try:
                total = float(probe.stdout.decode().strip())
            except:
                total = 0.0
            if total <= 0.0:
                raise HTTPException(status_code=400, detail="Could not read audio duration")

            # Cut into ~60s pieces to avoid timeouts; transcribe each; join
            CHUNK = 60.0
            cur = 0.0
            pieces = []
            while cur < total:
                dur = min(CHUNK, total - cur)
                cut_wav = os.path.join(td, f"chunk_{int(cur)}.wav")
                run([
                    "ffmpeg", "-y",
                    "-ss", f"{cur:.3f}",
                    "-t", f"{dur:.3f}",
                    "-i", wav_path,
                    "-ac", "1", "-ar", "16000",
                    cut_wav
                ])
                with open(cut_wav, "rb") as f:
                    tr = client.audio.transcriptions.create(model="whisper-1", file=f)
                pieces.append(tr.text.strip() if hasattr(tr, "text") else "")
                cur += CHUNK

            return " ".join(p for p in pieces if p).strip()
    finally:
        try: os.remove(in_path)
        except: pass
