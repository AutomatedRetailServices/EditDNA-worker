# jobs.py
import os, uuid, json, tempfile, subprocess, shlex, pathlib
from typing import List, Dict, Any

# --- S3 helpers ---
try:
    from s3_utils import s3_put_file
except Exception:
    import boto3
    def s3_put_file(bucket: str, key: str, path: str, content_type: str = "video/mp4"):
        boto3.client("s3").upload_file(path, bucket, key, ExtraArgs={"ContentType": content_type})

def _run(cmd: str):
    proc = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"cmd failed: {cmd}\n\n{proc.stdout}")
    return proc.stdout

def _ff():  # ffmpeg path
    return "ffmpeg"

def _proxy_one(inp: str, out_path: str):
    # fast 720p proxy (2-pass not needed; ultrafast preset to keep it cheap)
    cmd = (
        f'{_ff()} -y -hide_banner -loglevel error -i "{inp}" '
        f'-vf "scale=-2:720" -r 30 -c:v libx264 -preset ultrafast -crf 23 '
        f'-c:a aac -b:a 128k "{out_path}"'
    )
    _run(cmd)

def _cut_snippet(src: str, start: float, dur: float, out_path: str):
    cmd = (
        f'{_ff()} -y -hide_banner -loglevel error -ss {start} -i "{src}" -t {dur} '
        f'-c copy "{out_path}"'
    )
    _run(cmd)

def _concat_mp4(parts: List[str], out_path: str):
    # concat via intermediate list
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in parts:
            f.write(f"file '{p}'\n")
        flist = f.name
    cmd = f'{_ff()} -y -hide_banner -loglevel error -f concat -safe 0 -i "{flist}" -c copy "{out_path}"'
    _run(cmd)

def render_main(params: Dict[str, Any], *, bucket: str, prefix: str) -> Dict[str, Any]:
    """
    Minimal pipeline:
      1) proxy all inputs -> 720p
      2) take 3x3s snippets from the first N proxies
      3) concat -> final mp4
      4) upload to S3
    """
    files: List[str] = params.get("files") or []
    if not files:
        raise ValueError("No input files provided")

    # where to store artifacts locally during job
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        proxies = []

        # 1) make proxies
        for i, url in enumerate(files):
            p = td / f"proxy_{i:02d}.mp4"
            _proxy_one(url, str(p))
            proxies.append(str(p))

        # 2) slice simple demo snippets (0-3s, 5-8s, 10-13s) from first proxy (or round-robin)
        parts = []
        picks = [(0, 3), (5, 3), (10, 3)]  # (start, duration)
        base = proxies[0]
        for j, (start, dur) in enumerate(picks):
            out = td / f"part_{j:02d}.mp4"
            _cut_snippet(base, start, dur, str(out))
            parts.append(str(out))

        # 3) concat to final
        final_path = td / "editdna_out.mp4"
        _concat_mp4(parts, str(final_path))

        # 4) upload to S3
        out_key = f"{prefix.rstrip('/')}/videos/ad_{uuid.uuid4().hex}.mp4"
        s3_put_file(bucket, out_key, str(final_path), content_type="video/mp4")

        return {
            "ok": True,
            "bucket": bucket,
            "key": out_key,
            "url": f"https://{bucket}.s3.amazonaws.com/{out_key}",
        }
