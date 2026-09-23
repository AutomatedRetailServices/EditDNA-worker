from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def probe_duration(path: str) -> float:
    proc = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", path,
    ], capture_output=True, text=True, check=True)
    return float(proc.stdout.strip())


def load_cuts(path: str, duration: float) -> list[tuple[float, float]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = sorted((float(x["start"]), float(x["end"])) for x in data.get("cuts", []))
    if not rows:
        raise ValueError("microtrim manifest has no cuts")
    prev = 0.0
    for start, end in rows:
        if start < 0 or end <= start or end > duration + 1e-3:
            raise ValueError(f"invalid cut {start}-{end} for duration {duration}")
        if start < prev - 1e-6:
            raise ValueError("overlapping microtrim cuts")
        prev = end
    return rows


def build_keep(cuts: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    keep: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in cuts:
        if start > cursor + 1e-6:
            keep.append((cursor, start))
        cursor = end
    if cursor < duration - 1e-6:
        keep.append((cursor, duration))
    return keep


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: apply_output_microtrim.py INPUT.mp4 MANIFEST.json OUTPUT.mp4")
    src, manifest, dst = sys.argv[1:]
    duration = probe_duration(src)
    cuts = load_cuts(manifest, duration)
    keep = build_keep(cuts, duration)

    filter_parts: list[str] = []
    concat_inputs: list[str] = []
    for i, (start, end) in enumerate(keep):
        filter_parts.append(f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[v{i}]")
        filter_parts.append(f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[a{i}]")
        concat_inputs.append(f"[v{i}][a{i}]")
    filter_parts.append("".join(concat_inputs) + f"concat=n={len(keep)}:v=1:a=1[v][a]")

    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", src,
        "-filter_complex", ";".join(filter_parts),
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
        "-movflags", "+faststart", dst,
    ]
    subprocess.run(command, check=True)

    out_duration = probe_duration(dst)
    expected = duration - sum(end - start for start, end in cuts)
    if abs(out_duration - expected) > 0.15:
        raise RuntimeError(f"unexpected output duration {out_duration:.3f}; expected {expected:.3f}")
    print(json.dumps({
        "input_duration": round(duration, 3),
        "removed_duration": round(sum(end - start for start, end in cuts), 3),
        "output_duration": round(out_duration, 3),
        "keep_segment_count": len(keep),
        "speech_changes_allowed": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
