"""Builds the official CutSell sounds (Audio panel) from the two CC0 packs, plus manifest.json.

Usage: python3 tools/make_cutsell_sounds.py KENNEY_DIR KEYBOARD_DIR OUT_DIR

KENNEY_DIR is the unzipped kenney_interfaceSounds.zip, KEYBOARD_DIR the unzipped
unicae_games_keyboard_soundpack_1_0.zip (both from OpenGameArt, see docs/LICENCIAS_sonidos.md).
The audio is never committed to this repository: it goes to the media bucket (assets/sfx/v1/).

Each file: AAC 160 kb/s, stereo, 44.1 kHz, about -16 LUFS (measured with the sound looped, because
these clicks are shorter than the 0.4 s loudness window), peaks under -1.5 dBTP. A single loudnorm
pass leaves short, peaky sounds at -21 to -23 LUFS, so the gain is measured and applied instead;
very short clicks stop where louder would need heavy limiting (Tap ends near -22 LUFS).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

TARGET_LUFS = -16.0
LIMIT = "alimiter=limit=0.84:attack=1:release=20:level=false"
# (id, menu name, output file, source pack, path inside the pack, trim seconds, extra filter, max gain dB)
SOUNDS = (
    ("tap", "Tap", "tap_select.m4a", "kenney", "Audio/switch_005.ogg", None, "", 6.0),
    ("good", "Good", "good.m4a", "kenney", "Audio/confirmation_002.ogg", None, "", 12.0),
    ("wrong", "Wrong", "wrong.m4a", "kenney", "Audio/error_008.ogg", None, "", 12.0),
    ("glass", "Glass", "glass.m4a", "kenney", "Audio/glass_004.ogg", None, "", 12.0),
    ("typing", "Typing", "typing_3s.m4a", "keyboard", "Human Typing/human_vel-001.wav", 3.0, "afade=t=out:st=2.7:d=0.3,", 40.0),
    ("typing_long", "Typing (long)", "typing.m4a", "keyboard", "Human Typing/human_vel-001.wav", 9.0, "", 40.0),
)
AUTHORS = {"kenney": "Kenney", "keyboard": "unicaegames"}


def _measure(path: Path) -> tuple[float, float]:
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "aloop=loop=7:size=2e9,ebur128=peak=true",
         "-f", "null", "-"], capture_output=True, text=True).stderr
    return float(re.findall(r"I:\s+(-?[\d.]+) LUFS", out)[-1]), float(re.findall(r"Peak:\s+(-?[\d.]+) dBFS", out)[-1])


def _encode(source: Path, out: Path, trim: float | None, pre: str, gain: float) -> None:
    command = ["ffmpeg", "-loglevel", "error", "-y"]
    if trim:
        command += ["-t", f"{trim}"]
    command += ["-i", str(source), "-af", f"{pre}volume={gain:.2f}dB,{LIMIT}",
                "-ar", "44100", "-ac", "2", "-c:a", "aac", "-b:a", "160k", str(out)]
    subprocess.run(command, check=True)


def _duration(path: Path) -> float:
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                 str(path)], capture_output=True, text=True, check=True).stdout)


def main(kenney: str, keyboard: str, out_dir: str) -> None:
    packs = {"kenney": Path(kenney), "keyboard": Path(keyboard)}
    license_text = (packs["kenney"] / "License.txt").read_text(errors="replace")
    readme_text = (packs["keyboard"] / "readme.txt").read_text(errors="replace")
    if "CC0" not in license_text:
        raise SystemExit("Kenney License.txt does not say CC0")
    if "Free to use however you like" not in readme_text:
        raise SystemExit("keyboard pack readme.txt does not say 'Free to use however you like'")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sfx_id, name, file, pack, inner, trim, pre, max_gain in SOUNDS:
        source = packs[pack] / inner
        target = out / file
        gain = 0.0
        loudness = peak = 0.0
        for _ in range(6):
            _encode(source, target, trim, pre, gain)
            loudness, peak = _measure(target)
            if abs(loudness - TARGET_LUFS) < 0.4 or gain >= max_gain:
                break
            gain = min(max_gain, gain + (TARGET_LUFS - loudness))
        # Never let the peak go over the limit (back off the gain until it does not).
        while peak > -1.0 and gain > -12:
            gain -= 2.0
            _encode(source, target, trim, pre, gain)
            loudness, peak = _measure(target)
        rows.append({"id": sfx_id, "name": name, "file": file, "duration_sec": round(_duration(target), 3),
                     "license": "CC0 1.0", "author": AUTHORS[pack],
                     "loudness_lufs": round(loudness, 1), "peak_dbfs": round(peak, 1)})
        print(f"{file:16s} {rows[-1]['duration_sec']:6.3f} s  {loudness:6.1f} LUFS  peak {peak:5.1f} dBFS")
    (out / "manifest.json").write_text(json.dumps({"version": "v1", "sounds": rows}, indent=2) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
