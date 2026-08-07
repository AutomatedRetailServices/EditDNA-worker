"""CLI for one isolated CutSell real-video validation run."""
from __future__ import annotations

import argparse
from pathlib import Path

from .validation import list_validation_videos, report_json, run_single_validation


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one CutSell clean-worker video validation")
    parser.add_argument("--key", default=None, help="Explicit S3 key. If omitted, use first eligible validation video.")
    parser.add_argument("--language", default=None, help="Optional language hint such as en or es")
    parser.add_argument("--start", type=float, default=None, help="Optional source-window start in seconds")
    parser.add_argument("--end", type=float, default=None, help="Optional source-window end in seconds")
    parser.add_argument("--report", default="cutsell-validation-report.json")
    parser.add_argument("--preview", default="cutsell-validation-preview.mp4")
    args = parser.parse_args()

    key = args.key
    if not key:
        inventory = list_validation_videos(limit=1)
        if not inventory:
            raise RuntimeError("no eligible validation videos found")
        key = inventory[0]["key"]

    report = run_single_validation(
        key,
        language_hint=args.language,
        preview_output=args.preview,
        source_start_sec=args.start,
        source_end_sec=args.end,
    )
    Path(args.report).write_text(report_json(report), encoding="utf-8")
    print(report_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
