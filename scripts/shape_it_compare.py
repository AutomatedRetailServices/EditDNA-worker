#!/usr/bin/env python3
"""Shape it -- real side-by-side of the 3 providers on 3 of your own photos (9 PAID calls).

    python scripts/shape_it_compare.py PANTS.png SWEATER.png SHOES.png --out ~/shape-it-compare --yes-spend

Each input must be a PNG with the background already removed (alpha), like the phone sends. Outputs go to --out,
which must be OUTSIDE this repository (it is public: never commit photos). Keys are read from the environment
(CUTSELL_SEEDREAM_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY) and are never printed.

Without --yes-spend the script only validates the inputs and prints the estimated cost; nothing is sent.
Estimated total (11 Nov 2026 list prices): 3 x ($0.018 + ~$0.066 + ~$0.034) = ~$0.35.
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from cutsell_app import shop_shape as shape  # noqa: E402

ITEMS = (("pants", "clothing"), ("sweater", "clothing"), ("shoes", "shoes"))
ESTIMATE = {"seedream": shape.SEEDREAM_PRICE_PER_IMAGE, "gpt-image": shape.OPENAI_ESTIMATE_PER_CALL,
            "nano-banana": shape.GEMINI_ESTIMATE_PER_CALL}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pants", nargs="?")
    parser.add_argument("sweater", nargs="?")
    parser.add_argument("shoes", nargs="?")
    parser.add_argument("--item", action="append", default=[], metavar="LABEL:KIND:PATH",
                        help="extra or replacement item, KIND is clothing or shoes (repeatable)")
    parser.add_argument("--out", required=True, help="output folder, outside the repository")
    parser.add_argument("--providers", default="seedream,gpt-image,nano-banana")
    parser.add_argument("--yes-spend", action="store_true", help="really call the paid APIs")
    args = parser.parse_args(argv)

    out = Path(os.path.expanduser(args.out)).resolve()
    if out == REPO or REPO in out.parents:
        print("refusing: --out must be outside the repository (public repo, no images)", file=sys.stderr)
        return 2
    providers = [shape.provider_name(p.strip()) for p in args.providers.split(",") if p.strip()]

    items = [(label, kind, path) for (label, kind), path in zip(ITEMS, (args.pants, args.sweater, args.shoes)) if path]
    for spec in args.item:
        label, kind, path = spec.split(":", 2)
        if kind not in ("clothing", "shoes"):
            parser.error(f"--item {label}: KIND must be clothing or shoes")
        items.append((label, kind, path))
    if not items:
        parser.error("give the three positional photos or at least one --item")
    garments = []
    for label, kind, path in items:
        data = Path(path).expanduser().read_bytes()
        garment = shape.decode_image(base64.b64encode(data).decode())
        garments.append((label, kind, garment))
        print(f"input {label:8s} {garment.width}x{garment.height}  key={garment.key_name}")

    estimate = sum(ESTIMATE[p] for p in providers) * len(garments)
    print(f"\n{len(providers) * len(garments)} calls, estimated total ${estimate:.3f}")
    for p in providers:
        print(f"  {p:12s} ~${ESTIMATE[p]:.3f} per Shape it")
    if not args.yes_spend:
        print("\ndry run: add --yes-spend to call the providers")
        return 0
    for p in providers:
        if not os.environ.get(shape.PROVIDERS[p].key_env, "").strip():
            print(f"missing {shape.PROVIDERS[p].key_env} for {p}", file=sys.stderr)
            return 2

    out.mkdir(parents=True, exist_ok=True)
    total = 0.0
    print()
    for label, kind, garment in garments:
        for p in providers:
            provider = shape.get_provider(p)
            started = time.monotonic()
            try:
                result = shape.call_provider(provider, garment, kind)
                (out / f"{label}__{p}__raw.png").write_bytes(result.image_bytes)
                finished = shape.finish_output(result.image_bytes, garment)
                (out / f"{label}__{p}.png").write_bytes(finished)
                total += result.cost_usd
                status = "ok"
                cost = f"${result.cost_usd:.4f}"
            except shape.ShapeError as exc:
                status, cost = exc.code, "-"
            print(f"{label:8s} {p:12s} {status:24s} {cost:>9s}  {time.monotonic() - started:5.1f}s")
    print(f"\ntotal recorded cost ${total:.4f}   outputs in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
