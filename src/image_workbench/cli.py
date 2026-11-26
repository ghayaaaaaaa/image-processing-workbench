"""Command-line interface for the local image-processing demo."""

from __future__ import annotations

import argparse
from pathlib import Path

from .demo import build_demo


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-workbench",
        description="Generate a synthetic image and compare local processing steps.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="create the synthetic image walkthrough")
    demo.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))
    demo.add_argument("--size", type=int, default=320, help="image width and height (128–1024)")
    demo.add_argument("--seed", type=int, default=17, help="non-negative seed for generated shapes")
    demo.add_argument("--cutoff", type=float, default=0.12, help="Fourier low-pass cutoff (0.02–0.5)")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        output_dir = build_demo(
            output_dir=args.output_dir,
            size=args.size,
            seed=args.seed,
            cutoff=args.cutoff,
        )
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"image-workbench: {exc}")
        return 2
    print(f"Created synthetic image-processing demo in {output_dir.resolve()}")
    print("Artifacts: 6 PNG variants, 1 contact sheet, and manifest.json")
    return 0
