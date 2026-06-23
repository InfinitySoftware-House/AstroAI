from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .inference import main as inference_main
from .training import main as training_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or run the DeepSkyDenoiser patch-based denoiser.")
    parser.add_argument("command", choices=["train", "denoise"])
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        build_parser().parse_args(argv)
        return

    command = argv[0]
    remaining = argv[1:]
    if command == "train":
        training_main(remaining)
        return
    if command == "denoise":
        inference_main(remaining)
        return
    raise ValueError(f"Unknown command: {command}")
