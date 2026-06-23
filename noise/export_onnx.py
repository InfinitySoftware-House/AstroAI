from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .core import export_model_to_onnx


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a DeepSkyDenoiser checkpoint to ONNX.")
    parser.add_argument("--model-path", type=Path, required=True, help="Input .pt/.pth checkpoint.")
    parser.add_argument("--output", type=Path, required=True, help="Output .onnx file.")
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Dummy export tile size used to trace the model. Height and width remain dynamic in the ONNX graph.",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    export_model_to_onnx(
        model_path=args.model_path,
        output_path=args.output,
        patch_size=args.patch_size,
        opset_version=args.opset,
    )
    print(f"ONNX model saved to: {args.output}")


if __name__ == "__main__":
    main()
