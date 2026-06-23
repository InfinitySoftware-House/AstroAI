from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .core import denoise_image_file, write_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run image denoising with a trained DeepSkyDenoiser model.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64, help="Patch stride; smaller means more overlap.")
    parser.add_argument("--tta", type=int, default=4, help="Number of test-time augmentation modes to average (1-8).")
    parser.add_argument("--batch-size", type=int, default=32, help="Number of patches processed at once.")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision on CUDA.")
    parser.add_argument("--strength", type=float, default=1.0, help="Inference strength: 0=no effect, 1=normal, >1=stronger.")
    parser.add_argument(
        "--detail-preservation",
        type=float,
        default=0.2,
        help="Blend back fine detail from the source image: 0=smoother, 1=more detail retained.",
    )
    parser.add_argument(
        "--background-threshold",
        type=float,
        default=0.12,
        help="Dark pixels below this level receive stronger background cleanup.",
    )
    parser.add_argument("--background-strength", type=float, default=1.2, help="Extra denoise strength for dark regions.")
    parser.add_argument(
        "--subject-detail-preservation",
        type=float,
        default=0.28,
        help="Detail preservation applied to brighter subjects and structures.",
    )
    parser.add_argument(
        "--background-detail-preservation",
        type=float,
        default=0.05,
        help="Detail preservation applied to dark background regions.",
    )
    parser.add_argument(
        "--faint-structure-protection",
        type=float,
        default=0.85,
        help="Protect low-contrast local structures from being treated as empty background.",
    )
    parser.add_argument(
        "--faint-signal-preservation",
        type=float,
        default=0.70,
        help="Restore model-dimmed faint signal after denoising.",
    )
    parser.add_argument(
        "--faint-signal-boost",
        type=float,
        default=0.10,
        help="Subtle local contrast boost for detected faint signal.",
    )
    parser.add_argument(
        "--sharpen",
        type=float,
        default=0.15,
        help="Unsharp mask strength after denoising: 0=off, 0.3=subtle, 1.0=strong.",
    )
    parser.add_argument(
        "--noise-floor",
        type=float,
        default=0.0,
        help="Subtract a bias floor from the output (like bias-frame subtraction). Removes residual amp glow.",
    )
    parser.add_argument(
        "--highlight-protection",
        type=float,
        default=1.0,
        help="Blend original back above this luminance threshold to protect star cores from ringing (0.8-0.95).",
    )
    parser.add_argument("--device", type=str, default="")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _, denoised, meta = denoise_image_file(
        model_path=args.model_path,
        input_path=args.input,
        patch_size=args.patch_size,
        stride=args.stride,
        tta=args.tta,
        amp=args.amp,
        batch_size=args.batch_size,
        strength=args.strength,
        detail_preservation=args.detail_preservation,
        background_threshold=args.background_threshold,
        background_strength=args.background_strength,
        subject_detail_preservation=args.subject_detail_preservation,
        background_detail_preservation=args.background_detail_preservation,
        faint_structure_protection=args.faint_structure_protection,
        faint_signal_preservation=args.faint_signal_preservation,
        faint_signal_boost=args.faint_signal_boost,
        sharpen=args.sharpen,
        noise_floor=args.noise_floor,
        highlight_protection=args.highlight_protection,
        device_name=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_image(args.output, denoised, meta)
    print(f"Denoised image saved to: {args.output}")


if __name__ == "__main__":
    main()
