from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .core import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the DeepSkyDenoiser from paired image patches.")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/image_pairs"))
    parser.add_argument("--model-path", type=Path, default=Path("models/astro_denoiser.pt"))
    parser.add_argument(
        "--task",
        choices=["denoise", "sharpen"],
        default="denoise",
        help="denoise: learn LR->HR pairs. sharpen: deconvolution, input=blur(HR) synthesized on the fly, target=HR.",
    )
    parser.add_argument("--blur-sigma-min", type=float, default=0.7, help="[sharpen] min Gaussian PSF sigma.")
    parser.add_argument("--blur-sigma-max", type=float, default=2.2, help="[sharpen] max Gaussian PSF sigma.")
    parser.add_argument("--degrade-noise", type=float, default=0.01, help="[sharpen] max additive noise sigma on degraded input.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--enc-blocks", type=int, default=2, help="Base encoder blocks (doubles each deeper level).")
    parser.add_argument("--num-levels", type=int, default=4, help="UNet depth (encoder/decoder levels).")
    parser.add_argument("--bottleneck-blocks", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--drop-path-rate", type=float, default=0.05, help="Max stochastic-depth rate (linearly scaled across blocks).")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--synth-mix-prob", type=float, default=0.0)
    parser.add_argument("--ssim-weight", type=float, default=0.20)
    parser.add_argument("--fft-weight", type=float, default=0.05)
    parser.add_argument("--gradient-weight", type=float, default=0.10)
    parser.add_argument(
        "--faint-signal-weight",
        type=float,
        default=0.25,
        help="Extra loss weight for low-contrast clean signal so faint structures are not trained away.",
    )
    parser.add_argument("--tv-weight", type=float, default=0.0, help="Total variation smoothness weight (useful for gradient-removal models).")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Linear LR warmup epochs before cosine decay.")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision on CUDA.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from --model-path. Older checkpoints fall back to weight-only warm start.",
    )
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="With --resume, keep model weights but start with a fresh optimizer/scheduler/scaler.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
