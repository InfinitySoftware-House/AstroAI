from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter

from noise.core import (
    adapt_channels_for_model,
    adapt_channels_for_output,
    load_model_cached,
    load_onnx_session_cached,
    read_image,
    resolve_device,
    run_onnx_tta_denoise,
    run_tta_denoise,
    write_image,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run gradient-removal inference with a trained gradient-removal model."
    )
    parser.add_argument("--model-path", type=Path, default=Path("models/astro_gradient_remover.pth"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64, help="Patch stride; smaller means more overlap.")
    parser.add_argument("--tta", type=int, default=4, help="Number of test-time augmentation modes to average (1-8).")
    parser.add_argument("--batch-size", type=int, default=32, help="Number of patches processed at once.")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision on CUDA.")
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Gradient-removal strength: 0=no effect, 1=normal, >1=stronger.",
    )
    parser.add_argument(
        "--detail-preservation",
        type=float,
        default=0.1,
        help="Blend back fine detail from the source image: 0=smoother, 1=more original detail retained.",
    )
    parser.add_argument(
        "--background-threshold",
        type=float,
        default=0.2,
        help="Dark background below this level receives stronger correction.",
    )
    parser.add_argument(
        "--background-strength",
        type=float,
        default=1.25,
        help="Extra correction strength for dark background regions.",
    )
    parser.add_argument(
        "--subject-detail-preservation",
        type=float,
        default=0.15,
        help="Detail preservation applied to brighter subjects and structures.",
    )
    parser.add_argument(
        "--background-detail-preservation",
        type=float,
        default=0.03,
        help="Detail preservation applied to dark background regions.",
    )
    parser.add_argument(
        "--gradient-blur-sigma",
        type=float,
        default=3.0,
        help="Gaussian blur sigma applied to the predicted gradient before subtraction. "
             "Prevents high-frequency noise removal. Set to 0 to disable.",
    )
    parser.add_argument("--device", type=str, default="")
    return parser


def decode_gradient_prediction(prediction: np.ndarray) -> np.ndarray:
    return prediction.astype(np.float32) - 0.5


def run_gradient_inference(
    model_path: Path,
    input_path: Path,
    patch_size: int = 128,
    stride: int = 64,
    tta: int = 4,
    amp: bool = False,
    batch_size: int = 32,
    strength: float = 1.0,
    detail_preservation: float = 0.1,
    background_threshold: float = 0.2,
    background_strength: float = 1.25,
    subject_detail_preservation: float = 0.15,
    background_detail_preservation: float = 0.03,
    gradient_blur_sigma: float = 3.0,
    device_name: str = "",
    progress_callback: Callable[[float, str], None] | None = None,
):
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    use_onnx = model_path.suffix.lower() == ".onnx"
    model = None
    device = None
    session = None
    expected_channels = 0

    if progress_callback is not None:
        progress_callback(0.02, "Resolving backend")
    if use_onnx:
        if progress_callback is not None:
            progress_callback(0.08, "Loading ONNX model")
        session = load_onnx_session_cached(model_path, device_name=device_name)
        input_shape = session.get_inputs()[0].shape
        if len(input_shape) != 4 or not isinstance(input_shape[1], int):
            raise RuntimeError(f"Unsupported ONNX input shape: {input_shape}")
        expected_channels = int(input_shape[1])
    else:
        device = resolve_device(device_name)
        if progress_callback is not None:
            progress_callback(0.08, "Loading model")
        model = load_model_cached(model_path, device)
        expected_channels = next(model.parameters()).shape[1]

    if progress_callback is not None:
        progress_callback(0.12, "Reading image")
    img_hwc, meta = read_image(input_path)
    img_chw = np.transpose(img_hwc, (2, 0, 1)).astype(np.float32)
    original_channels = int(img_chw.shape[0])
    model_input = adapt_channels_for_model(img_chw, expected_channels)

    backend_progress = (
        None
        if progress_callback is None
        else lambda fraction, message: progress_callback(0.16 + 0.74 * fraction, message)
    )
    if progress_callback is not None:
        progress_callback(0.16, "Estimating background gradient")

    if use_onnx:
        pred_encoded = run_onnx_tta_denoise(
            session=session,
            image_chw=model_input,
            patch_size=patch_size,
            stride=stride,
            tta=tta,
            batch_size=batch_size,
            progress_callback=backend_progress,
        )
    else:
        pred_encoded = run_tta_denoise(
            model=model,
            image_chw=model_input,
            patch_size=patch_size,
            stride=stride,
            device=device,
            tta=tta,
            amp=amp,
            batch_size=batch_size,
            progress_callback=backend_progress,
        )

    pred_encoded = adapt_channels_for_output(pred_encoded, original_channels)
    predicted_gradient = decode_gradient_prediction(pred_encoded)

    # Blur the gradient map so only low-frequency structure is removed.
    # This prevents noise or fine detail from being subtracted along with the gradient.
    if gradient_blur_sigma > 0:
        predicted_gradient = np.stack(
            [gaussian_filter(predicted_gradient[c], sigma=gradient_blur_sigma) for c in range(predicted_gradient.shape[0])],
            axis=0,
        ).astype(np.float32)

    corrected = np.clip(img_chw - predicted_gradient * strength, 0.0, 1.0).astype(np.float32)

    if detail_preservation > 0:
        blend = np.clip(float(detail_preservation), 0.0, 1.0)
        corrected = np.clip(corrected * (1.0 - blend) + img_chw * blend, 0.0, 1.0).astype(np.float32)

    if progress_callback is not None:
        progress_callback(1.0, "Gradient removal complete")
    return img_hwc, corrected, meta


def remove_gradient_file(
    model_path: Path,
    input_path: Path,
    output_path: Path,
    patch_size: int = 128,
    stride: int = 64,
    tta: int = 4,
    amp: bool = False,
    batch_size: int = 32,
    strength: float = 1.0,
    detail_preservation: float = 0.1,
    background_threshold: float = 0.2,
    background_strength: float = 1.25,
    subject_detail_preservation: float = 0.15,
    background_detail_preservation: float = 0.03,
    gradient_blur_sigma: float = 3.0,
    device_name: str = "",
    progress_callback: Callable[[float, str], None] | None = None,
) -> Path:
    _, corrected, meta = run_gradient_inference(
        model_path=model_path,
        input_path=input_path,
        patch_size=patch_size,
        stride=stride,
        tta=tta,
        amp=amp,
        batch_size=batch_size,
        strength=strength,
        detail_preservation=detail_preservation,
        background_threshold=background_threshold,
        background_strength=background_strength,
        subject_detail_preservation=subject_detail_preservation,
        background_detail_preservation=background_detail_preservation,
        gradient_blur_sigma=gradient_blur_sigma,
        device_name=device_name,
        progress_callback=progress_callback,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_image(output_path, corrected, meta)
    return output_path


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_path = remove_gradient_file(
        model_path=args.model_path,
        input_path=args.input,
        output_path=args.output,
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
        gradient_blur_sigma=args.gradient_blur_sigma,
        device_name=args.device,
    )
    print(f"Gradient-corrected image saved to: {output_path}")


if __name__ == "__main__":
    main()
