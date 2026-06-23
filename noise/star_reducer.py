from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .core import (
    ImageMetadata,
    adapt_channels_for_model,
    adapt_channels_for_output,
    load_model_cached,
    load_onnx_session_cached,
    read_image,
    resolve_device,
    run_onnx_tta_denoise,
    run_tta_denoise,
)


def mean_filter_2d(image: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return image.astype(np.float32, copy=False)
    padded = np.pad(image.astype(np.float32, copy=False), radius, mode="reflect")
    out = np.zeros_like(image, dtype=np.float32)
    diameter = radius * 2 + 1
    for y_offset in range(diameter):
        for x_offset in range(diameter):
            out += padded[y_offset : y_offset + image.shape[0], x_offset : x_offset + image.shape[1]]
    return out / float(diameter * diameter)


def smoothstep_mask(value: np.ndarray, low: float, high: float) -> np.ndarray:
    if high <= low:
        return (value >= high).astype(np.float32)
    t = np.clip((value - low) / (high - low), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32, copy=False)


def preserve_star_core_brightness(original: np.ndarray, reduced: np.ndarray) -> np.ndarray:
    """
    Keep compact star cores from getting dark centers while still allowing the
    model to shrink the surrounding halo.

    Only fires on *compact* bright peaks (small detail radius), so it does not
    fight against the model when it correctly shrinks large / bloated stars.
    A larger background radius (4 instead of 2) improves detection of small
    cores that sit inside wide halos.
    """
    if original.shape != reduced.shape:
        raise ValueError(f"Brightness preservation shape mismatch: {original.shape} != {reduced.shape}")

    if original.shape[0] == 1:
        original_luma = original[0]
    else:
        original_luma = np.max(original, axis=0)

    # Radius-4 mean filter subtracts the local halo more completely, so the
    # 'detail' map has a high response only for genuinely compact, sharp cores.
    background = mean_filter_2d(original_luma, radius=4)
    detail = original_luma - background

    threshold = max(float(np.percentile(original_luma, 97.5)), float(original_luma.mean() + original_luma.std()))
    brightness_mask = np.clip((original_luma - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
    # Require stronger local contrast (0.04 vs 0.035) to avoid triggering on diffuse halos
    detail_mask = np.clip(detail / 0.04, 0.0, 1.0)
    bright_core_mask = brightness_mask * detail_mask

    # Faint compact cores: tighter gate — only very sharp local peaks
    faint_detail_mask = smoothstep_mask(detail, low=0.008, high=0.030)
    faint_core_gate = smoothstep_mask(original_luma - background, low=0.006, high=0.022)
    faint_core_mask = faint_detail_mask * faint_core_gate
    core_mask = np.maximum(bright_core_mask, faint_core_mask)[None, :, :].astype(np.float32, copy=False)

    dimming = np.maximum(original - reduced, 0.0)
    restored = reduced + dimming * core_mask
    return np.clip(restored, 0.0, 1.0).astype(np.float32)


def run_star_reducer_inference(
    model_path: Path,
    input_path: Path,
    patch_size: int = 128,
    stride: int = 64,
    tta: int = 4,
    amp: bool = False,
    batch_size: int = 32,
    strength: float = 1.0,
    device_name: str = "",
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, ImageMetadata]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    use_onnx = model_path.suffix.lower() == ".onnx"
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
        model = None
        device = None
    else:
        device = resolve_device(device_name)
        if progress_callback is not None:
            progress_callback(0.08, "Loading star reducer")
        model = load_model_cached(model_path, device)
        expected_channels = next(model.parameters()).shape[1]
        session = None

    if progress_callback is not None:
        progress_callback(0.12, "Reading image")
    img_hwc, meta = read_image(input_path)
    img_chw = np.transpose(img_hwc, (2, 0, 1)).astype(np.float32)
    original_channels = int(img_chw.shape[0])
    model_input = adapt_channels_for_model(img_chw, expected_channels)

    backend_progress = (
        None
        if progress_callback is None
        else lambda fraction, message: progress_callback(0.18 + 0.74 * fraction, message)
    )
    if progress_callback is not None:
        progress_callback(0.18, "Reducing stars")

    if use_onnx:
        reduced = run_onnx_tta_denoise(
            session=session,
            image_chw=model_input,
            patch_size=patch_size,
            stride=stride,
            tta=tta,
            batch_size=batch_size,
            progress_callback=backend_progress,
        )
    else:
        reduced = run_tta_denoise(
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

    reduced = adapt_channels_for_output(reduced, original_channels)
    strength = max(0.0, float(strength))
    if strength != 1.0:
        reduced = np.clip(img_chw + (reduced - img_chw) * strength, 0.0, 1.0).astype(np.float32)
    reduced = preserve_star_core_brightness(img_chw, reduced)
    if progress_callback is not None:
        progress_callback(0.94, "Star reduction complete")
    return img_hwc, reduced, meta
