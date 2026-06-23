#!/usr/bin/env python3
"""Core training and inference logic for the DeepSkyDenoiser patch-based denoiser."""

from __future__ import annotations

import math
import random
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
import tifffile

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, random_split
except ImportError:
    torch = None
    F = None
    nn = None
    Dataset = object
    DataLoader = None
    random_split = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

Image.MAX_IMAGE_PIXELS = None


from .arch_lite import LITE_ARCH, build_lite_model, build_lite_model_from_checkpoint

MODEL_ARCH = "astro_unet_v4"
ProgressCallback = Callable[[float, str], None]
PatchPredictor = Callable[[np.ndarray], np.ndarray]
_PYTORCH_MODEL_CACHE: Dict[Tuple[str, str], Tuple[float, "nn.Module"]] = {}
_ONNX_SESSION_CACHE: Dict[Tuple[str, str], Tuple[float, object]] = {}


@dataclass
class ImageMetadata:
    source_ext: str
    original_dtype: np.dtype
    original_channels: int
    value_min: float | None = None
    value_max: float | None = None
    norm_lo: np.ndarray | None = None
    norm_hi: np.ndarray | None = None


def require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required. Install it first, for example:\n"
            "  pip install torch torchvision"
        )


def _cuda_runtime_failed(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        token in text
        for token in (
            "no kernel image is available for execution on the device",
            "cuda error",
            "cuda driver",
            "invalid device function",
        )
    )


def resolve_device(requested_device: str = "") -> "torch.device":
    require_torch()
    if requested_device:
        device = torch.device(requested_device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        return device

    try:
        # Run a tiny convolution, because simple tensor allocation can succeed
        # even when the current PyTorch build cannot execute CUDA kernels here.
        probe = nn.Conv2d(1, 1, kernel_size=1).to(device)
        sample = torch.zeros((1, 1, 4, 4), device=device, dtype=torch.float32)
        _ = probe(sample)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        return device
    except Exception as exc:
        if requested_device:
            raise RuntimeError(
                f"CUDA device '{requested_device}' is not usable with this PyTorch install/GPU: {exc}\n"
                "Use --device cpu, or install a PyTorch build compatible with your GPU."
            ) from exc
        if _cuda_runtime_failed(exc):
            print(
                "CUDA was detected, but it is not usable with this PyTorch install/GPU. "
                "Falling back to CPU."
            )
            return torch.device("cpu")
        raise


def robust_normalize_with_stats(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image.ndim == 2:
        finite = np.isfinite(image)
        if not finite.any():
            zeros = np.zeros_like(image, dtype=np.float32)
            return zeros, np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32)
        valid = image[finite]
        lo = float(np.percentile(valid, 1.0))
        hi = float(np.percentile(valid, 99.0))
        if hi <= lo:
            lo, hi = float(np.min(valid)), float(np.max(valid))
            if hi <= lo:
                zeros = np.zeros_like(image, dtype=np.float32)
                return zeros, np.array([lo], dtype=np.float32), np.array([lo + 1.0], dtype=np.float32)
        out = (image - lo) / (hi - lo)
        out = np.nan_to_num(np.clip(out, 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        return out, np.array([lo], dtype=np.float32), np.array([hi], dtype=np.float32)
    if image.ndim == 3 and image.shape[2] == 3:
        out = np.empty_like(image, dtype=np.float32)
        los: List[float] = []
        his: List[float] = []
        for c in range(3):
            out[:, :, c], lo, hi = robust_normalize_with_stats(image[:, :, c])
            los.append(float(lo[0]))
            his.append(float(hi[0]))
        return out, np.asarray(los, dtype=np.float32), np.asarray(his, dtype=np.float32)
    raise ValueError(f"Expected 2D or RGB image, got shape {image.shape}")


def normalize_with_stats(image: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    lo = lo.astype(np.float32)
    hi = hi.astype(np.float32)
    if image.ndim == 2:
        scale = max(float(hi[0] - lo[0]), 1e-6)
        out = (image.astype(np.float32) - float(lo[0])) / scale
        return np.nan_to_num(np.clip(out, 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    if image.ndim == 3 and image.shape[2] in {1, 3}:
        out = np.empty_like(image, dtype=np.float32)
        for c in range(image.shape[2]):
            scale = max(float(hi[c] - lo[c]), 1e-6)
            out[:, :, c] = (image[:, :, c].astype(np.float32) - float(lo[c])) / scale
        return np.nan_to_num(np.clip(out, 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    raise ValueError(f"Expected 2D or RGB image, got shape {image.shape}")


def compute_pair_normalization_stats(noisy: np.ndarray, clean: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if noisy.shape != clean.shape:
        raise ValueError(f"Expected matching noisy/clean shapes, got {noisy.shape} and {clean.shape}")

    if noisy.ndim == 2:
        valid = np.concatenate(
            [noisy[np.isfinite(noisy)].astype(np.float32), clean[np.isfinite(clean)].astype(np.float32)]
        )
        if valid.size == 0:
            return np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32)
        lo = float(np.percentile(valid, 1.0))
        hi = float(np.percentile(valid, 99.0))
        if hi <= lo:
            lo = float(np.min(valid))
            hi = float(np.max(valid))
            if hi <= lo:
                hi = lo + 1.0
        return np.array([lo], dtype=np.float32), np.array([hi], dtype=np.float32)

    if noisy.ndim == 3 and noisy.shape[2] in {1, 3}:
        los: List[float] = []
        his: List[float] = []
        for c in range(noisy.shape[2]):
            channel_valid = np.concatenate(
                [
                    noisy[:, :, c][np.isfinite(noisy[:, :, c])].astype(np.float32),
                    clean[:, :, c][np.isfinite(clean[:, :, c])].astype(np.float32),
                ]
            )
            if channel_valid.size == 0:
                los.append(0.0)
                his.append(1.0)
                continue
            lo = float(np.percentile(channel_valid, 1.0))
            hi = float(np.percentile(channel_valid, 99.0))
            if hi <= lo:
                lo = float(np.min(channel_valid))
                hi = float(np.max(channel_valid))
                if hi <= lo:
                    hi = lo + 1.0
            los.append(lo)
            his.append(hi)
        return np.asarray(los, dtype=np.float32), np.asarray(his, dtype=np.float32)

    raise ValueError(f"Expected 2D or RGB image, got shape {noisy.shape}")


def _parse_png_chunks(data: bytes) -> List[Tuple[bytes, bytes]]:
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Not a PNG file")
    chunks: List[Tuple[bytes, bytes]] = []
    offset = 8
    while offset + 12 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        chunks.append((chunk_type, chunk_data))
        offset += 12 + length
        if chunk_type == b"IEND":
            break
    return chunks


def _png_paeth_predictor(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def read_png_rgb16(path: Path) -> np.ndarray | None:
    chunks = _parse_png_chunks(path.read_bytes())
    ihdr = next((chunk_data for chunk_type, chunk_data in chunks if chunk_type == b"IHDR"), None)
    if ihdr is None:
        raise ValueError(f"Missing IHDR chunk in PNG: {path}")

    width, height, bit_depth, color_type, compression, png_filter, interlace = struct.unpack(">IIBBBBB", ihdr)
    if bit_depth != 16 or color_type != 2:
        return None
    if compression != 0 or png_filter != 0 or interlace != 0:
        raise ValueError(f"Unsupported 16-bit RGB PNG layout in {path}")

    idat = b"".join(chunk_data for chunk_type, chunk_data in chunks if chunk_type == b"IDAT")
    raw = zlib.decompress(idat)
    bytes_per_pixel = 6
    stride = width * bytes_per_pixel
    expected = height * (stride + 1)
    if len(raw) != expected:
        raise ValueError(f"Unexpected decompressed PNG size for {path}: {len(raw)} != {expected}")

    rows: List[bytes] = []
    prev = bytearray(stride)
    offset = 0
    for _ in range(height):
        filter_type = raw[offset]
        offset += 1
        src = raw[offset : offset + stride]
        offset += stride
        recon = bytearray(stride)
        if filter_type == 0:
            recon[:] = src
        elif filter_type == 1:
            for i in range(stride):
                left = recon[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                recon[i] = (src[i] + left) & 0xFF
        elif filter_type == 2:
            for i in range(stride):
                recon[i] = (src[i] + prev[i]) & 0xFF
        elif filter_type == 3:
            for i in range(stride):
                left = recon[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                up = prev[i]
                recon[i] = (src[i] + ((left + up) // 2)) & 0xFF
        elif filter_type == 4:
            for i in range(stride):
                left = recon[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                up = prev[i]
                up_left = prev[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                recon[i] = (src[i] + _png_paeth_predictor(left, up, up_left)) & 0xFF
        else:
            raise ValueError(f"Unsupported PNG filter type {filter_type} in {path}")
        rows.append(bytes(recon))
        prev = recon

    return np.frombuffer(b"".join(rows), dtype=">u2").astype(np.uint16).reshape(height, width, 3)


def read_standard_image(path: Path) -> Tuple[np.ndarray, ImageMetadata]:
    if path.suffix.lower() == ".png":
        png_rgb16 = read_png_rgb16(path)
        if png_rgb16 is not None:
            original_dtype = png_rgb16.dtype
            info = np.iinfo(original_dtype)
            value_min = float(info.min)
            value_max = float(info.max)
            image = ((png_rgb16.astype(np.float32) - value_min) / (value_max - value_min)).astype(np.float32)
            image = np.clip(image, 0.0, 1.0)
            meta = ImageMetadata(
                source_ext=path.suffix.lower(),
                original_dtype=original_dtype,
                original_channels=3,
                value_min=value_min,
                value_max=value_max,
            )
            return image, meta
    if path.suffix.lower() in {".tif", ".tiff"}:
        arr = tifffile.imread(path)
    else:
        with Image.open(path) as img:
            if img.mode in {"RGBA", "LA"}:
                arr = np.asarray(img)
                arr = arr[..., :3] if arr.ndim == 3 else arr
            elif img.mode in {"P", "CMYK"}:
                img = img.convert("RGB")
                arr = np.asarray(img)
            else:
                arr = np.asarray(img)
    original_dtype = arr.dtype
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.ndim == 2:
        arr = arr[:, :, None]
    elif arr.ndim != 3 or arr.shape[2] not in {1, 3}:
        raise ValueError(f"Expected grayscale or RGB image, got shape {arr.shape} for {path}")
    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        value_min = float(info.min)
        value_max = float(info.max)
        image = ((arr.astype(np.float32) - value_min) / (value_max - value_min)).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)
        meta = ImageMetadata(
            source_ext=path.suffix.lower(),
            original_dtype=original_dtype,
            original_channels=int(image.shape[2]),
            value_min=value_min,
            value_max=value_max,
        )
        return image, meta
    arr_f = arr.astype(np.float32)
    finite = np.isfinite(arr_f)
    if finite.any():
        lo = float(np.min(arr_f[finite]))
        hi = float(np.max(arr_f[finite]))
    else:
        lo, hi = 0.0, 1.0
    if hi <= lo:
        hi = lo + 1.0
    image = np.nan_to_num((arr_f - lo) / (hi - lo), nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    meta = ImageMetadata(
        source_ext=path.suffix.lower(),
        original_dtype=original_dtype,
        original_channels=int(image.shape[2]),
        norm_lo=np.full((image.shape[2],), lo, dtype=np.float32),
        norm_hi=np.full((image.shape[2],), hi, dtype=np.float32),
    )
    return image, meta


def read_fits_image(path: Path) -> Tuple[np.ndarray, ImageMetadata]:
    try:
        from astropy.io import fits as astrofits
    except ImportError:
        raise ImportError("astropy is required for FITS support: pip install astropy")

    with astrofits.open(path) as hdul:
        hdu = next((h for h in hdul if h.data is not None), None)
        if hdu is None:
            raise ValueError(f"No image data found in FITS file: {path}")
        data = hdu.data.copy()

    # FITS data may be big-endian; convert to native byte order
    if not data.dtype.isnative:
        data = data.astype(data.dtype.newbyteorder('='))

    original_dtype = data.dtype

    # FITS stores as (H, W) grayscale or (C, H, W) multi-channel
    if data.ndim == 2:
        arr_hwc = data[:, :, np.newaxis]
    elif data.ndim == 3:
        if data.shape[0] in {1, 3}:
            arr_hwc = np.transpose(data, (1, 2, 0))  # CHW -> HWC
        elif data.shape[2] in {1, 3}:
            arr_hwc = data
        else:
            raise ValueError(f"Unsupported FITS image shape: {data.shape}")
    else:
        raise ValueError(f"Unsupported FITS image ndim={data.ndim} (shape {data.shape})")

    n_channels = arr_hwc.shape[2]

    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        value_min = float(info.min)
        value_max = float(info.max)
        image = ((arr_hwc.astype(np.float32) - value_min) / (value_max - value_min)).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)
        return image, ImageMetadata(
            source_ext=path.suffix.lower(),
            original_dtype=original_dtype,
            original_channels=n_channels,
            value_min=value_min,
            value_max=value_max,
        )

    arr_f = arr_hwc.astype(np.float32)
    finite = np.isfinite(arr_f)
    if finite.any():
        lo = float(np.min(arr_f[finite]))
        hi = float(np.max(arr_f[finite]))
    else:
        lo, hi = 0.0, 1.0
    if hi <= lo:
        hi = lo + 1.0
    image = np.nan_to_num((arr_f - lo) / (hi - lo), nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    return image, ImageMetadata(
        source_ext=path.suffix.lower(),
        original_dtype=original_dtype,
        original_channels=n_channels,
        norm_lo=np.full((n_channels,), lo, dtype=np.float32),
        norm_hi=np.full((n_channels,), hi, dtype=np.float32),
    )


def write_fits_image(path: Path, image_chw: np.ndarray, meta: ImageMetadata) -> None:
    try:
        from astropy.io import fits as astrofits
    except ImportError:
        raise ImportError("astropy is required for FITS support: pip install astropy")

    image = np.clip(image_chw, 0.0, 1.0)

    if (
        meta.value_min is not None
        and meta.value_max is not None
        and np.issubdtype(meta.original_dtype, np.integer)
    ):
        restored = image * (meta.value_max - meta.value_min) + meta.value_min
        arr_chw = np.clip(np.rint(restored), meta.value_min, meta.value_max).astype(meta.original_dtype)
    elif meta.norm_lo is not None and meta.norm_hi is not None:
        arr_chw = restore_original_range(image, meta).astype(meta.original_dtype)
    else:
        arr_chw = image.astype(np.float32)

    # Write as (H, W) for grayscale or (C, H, W) for multi-channel
    if arr_chw.shape[0] == 1:
        fits_data = arr_chw[0]
    else:
        fits_data = arr_chw

    hdu = astrofits.PrimaryHDU(fits_data)
    astrofits.HDUList([hdu]).writeto(path, overwrite=True)


def read_image(path: Path) -> Tuple[np.ndarray, ImageMetadata]:
    ext = path.suffix.lower()
    if ext in {".fits", ".fit"}:
        return read_fits_image(path)
    if ext in {".png", ".tif", ".tiff", ".jpg", ".jpeg"}:
        return read_standard_image(path)
    raise ValueError(
        f"Unsupported extension: {path.suffix} ({path}). "
        "Supported formats: FITS, PNG, TIFF, JPG, JPEG."
    )


def read_image_pair(path_clean: Path, path_noisy: Path) -> Tuple[np.ndarray, np.ndarray]:
    clean, clean_meta = read_image(path_clean)
    noisy, noisy_meta = read_image(path_noisy)
    if clean.shape != noisy.shape:
        raise ValueError(f"Expected matching image pair shapes, got {clean.shape} and {noisy.shape}")

    needs_pair_norm = (
        np.issubdtype(clean_meta.original_dtype, np.floating)
        and np.issubdtype(noisy_meta.original_dtype, np.floating)
        and clean_meta.value_min is None
        and noisy_meta.value_min is None
    )
    if not needs_pair_norm:
        return clean, noisy

    clean_raw = restore_original_range(np.transpose(clean, (2, 0, 1)), clean_meta)
    noisy_raw = restore_original_range(np.transpose(noisy, (2, 0, 1)), noisy_meta)
    clean_hwc = np.transpose(clean_raw, (1, 2, 0))
    noisy_hwc = np.transpose(noisy_raw, (1, 2, 0))
    lo, hi = compute_pair_normalization_stats(noisy_hwc, clean_hwc)
    clean_norm = normalize_with_stats(clean_hwc, lo, hi)
    noisy_norm = normalize_with_stats(noisy_hwc, lo, hi)
    return clean_norm, noisy_norm


def restore_original_range(image: np.ndarray, meta: ImageMetadata) -> np.ndarray:
    if meta.norm_lo is None or meta.norm_hi is None:
        return image
    lo = meta.norm_lo.astype(np.float32)
    hi = meta.norm_hi.astype(np.float32)
    if image.shape[0] == 1:
        restored = image * (hi[0] - lo[0]) + lo[0]
        return restored.astype(np.float32)
    restored = np.empty_like(image, dtype=np.float32)
    for c in range(image.shape[0]):
        restored[c] = image[c] * (hi[c] - lo[c]) + lo[c]
    return restored.astype(np.float32)


def _write_png_chunk(handle, chunk_type: bytes, payload: bytes) -> None:
    handle.write(struct.pack(">I", len(payload)))
    handle.write(chunk_type)
    handle.write(payload)
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(payload, crc)
    handle.write(struct.pack(">I", crc & 0xFFFFFFFF))


def write_png_rgb16(path: Path, image_hwc: np.ndarray) -> None:
    if image_hwc.dtype != np.uint16:
        raise ValueError("write_png_rgb16 expects a uint16 RGB array")
    if image_hwc.ndim != 3 or image_hwc.shape[2] != 3:
        raise ValueError(f"write_png_rgb16 expects an HWC RGB array, got {image_hwc.shape}")

    height, width, _ = image_hwc.shape
    image_be = np.ascontiguousarray(image_hwc.astype(">u2", copy=False))
    raw_scanlines = b"".join(b"\x00" + image_be[y].tobytes() for y in range(height))
    compressed = zlib.compress(raw_scanlines, level=6)
    ihdr = struct.pack(">IIBBBBB", width, height, 16, 2, 0, 0, 0)

    with path.open("wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        _write_png_chunk(handle, b"IHDR", ihdr)
        _write_png_chunk(handle, b"IDAT", compressed)
        _write_png_chunk(handle, b"IEND", b"")


def write_image(path: Path, image_chw: np.ndarray, meta: ImageMetadata) -> None:
    image = np.clip(image_chw, 0.0, 1.0)
    output_ext = path.suffix.lower()
    if output_ext in {".fits", ".fit"}:
        write_fits_image(path, image_chw, meta)
        return
    supported_output_exts = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
    if output_ext not in supported_output_exts:
        raise ValueError(
            f"Unsupported output extension: {path.suffix} ({path}). "
            "Supported output formats: FITS, PNG, TIFF, JPG, JPEG."
        )

    if image.shape[0] == 1:
        hw = image[0]
        if (
            meta.value_min is not None
            and meta.value_max is not None
            and np.issubdtype(meta.original_dtype, np.integer)
        ):
            restored = hw * (meta.value_max - meta.value_min) + meta.value_min
            arr = np.clip(np.rint(restored), meta.value_min, meta.value_max).astype(meta.original_dtype)
            if arr.dtype == np.uint16 and output_ext in {".tif", ".tiff", ".png"}:
                Image.fromarray(arr).save(path)
                return
            if arr.dtype == np.uint8:
                Image.fromarray(arr, mode="L").save(path)
                return
            Image.fromarray(arr).save(path)
            return
        restored = restore_original_range(image, meta)[0]
        if np.issubdtype(meta.original_dtype, np.floating):
            Image.fromarray(restored.astype(np.float32)).save(path)
            return
        Image.fromarray(np.clip(restored, 0.0, 255.0).round().astype(np.uint8), mode="L").save(path)
        return

    if image.shape[0] == 3:
        hwc = np.transpose(image, (1, 2, 0))
        if (
            meta.value_min is not None
            and meta.value_max is not None
            and np.issubdtype(meta.original_dtype, np.integer)
        ):
            restored = hwc * (meta.value_max - meta.value_min) + meta.value_min
            arr = np.clip(np.rint(restored), meta.value_min, meta.value_max).astype(meta.original_dtype)
            if arr.dtype == np.uint16 and output_ext == ".png":
                write_png_rgb16(path, arr)
                return
            if arr.dtype == np.uint16 and output_ext in {".tif", ".tiff"}:
                # PIL can't save 16-bit RGB TIFF; use tifffile.
                tifffile.imwrite(str(path), arr)
                return
            Image.fromarray(arr).save(path)
            return
        restored = np.transpose(restore_original_range(image, meta), (1, 2, 0))
        if np.issubdtype(meta.original_dtype, np.floating):
            Image.fromarray(restored.astype(np.float32)).save(path)
            return
        Image.fromarray(np.clip(restored, 0.0, 255.0).round().astype(np.uint8), mode="RGB").save(path)
        return

    raise ValueError(f"Unsupported channel count for output: {image.shape[0]}")


def adapt_channels_for_model(image_chw: np.ndarray, expected_channels: int) -> np.ndarray:
    current_channels = int(image_chw.shape[0])
    if current_channels == expected_channels:
        return image_chw.astype(np.float32, copy=False)
    if current_channels == 1 and expected_channels == 3:
        return np.repeat(image_chw, 3, axis=0).astype(np.float32, copy=False)
    if current_channels == 3 and expected_channels == 1:
        gray = np.mean(image_chw, axis=0, keepdims=True)
        return gray.astype(np.float32, copy=False)
    raise ValueError(
        f"Unsupported channel conversion for inference: input={current_channels}, model={expected_channels}"
    )


def adapt_channels_for_output(image_chw: np.ndarray, target_channels: int) -> np.ndarray:
    current_channels = int(image_chw.shape[0])
    if current_channels == target_channels:
        return image_chw.astype(np.float32, copy=False)
    if current_channels == 1 and target_channels == 3:
        return np.repeat(image_chw, 3, axis=0).astype(np.float32, copy=False)
    if current_channels == 3 and target_channels == 1:
        gray = np.mean(image_chw, axis=0, keepdims=True)
        return gray.astype(np.float32, copy=False)
    raise ValueError(
        f"Unsupported output channel conversion for inference: output={current_channels}, target={target_channels}"
    )


def signal_luma(image_chw: np.ndarray) -> np.ndarray:
    """Brightness proxy for astronomy data; max-channel keeps narrowband signal visible."""
    image = image_chw.astype(np.float32, copy=False)
    if image.shape[0] == 1:
        return image[0]
    return np.max(image, axis=0)


def mean_filter_2d(image: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return image.astype(np.float32, copy=False)
    image_f = image.astype(np.float32, copy=False)
    padded = np.pad(image_f, radius, mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant")
    integral = np.cumsum(np.cumsum(integral, axis=0), axis=1)
    diameter = radius * 2 + 1
    area = float(diameter * diameter)
    out = (
        integral[diameter:, diameter:]
        - integral[:-diameter, diameter:]
        - integral[diameter:, :-diameter]
        + integral[:-diameter, :-diameter]
    )
    return (out / area).astype(np.float32, copy=False)


def mean_filter_chw(image_chw: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return image_chw.astype(np.float32, copy=False)
    return np.stack([mean_filter_2d(channel, radius) for channel in image_chw], axis=0).astype(np.float32)


def smoothstep_mask(value: np.ndarray, low: float, high: float) -> np.ndarray:
    if high <= low:
        return (value >= high).astype(np.float32)
    t = np.clip((value - low) / (high - low), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32, copy=False)


def to_tensor_chw(image_hwc: np.ndarray) -> "torch.Tensor":
    return torch.from_numpy(np.transpose(image_hwc, (2, 0, 1))).float()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def apply_pair_augmentation(noisy: np.ndarray, clean: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        noisy = np.flip(noisy, axis=1)
        clean = np.flip(clean, axis=1)
    if rng.random() < 0.5:
        noisy = np.flip(noisy, axis=0)
        clean = np.flip(clean, axis=0)
    k = int(rng.integers(0, 4))
    if k:
        noisy = np.rot90(noisy, k=k, axes=(0, 1))
        clean = np.rot90(clean, k=k, axes=(0, 1))
    return noisy.copy(), clean.copy()


def add_astronomical_corruption(
    clean: np.ndarray,
    rng: np.random.Generator,
    gaussian_sigma_range: Tuple[float, float] = (0.002, 0.03),
    poisson_scale_range: Tuple[float, float] = (24.0, 180.0),
    hot_pixel_prob: float = 0.0015,
    background_std_range: Tuple[float, float] = (0.0, 0.02),
) -> np.ndarray:
    noisy = clean.astype(np.float32).copy()

    if poisson_scale_range[1] > 0:
        poisson_scale = float(rng.uniform(*poisson_scale_range))
        counts = np.clip(noisy * poisson_scale, 0.0, None)
        noisy = rng.poisson(counts).astype(np.float32) / poisson_scale

    sigma = float(rng.uniform(*gaussian_sigma_range))
    noisy += rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32)

    bg_std = float(rng.uniform(*background_std_range))
    if bg_std > 0:
        h, w = noisy.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        yy = yy / max(h - 1, 1)
        xx = xx / max(w - 1, 1)
        plane = (
            rng.normal(0.0, bg_std) * xx
            + rng.normal(0.0, bg_std) * yy
            + rng.normal(0.0, bg_std * 0.5)
        )
        if noisy.ndim == 3:
            plane = plane[..., None]
        noisy += plane.astype(np.float32)

    mask = rng.random(size=noisy.shape[:2]) < hot_pixel_prob
    if np.any(mask):
        if noisy.ndim == 2:
            noisy[mask] = np.maximum(noisy[mask], rng.uniform(0.8, 1.0, size=mask.sum()).astype(np.float32))
        else:
            for ch in range(noisy.shape[2]):
                boost = rng.uniform(0.8, 1.0, size=mask.sum()).astype(np.float32)
                noisy[..., ch][mask] = np.maximum(noisy[..., ch][mask], boost)

    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


class ImagePairsDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        augment: bool = False,
        synth_mix_prob: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.hr_dir = root_dir / "HR"
        self.lr_dir = root_dir / "LR"
        self.augment = augment
        self.synth_mix_prob = synth_mix_prob
        self.rng = np.random.default_rng(seed)
        if not self.hr_dir.exists() or not self.lr_dir.exists():
            raise FileNotFoundError(f"Expected folders: {self.hr_dir} and {self.lr_dir}")

        hr_files = sorted([p for p in self.hr_dir.iterdir() if p.is_file()])
        self.pairs: List[Tuple[Path, Path]] = []
        for hr_path in hr_files:
            lr_path = self.lr_dir / hr_path.name
            if lr_path.exists():
                self.pairs.append((hr_path, lr_path))

        if not self.pairs:
            raise RuntimeError(
                f"No matching LR/HR pairs found in {root_dir}. "
                "Ensure same filenames exist in both LR and HR."
            )

        # Infer channels from first valid pair and keep only matching pairs.
        first_lr, _ = read_image(self.pairs[0][1])
        self.channels = int(first_lr.shape[2])
        valid_pairs: List[Tuple[Path, Path]] = []
        for hr, lr in self.pairs:
            lr_img, _ = read_image(lr)
            hr_img, _ = read_image(hr)
            if lr_img.shape != hr_img.shape:
                continue
            if lr_img.shape[2] != self.channels:
                continue
            valid_pairs.append((hr, lr))
        self.pairs = valid_pairs
        if not self.pairs:
            raise RuntimeError("No valid shape/channel-consistent image pairs found.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
        hr_path, lr_path = self.pairs[idx]
        clean, noisy = read_image_pair(hr_path, lr_path)
        if self.augment:
            noisy, clean = apply_pair_augmentation(noisy, clean, self.rng)
            if self.synth_mix_prob > 0 and self.rng.random() < self.synth_mix_prob:
                synth_noisy = add_astronomical_corruption(clean, self.rng)
                mix = float(self.rng.uniform(0.35, 0.75))
                noisy = np.clip(noisy * (1.0 - mix) + synth_noisy * mix, 0.0, 1.0).astype(np.float32)
        clean_t = to_tensor_chw(clean)
        noisy_t = to_tensor_chw(noisy)
        return noisy_t, clean_t


if nn is not None:
    class LayerNorm2d(nn.Module):
        """Per-pixel channel LayerNorm, evaluated in float32 for AMP stability."""

        def __init__(self, channels: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
            self.eps = eps

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            dtype = x.dtype
            xf = x.float()
            mean = xf.mean(dim=1, keepdim=True)
            var = (xf - mean).pow(2).mean(dim=1, keepdim=True)
            xf = (xf - mean) / torch.sqrt(var + self.eps)
            return (xf * self.weight + self.bias).to(dtype)


    class SimpleGate(nn.Module):
        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x1, x2 = x.chunk(2, dim=1)
            return x1 * x2


    class DropPath(nn.Module):
        """Stochastic depth: randomly zeroes the residual branch per sample."""

        def __init__(self, drop_prob: float = 0.0) -> None:
            super().__init__()
            self.drop_prob = float(drop_prob)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            if self.drop_prob <= 0.0 or not self.training:
                return x
            keep = 1.0 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
            return x * mask / keep


    class NAFBlock(nn.Module):
        def __init__(
            self,
            channels: int,
            dw_expand: int = 2,
            ffn_expand: int = 2,
            dropout: float = 0.0,
            drop_path: float = 0.0,
            residual_scale_init: float = 1e-2,
        ) -> None:
            super().__init__()
            dw_channels = channels * dw_expand
            ffn_channels = channels * ffn_expand
            self.norm1 = LayerNorm2d(channels)
            self.pw1 = nn.Conv2d(channels, dw_channels, kernel_size=1)
            self.dw = nn.Conv2d(dw_channels, dw_channels, kernel_size=3, padding=1, groups=dw_channels)
            self.sg = SimpleGate()
            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dw_channels // 2, dw_channels // 2, kernel_size=1),
            )
            self.pw2 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1)
            self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.beta = nn.Parameter(torch.full((1, channels, 1, 1), residual_scale_init))

            self.norm2 = LayerNorm2d(channels)
            self.ffn1 = nn.Conv2d(channels, ffn_channels, kernel_size=1)
            self.ffn_sg = SimpleGate()
            self.ffn2 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1)
            self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), residual_scale_init))
            self.drop_path = DropPath(drop_path)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            y = self.norm1(x)
            y = self.pw1(y)
            y = self.dw(y)
            y = self.sg(y)
            y = y * self.sca(y)
            y = self.pw2(y)
            y = self.dropout1(y)
            x = x + self.drop_path(y * self.beta)

            y = self.norm2(x)
            y = self.ffn1(y)
            y = self.ffn_sg(y)
            y = self.ffn2(y)
            y = self.dropout2(y)
            return x + self.drop_path(y * self.gamma)


    class Downsample(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            # Pure strided conv (no activation): fewer ops, no GELU dead-zone -> steadier gradients.
            self.body = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.body(x)


    class Upsample(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, bias=False),
                nn.PixelShuffle(2),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.body(x)


    class AstroUNet(nn.Module):
        def __init__(
            self,
            channels: int,
            width: int = 48,
            encoder_blocks: Sequence[int] = (2, 2, 2, 4),
            bottleneck_blocks: int = 10,
            dropout: float = 0.0,
            drop_path_rate: float = 0.05,
        ) -> None:
            super().__init__()
            encoder_blocks = tuple(int(b) for b in encoder_blocks)
            self.intro = nn.Conv2d(channels, width, kernel_size=3, padding=1)
            self.ending = nn.Conv2d(width, channels, kernel_size=3, padding=1)

            self.encoders = nn.ModuleList()
            self.downs = nn.ModuleList()
            self.decoders = nn.ModuleList()
            self.ups = nn.ModuleList()
            self.skip_projs = nn.ModuleList()

            # Linearly increasing stochastic-depth across encoder -> bottleneck -> decoder.
            total_blocks = 2 * sum(encoder_blocks) + bottleneck_blocks
            dpr = [float(v) for v in torch.linspace(0.0, drop_path_rate, max(total_blocks, 1))]
            bi = 0

            chan = width
            skip_channels: List[int] = []
            for blocks in encoder_blocks:
                self.encoders.append(
                    nn.Sequential(*[NAFBlock(chan, dropout=dropout, drop_path=dpr[bi + k]) for k in range(blocks)])
                )
                bi += blocks
                skip_channels.append(chan)
                self.downs.append(Downsample(chan, chan * 2))
                chan *= 2

            self.middle = nn.Sequential(
                *[NAFBlock(chan, dropout=dropout, drop_path=dpr[bi + k]) for k in range(bottleneck_blocks)]
            )
            bi += bottleneck_blocks

            for blocks, skip_chan in zip(reversed(encoder_blocks), reversed(skip_channels)):
                self.ups.append(Upsample(chan, skip_chan))
                self.skip_projs.append(nn.Conv2d(skip_chan * 2, skip_chan, kernel_size=1))
                chan = skip_chan
                self.decoders.append(
                    nn.Sequential(*[NAFBlock(chan, dropout=dropout, drop_path=dpr[bi + k]) for k in range(blocks)])
                )
                bi += blocks

            self.apply(self._init_weights)

        @staticmethod
        def _init_weights(m: "nn.Module") -> None:
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            inp = x
            x = self.intro(x)
            skips: List[torch.Tensor] = []
            for encoder, down in zip(self.encoders, self.downs):
                x = encoder(x)
                skips.append(x)
                x = down(x)

            x = self.middle(x)

            for up, proj, decoder, skip in zip(self.ups, self.skip_projs, self.decoders, reversed(skips)):
                x = up(x)
                if x.shape[-2:] != skip.shape[-2:]:
                    x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                x = proj(torch.cat([x, skip], dim=1))
                x = decoder(x)

            residual_noise = self.ending(x)
            return inp - residual_noise
else:
    class AstroUNet:
        def __init__(self, *args: object, **kwargs: object) -> None:
            require_torch()


def build_weight_window(patch_size: int, channels: int) -> np.ndarray:
    base = np.hanning(patch_size).astype(np.float32)
    win2d = np.outer(base, base).astype(np.float32)
    win2d = np.clip(win2d, 1e-3, None)
    return np.repeat(win2d[None, :, :], channels, axis=0)


def charbonnier_loss(pred: "torch.Tensor", target: "torch.Tensor", eps: float = 1e-3) -> "torch.Tensor":
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps**2))


def gradient_loss(pred: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def fft_loss(pred: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    pred_fft = torch.fft.rfft2(pred, norm="ortho")
    target_fft = torch.fft.rfft2(target, norm="ortho")
    return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))


def tv_loss(pred: "torch.Tensor") -> "torch.Tensor":
    """Total variation loss — penalises spatial roughness in the prediction.

    Useful when the prediction should be smooth (e.g. background-gradient maps).
    """
    dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    return torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))


def torch_signal_luma(image: "torch.Tensor") -> "torch.Tensor":
    if image.shape[1] == 1:
        return image
    return torch.amax(image, dim=1, keepdim=True)


def torch_mean_filter(image: "torch.Tensor", radius: int) -> "torch.Tensor":
    if radius <= 0:
        return image
    kernel_size = radius * 2 + 1
    padded = F.pad(image, (radius, radius, radius, radius), mode="reflect")
    return F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)


def faint_signal_loss(
    pred: "torch.Tensor",
    target: "torch.Tensor",
    weight: float,
) -> "torch.Tensor":
    if weight <= 0:
        return pred.new_tensor(0.0)

    target_luma = torch_signal_luma(target)
    local_background = torch_mean_filter(target_luma, radius=7)
    local_signal = torch.clamp(target_luma - local_background, min=0.0)
    signal_mask = torch.sigmoid((local_signal - 0.01) / 0.004).detach()

    weighted_reconstruction = torch.mean(torch.sqrt((pred - target) ** 2 + 1e-6) * signal_mask)
    dimming = torch.relu(target - pred)
    under_preservation = torch.mean(dimming * signal_mask)
    return (weighted_reconstruction + under_preservation) * weight


def gaussian_kernel(device: "torch.device", channels: int, kernel_size: int = 11, sigma: float = 1.5) -> "torch.Tensor":
    coords = torch.arange(kernel_size, device=device, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel_2d = torch.outer(g, g)
    return kernel_2d.expand(channels, 1, kernel_size, kernel_size).contiguous()


def ssim_loss(pred: "torch.Tensor", target: "torch.Tensor", kernel_size: int = 11, sigma: float = 1.5) -> "torch.Tensor":
    channels = pred.shape[1]
    kernel = gaussian_kernel(pred.device, channels, kernel_size=kernel_size, sigma=sigma)
    padding = kernel_size // 2
    mu_x = F.conv2d(pred, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(target, kernel, padding=padding, groups=channels)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, kernel, padding=padding, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(target * target, kernel, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(pred * target, kernel, padding=padding, groups=channels) - mu_xy

    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-8)
    return 1.0 - ssim_map.mean()


def combined_loss(
    pred: "torch.Tensor",
    target: "torch.Tensor",
    ssim_weight: float,
    fft_weight: float,
    gradient_weight: float,
    tv_weight: float = 0.0,
    faint_signal_weight: float = 0.0,
) -> "torch.Tensor":
    loss = charbonnier_loss(pred, target)
    if ssim_weight > 0:
        loss = loss + ssim_weight * ssim_loss(pred, target)
    if fft_weight > 0:
        loss = loss + fft_weight * fft_loss(pred, target)
    if gradient_weight > 0:
        loss = loss + gradient_weight * gradient_loss(pred, target)
    if tv_weight > 0:
        loss = loss + tv_weight * tv_loss(pred)
    if faint_signal_weight > 0:
        loss = loss + faint_signal_loss(pred, target, faint_signal_weight)
    return loss


def compute_mae(pred: "torch.Tensor", target: "torch.Tensor") -> float:
    return float(torch.mean(torch.abs(pred - target)).item())


def compute_rmse(pred: "torch.Tensor", target: "torch.Tensor") -> float:
    return float(torch.sqrt(torch.mean((pred - target) ** 2) + 1e-12).item())


def compute_ssim(pred: "torch.Tensor", target: "torch.Tensor") -> float:
    return float((1.0 - ssim_loss(pred, target)).item())


def compute_psnr(pred: "torch.Tensor", target: "torch.Tensor") -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return float(-10.0 * math.log10(mse))


def generate_patch_positions(length: int, patch_size: int, stride: int) -> List[int]:
    if length <= patch_size:
        return [0]
    positions = list(range(0, length - patch_size + 1, stride))
    if positions[-1] != length - patch_size:
        positions.append(length - patch_size)
    return positions


def apply_test_time_transform(image: np.ndarray, mode: int) -> np.ndarray:
    if mode == 0:
        return image
    if mode == 1:
        return np.flip(image, axis=2).copy()
    if mode == 2:
        return np.flip(image, axis=1).copy()
    if mode == 3:
        return np.rot90(image, k=1, axes=(1, 2)).copy()
    if mode == 4:
        return np.rot90(image, k=2, axes=(1, 2)).copy()
    if mode == 5:
        return np.rot90(image, k=3, axes=(1, 2)).copy()
    if mode == 6:
        return np.flip(np.rot90(image, k=1, axes=(1, 2)), axis=2).copy()
    if mode == 7:
        return np.flip(np.rot90(image, k=1, axes=(1, 2)), axis=1).copy()
    raise ValueError(f"Invalid TTA mode: {mode}")


def invert_test_time_transform(image: np.ndarray, mode: int) -> np.ndarray:
    if mode == 0:
        return image
    if mode == 1:
        return np.flip(image, axis=2).copy()
    if mode == 2:
        return np.flip(image, axis=1).copy()
    if mode == 3:
        return np.rot90(image, k=3, axes=(1, 2)).copy()
    if mode == 4:
        return np.rot90(image, k=2, axes=(1, 2)).copy()
    if mode == 5:
        return np.rot90(image, k=1, axes=(1, 2)).copy()
    if mode == 6:
        return np.rot90(np.flip(image, axis=2), k=3, axes=(1, 2)).copy()
    if mode == 7:
        return np.rot90(np.flip(image, axis=1), k=3, axes=(1, 2)).copy()
    raise ValueError(f"Invalid TTA mode: {mode}")


def denoise_with_patches(
    model: nn.Module,
    image_chw: np.ndarray,
    patch_size: int,
    stride: int,
    device: "torch.device",
    amp: bool = True,
    batch_size: int = 32,
    progress_callback: ProgressCallback | None = None,
    progress_range: Tuple[float, float] = (0.0, 1.0),
    progress_label: str = "Denoising",
) -> np.ndarray:
    use_amp = bool(amp and device.type == "cuda")
    model.eval()
    if device.type == "cuda" and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True

    def torch_predict(patches: np.ndarray) -> np.ndarray:
        batch_t = torch.from_numpy(patches).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=use_amp):
                denoised_t = model(batch_t)
        denoised_np = denoised_t.detach().cpu().numpy().astype(np.float32)
        return np.clip(denoised_np, 0.0, 1.0)

    return denoise_with_patch_predictor(
        predict_batch=torch_predict,
        image_chw=image_chw,
        patch_size=patch_size,
        stride=stride,
        batch_size=batch_size,
        progress_callback=progress_callback,
        progress_range=progress_range,
        progress_label=progress_label,
    )


def denoise_with_patch_predictor(
    predict_batch: PatchPredictor,
    image_chw: np.ndarray,
    patch_size: int,
    stride: int,
    batch_size: int = 32,
    progress_callback: ProgressCallback | None = None,
    progress_range: Tuple[float, float] = (0.0, 1.0),
    progress_label: str = "Denoising",
) -> np.ndarray:
    c, h, w = image_chw.shape
    if stride <= 0 or stride > patch_size:
        raise ValueError("stride must be in [1, patch_size]")

    ys = generate_patch_positions(h, patch_size, stride)
    xs = generate_patch_positions(w, patch_size, stride)
    total_patches = len(ys) * len(xs)
    weight = build_weight_window(patch_size, c)

    out = np.zeros((c, h, w), dtype=np.float32)
    acc = np.zeros((c, h, w), dtype=np.float32)

    patch_coords = [(y, x) for y in ys for x in xs]

    batches: List[List[Tuple[int, int]]] = []
    for i in range(0, total_patches, batch_size):
        batches.append(patch_coords[i : i + batch_size])

    batch_iter: object = batches
    if tqdm is not None:
        batch_iter = tqdm(batches, total=len(batches), desc="Inferencing", unit="batch")

    patches_done = 0
    for batch_coords in batch_iter:
        patches = np.stack(
            [image_chw[:, y : y + patch_size, x : x + patch_size] for y, x in batch_coords]
        ).astype(np.float32, copy=False)
        denoised_np = np.clip(predict_batch(patches), 0.0, 1.0).astype(np.float32, copy=False)
        for i, (y, x) in enumerate(batch_coords):
            out[:, y : y + patch_size, x : x + patch_size] += denoised_np[i] * weight
            acc[:, y : y + patch_size, x : x + patch_size] += weight
        patches_done += len(batch_coords)
        if progress_callback is not None:
            p_start, p_end = progress_range
            progress_callback(
                p_start + (p_end - p_start) * patches_done / max(total_patches, 1),
                f"{progress_label} patch {patches_done}/{total_patches}",
            )

    return out / np.clip(acc, 1e-6, None)


def apply_denoise_strength(original: np.ndarray, denoised: np.ndarray, strength: float) -> np.ndarray:
    """
    Blend between original and denoised result.
    - 0.0: no denoising
    - 1.0: model output
    - >1.0: stronger (can smooth more details)
    """
    if strength < 0.0:
        raise ValueError("strength must be >= 0")
    blended = original + strength * (denoised - original)
    return np.clip(blended, 0.0, 1.0).astype(np.float32)


def compute_background_mask(image_chw: np.ndarray, threshold: float) -> np.ndarray:
    """
    Build a soft mask where 1.0 means "treat as dark background" and 0.0 means
    "treat as subject / brighter structure".
    """
    if threshold <= 0.0:
        return np.zeros((1, image_chw.shape[1], image_chw.shape[2]), dtype=np.float32)
    luma = signal_luma(image_chw)[None, :, :]
    mask = np.clip((threshold - luma) / max(threshold, 1e-6), 0.0, 1.0)
    return mask.astype(np.float32, copy=False)


def compute_faint_structure_mask(
    image_chw: np.ndarray,
    background_threshold: float,
    protection: float,
) -> np.ndarray:
    if protection <= 0.0:
        return np.zeros((1, image_chw.shape[1], image_chw.shape[2]), dtype=np.float32)

    luma = signal_luma(image_chw)
    wide_background = mean_filter_2d(luma, radius=12)
    medium_background = mean_filter_2d(luma, radius=4)
    tiny_background = mean_filter_2d(luma, radius=1)

    highpass = luma - tiny_background
    mad = float(np.median(np.abs(highpass - np.median(highpass))) * 1.4826)
    if not np.isfinite(mad):
        mad = 0.0

    broad_signal = np.maximum(luma - wide_background, 0.0)
    medium_signal = np.maximum(luma - medium_background, 0.0)

    broad_low = max(0.003, 1.2 * mad)
    broad_high = max(0.018, 4.5 * mad)
    medium_low = max(0.006, 2.0 * mad)
    medium_high = max(0.030, 7.0 * mad)

    broad_mask = smoothstep_mask(broad_signal, broad_low, broad_high)
    medium_mask = smoothstep_mask(medium_signal, medium_low, medium_high)
    local_contrast = medium_signal / np.maximum(wide_background + 0.03, 0.03)
    contrast_mask = smoothstep_mask(local_contrast, low=0.04, high=0.20)
    structure = np.maximum(broad_mask, medium_mask * 0.7)
    structure = np.maximum(structure, contrast_mask * 0.6)

    faint_low = max(background_threshold * 2.0, 0.18)
    faint_high = max(background_threshold * 4.0, 0.45)
    if faint_high <= faint_low:
        faint_high = faint_low + 0.05
    faint_gate = 1.0 - smoothstep_mask(luma, faint_low, faint_high)
    structure = structure * (0.35 + 0.65 * faint_gate)

    structure = mean_filter_2d(structure, radius=2)
    return np.clip(structure * protection, 0.0, 1.0)[None, :, :].astype(np.float32, copy=False)


def apply_spatial_denoise_strength(
    original: np.ndarray,
    denoised: np.ndarray,
    background_mask: np.ndarray,
    subject_strength: float,
    background_strength: float,
) -> np.ndarray:
    if subject_strength < 0.0 or background_strength < 0.0:
        raise ValueError("subject_strength and background_strength must be >= 0")
    strength_map = subject_strength + background_mask * (background_strength - subject_strength)
    blended = original + strength_map * (denoised - original)
    return np.clip(blended, 0.0, 1.0).astype(np.float32)


def preserve_detail_residual(
    original: np.ndarray,
    denoised: np.ndarray,
    detail_preservation: float,
) -> np.ndarray:
    """
    Re-inject a controlled amount of high-frequency residual from the source image.
    - 0.0: keep pure denoised output
    - 1.0: aggressively keep original micro-contrast/details
    """
    if not 0.0 <= detail_preservation <= 1.0:
        raise ValueError("detail_preservation must be in [0, 1]")
    detail = original - denoised
    preserved = denoised + detail * detail_preservation
    return np.clip(preserved, 0.0, 1.0).astype(np.float32)


def preserve_detail_residual_spatial(
    original: np.ndarray,
    denoised: np.ndarray,
    background_mask: np.ndarray,
    subject_detail_preservation: float,
    background_detail_preservation: float,
) -> np.ndarray:
    if not 0.0 <= subject_detail_preservation <= 1.0:
        raise ValueError("subject_detail_preservation must be in [0, 1]")
    if not 0.0 <= background_detail_preservation <= 1.0:
        raise ValueError("background_detail_preservation must be in [0, 1]")
    detail_map = (
        subject_detail_preservation
        + background_mask * (background_detail_preservation - subject_detail_preservation)
    )
    detail = original - denoised
    preserved = denoised + detail * detail_map
    return np.clip(preserved, 0.0, 1.0).astype(np.float32)


def preserve_faint_signal(
    original: np.ndarray,
    denoised: np.ndarray,
    structure_mask: np.ndarray,
    preservation: float,
    boost: float,
) -> np.ndarray:
    if preservation < 0.0:
        raise ValueError("faint_signal_preservation must be >= 0")
    if boost < 0.0:
        raise ValueError("faint_signal_boost must be >= 0")

    mask = np.clip(structure_mask, 0.0, 1.0)
    protected = denoised
    if preservation > 0.0:
        dimmed_signal = np.maximum(original - denoised, 0.0)
        orig_background = mean_filter_chw(original, radius=8)
        den_background = mean_filter_chw(denoised, radius=8)
        local_contrast_loss = np.maximum(
            (original - orig_background) - (denoised - den_background),
            0.0,
        )
        restored = np.maximum(dimmed_signal, local_contrast_loss)
        protected = protected + restored * mask * preservation

    if boost > 0.0:
        local_background = mean_filter_chw(protected, radius=8)
        local_signal = np.maximum(protected - local_background, 0.0)
        protected = protected + local_signal * mask * boost

    return np.clip(protected, 0.0, 1.0).astype(np.float32)


def _tta_mode_swaps_dims(mode: int) -> bool:
    """True for TTA modes that transpose H and W (90° / 270° rotations)."""
    return mode in (3, 5, 6, 7)


def run_tta_denoise(
    model: nn.Module,
    image_chw: np.ndarray,
    patch_size: int,
    stride: int,
    device: "torch.device",
    tta: int,
    amp: bool,
    batch_size: int = 32,
    progress_callback: ProgressCallback | None = None,
) -> np.ndarray:
    """Run inference with test-time augmentation.

    All TTA modes that share the same effective image dimensions are batched
    into a single forward pass per patch position, so the GPU handles
    ``batch_size * n_modes_in_group`` patches at once instead of running
    ``n_modes`` separate loops.

    On CUDA, if there is enough free VRAM, patch accumulation is done with
    GPU ``scatter_add_`` instead of a Python loop over patches. This removes
    the main CPU-side bottleneck for large images (6000 px and above).
    """
    modes = list(range(max(1, min(tta, 8))))
    use_amp = bool(amp and device.type == "cuda")
    model.eval()
    if device.type == "cuda" and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True

    c, h, w = image_chw.shape
    weight = build_weight_window(patch_size, c)

    # GPU weight tensor — allocated once, reused across all groups and batches
    weight_t: "torch.Tensor | None" = (
        torch.from_numpy(weight).to(device=device, dtype=torch.float32)
        if device.type == "cuda"
        else None
    )

    # Pre-apply all TTA transforms once (cheap CPU flips/rotations on the full image)
    transformed_images: List[Tuple[int, np.ndarray]] = [
        (mode, apply_test_time_transform(image_chw, mode)) for mode in modes
    ]

    # Group by effective image shape so patches can be batched together.
    # Non-rotating modes keep (H, W); 90°/270° rotation modes produce (W, H).
    group_same: List[Tuple[int, np.ndarray]] = []
    group_swap: List[Tuple[int, np.ndarray]] = []
    for mode, timg in transformed_images:
        (group_swap if _tta_mode_swaps_dims(mode) else group_same).append((mode, timg))

    # Total unique patch positions across all groups (for progress reporting)
    total_patches_all = 0
    for group in (group_same, group_swap):
        if not group:
            continue
        _, first = group[0]
        gh, gw = first.shape[1], first.shape[2]
        total_patches_all += (
            len(generate_patch_positions(gh, patch_size, stride))
            * len(generate_patch_positions(gw, patch_size, stride))
        )

    mode_results: Dict[int, np.ndarray] = {}
    patches_done_all = 0

    for group in (group_same, group_swap):
        if not group:
            continue
        _, first = group[0]
        gh, gw = first.shape[1], first.shape[2]
        ys = generate_patch_positions(gh, patch_size, stride)
        xs = generate_patch_positions(gw, patch_size, stride)
        total_g = len(ys) * len(xs)
        patch_coords = [(y, x) for y in ys for x in xs]

        # --- Choose accumulation strategy ---
        # GPU path: keep out/acc on GPU; use scatter_add_ (vectorized, no Python loop).
        # Falls back to CPU if not on CUDA or if VRAM is too tight.
        use_gpu_accu = False
        if device.type == "cuda":
            # outs + accs for every mode in this group, float32
            required_bytes = c * gh * gw * 4 * len(group) * 2
            try:
                free_bytes, _ = torch.cuda.mem_get_info(device)
                use_gpu_accu = free_bytes > required_bytes * 1.5  # keep 50 % headroom
            except Exception:
                pass

        if use_gpu_accu:
            outs_t = {mode: torch.zeros(c, gh, gw, device=device) for mode, _ in group}
            accs_t = {mode: torch.zeros(c, gh, gw, device=device) for mode, _ in group}
            # Relative flat offset for each (c, dy, dx) pixel within a patch.
            # full_flat_idx[patch_j, c, dy, dx]
            #   = c * gh * gw + (y_j + dy) * gw + (x_j + dx)
            #   = rel_offsets[c, dy, dx]  +  y_j * gw  +  x_j
            rel_offsets = (
                torch.arange(c, device=device, dtype=torch.long)[:, None, None] * (gh * gw)
                + torch.arange(patch_size, device=device, dtype=torch.long)[None, :, None] * gw
                + torch.arange(patch_size, device=device, dtype=torch.long)[None, None, :]
            ).reshape(-1)  # (C * patch_size^2,)
        else:
            outs = {mode: np.zeros((c, gh, gw), dtype=np.float32) for mode, _ in group}
            accs = {mode: np.zeros((c, gh, gw), dtype=np.float32) for mode, _ in group}

        for i in range(0, total_g, batch_size):
            batch_coords = patch_coords[i : i + batch_size]
            B = len(batch_coords)

            # Stack patches from every mode in this group → one big batch
            # Shape: (B * n_group, C, patch_size, patch_size)
            all_patches = np.concatenate(
                [
                    np.stack(
                        [timg[:, y : y + patch_size, x : x + patch_size] for y, x in batch_coords]
                    ).astype(np.float32, copy=False)
                    for _, timg in group
                ],
                axis=0,
            )

            batch_t = torch.from_numpy(all_patches).to(device=device, dtype=torch.float32)
            with torch.no_grad():
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    denoised_t = model(batch_t)

            if use_gpu_accu:
                denoised_t = denoised_t.clamp(0.0, 1.0)

                # Flat base offset per patch: y * gw + x  (same across all modes)
                ys_b = torch.tensor([y for y, x in batch_coords], device=device, dtype=torch.long)
                xs_b = torch.tensor([x for y, x in batch_coords], device=device, dtype=torch.long)
                # full_idx[j, pixel] = rel_offsets[pixel] + y_j * gw + x_j
                full_idx = (
                    rel_offsets.unsqueeze(0) + (ys_b * gw + xs_b).unsqueeze(1)
                ).reshape(-1)  # (B * C * patch_size^2,)

                # Weight repeated for every patch in the batch (same per-patch weight)
                w_flat = weight_t.reshape(1, -1).expand(B, -1).reshape(-1)  # type: ignore[union-attr]

                for mi, (mode, _) in enumerate(group):
                    mode_out = denoised_t[mi * B : (mi + 1) * B]  # (B, C, ps, ps)
                    outs_t[mode].reshape(-1).scatter_add_(
                        0, full_idx, (mode_out * weight_t).reshape(-1)  # type: ignore[operator]
                    )
                    accs_t[mode].reshape(-1).scatter_add_(0, full_idx, w_flat)
            else:
                denoised_np = np.clip(
                    denoised_t.detach().cpu().numpy().astype(np.float32), 0.0, 1.0
                )
                for mi, (mode, _) in enumerate(group):
                    mode_out = denoised_np[mi * B : (mi + 1) * B]
                    for j, (y, x) in enumerate(batch_coords):
                        outs[mode][:, y : y + patch_size, x : x + patch_size] += mode_out[j] * weight
                        accs[mode][:, y : y + patch_size, x : x + patch_size] += weight

            patches_done_all += B
            if progress_callback is not None:
                progress_callback(
                    patches_done_all / max(total_patches_all, 1),
                    f"Denoising patch {patches_done_all}/{total_patches_all}",
                )

        for mode, _ in group:
            if use_gpu_accu:
                out_np = outs_t[mode].cpu().numpy()
                acc_np = accs_t[mode].cpu().numpy()
            else:
                out_np = outs[mode]
                acc_np = accs[mode]
            mode_results[mode] = invert_test_time_transform(
                out_np / np.clip(acc_np, 1e-6, None), mode
            )

    return np.mean([mode_results[mode] for mode in modes], axis=0).astype(np.float32)


def encoder_blocks_from_args(args: argparse.Namespace) -> Tuple[int, ...]:
    """Per-level encoder block counts for a UNet of ``args.num_levels`` levels.

    Shallow levels get ``enc_blocks`` blocks; the deepest level gets extra capacity
    where the receptive field is largest and features are cheapest (smallest spatial).
    """
    num_levels = int(getattr(args, "num_levels", 4))
    enc = int(args.enc_blocks)
    # Blocks double from the 2nd level down: cheap at low resolution, large receptive
    # field where it matters. enc=2, levels=4 -> (2, 2, 4, 8) (NAFNet-proven schedule).
    return tuple(enc * (2 ** max(0, i - 1)) for i in range(num_levels))


def build_model(args: argparse.Namespace, channels: int) -> nn.Module:
    if str(getattr(args, "arch", "full")) == "lite":
        return build_lite_model(args, channels)
    return AstroUNet(
        channels=channels,
        width=args.width,
        encoder_blocks=encoder_blocks_from_args(args),
        bottleneck_blocks=args.bottleneck_blocks,
        dropout=args.dropout,
        drop_path_rate=float(getattr(args, "drop_path_rate", 0.05)),
    )


def build_model_from_checkpoint(ckpt: Dict[str, object], device: "torch.device") -> nn.Module:
    channels = int(ckpt["channels"])
    arch = str(ckpt.get("arch", "dncnn"))
    if arch == LITE_ARCH:
        model = build_lite_model_from_checkpoint(ckpt, device)
    elif arch == MODEL_ARCH:
        stored_blocks = ckpt.get("encoder_blocks")
        if stored_blocks:
            encoder_blocks = tuple(int(b) for b in stored_blocks)  # type: ignore[arg-type]
        else:
            enc_blocks = int(ckpt.get("enc_blocks", 2))
            num_levels = int(ckpt.get("num_levels", 4))
            encoder_blocks = tuple([enc_blocks] * (num_levels - 1) + [enc_blocks + 2])
        model = AstroUNet(
            channels=channels,
            width=int(ckpt.get("width", 48)),
            encoder_blocks=encoder_blocks,
            bottleneck_blocks=int(ckpt.get("bottleneck_blocks", 10)),
            dropout=float(ckpt.get("dropout", 0.0)),
            drop_path_rate=float(ckpt.get("drop_path_rate", 0.0)),
        ).to(device)
    elif arch in ("astro_unet_v2", "astro_unet_v3", "dncnn"):
        raise RuntimeError(
            f"Checkpoint uses legacy architecture '{arch}'. Retrain with the current script to use {MODEL_ARCH} "
            "(deeper 4-level UNet, stochastic depth, float32 norm)."
        )
    else:
        raise RuntimeError(f"Unsupported checkpoint architecture: {arch}")
    return model


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: "torch.device",
    ssim_weight: float,
    fft_weight: float,
    gradient_weight: float,
    amp: bool,
    tv_weight: float = 0.0,
    faint_signal_weight: float = 0.0,
) -> Dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "mae": 0.0,
        "rmse": 0.0,
        "noisy_psnr": 0.0,
    }
    total_items = 0
    use_amp = bool(amp and device.type == "cuda")
    with torch.no_grad():
        for noisy, clean in loader:
            noisy = noisy.to(device=device, dtype=torch.float32)
            clean = clean.to(device=device, dtype=torch.float32)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred = model(noisy)
            pred_for_loss = pred.float()
            clean_for_loss = clean.float()
            loss = combined_loss(
                pred_for_loss,
                clean_for_loss,
                ssim_weight,
                fft_weight,
                gradient_weight,
                tv_weight,
                faint_signal_weight,
            )
            if not torch.isfinite(loss):
                continue
            batch = noisy.shape[0]
            totals["loss"] += float(loss.item()) * batch
            totals["psnr"] += compute_psnr(pred_for_loss.detach(), clean_for_loss) * batch
            totals["ssim"] += compute_ssim(pred_for_loss.detach(), clean_for_loss) * batch
            totals["mae"] += compute_mae(pred_for_loss.detach(), clean_for_loss) * batch
            totals["rmse"] += compute_rmse(pred_for_loss.detach(), clean_for_loss) * batch
            totals["noisy_psnr"] += compute_psnr(noisy, clean_for_loss) * batch
            total_items += batch
    denom = max(total_items, 1)
    metrics = {key: value / denom for key, value in totals.items()}
    metrics["psnr_gain"] = metrics["psnr"] - metrics["noisy_psnr"]
    return metrics


def train(args: argparse.Namespace) -> None:
    require_torch()
    seed_everything(args.seed)

    base_dataset = ImagePairsDataset(args.data_dir, augment=False, synth_mix_prob=0.0, seed=args.seed)
    channels = base_dataset.channels
    total_items = len(base_dataset)
    val_items = max(1, int(round(total_items * args.val_split))) if total_items > 1 else 0
    if val_items >= total_items:
        val_items = total_items - 1
    train_items = total_items - val_items

    if val_items > 0:
        generator = torch.Generator().manual_seed(args.seed)
        train_subset, val_subset = random_split(base_dataset, [train_items, val_items], generator=generator)
        train_dataset = ImagePairsDataset(args.data_dir, augment=True, synth_mix_prob=args.synth_mix_prob, seed=args.seed)
        val_dataset = ImagePairsDataset(args.data_dir, augment=False, synth_mix_prob=0.0, seed=args.seed + 1)
        train_dataset.pairs = [base_dataset.pairs[i] for i in train_subset.indices]
        val_dataset.pairs = [base_dataset.pairs[i] for i in val_subset.indices]
    else:
        train_dataset = ImagePairsDataset(args.data_dir, augment=True, synth_mix_prob=args.synth_mix_prob, seed=args.seed)
        val_dataset = None

    device = resolve_device(args.device)
    num_workers = args.num_workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(num_workers > 0),
        )

    resume_ckpt: Dict[str, object] | None = None
    if args.resume:
        if not args.model_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.model_path}")
        resume_ckpt = torch.load(args.model_path, map_location=device)
        ckpt_channels = int(resume_ckpt["channels"])
        if ckpt_channels != channels:
            raise RuntimeError(
                f"Checkpoint channels ({ckpt_channels}) do not match dataset channels ({channels})."
            )
        model = build_model_from_checkpoint(resume_ckpt, device)
        model.load_state_dict(resume_ckpt["model_state"])
        args.arch = "lite" if str(resume_ckpt.get("arch", MODEL_ARCH)) == LITE_ARCH else "full"
        args.width = int(resume_ckpt.get("width", args.width))
        args.enc_blocks = int(resume_ckpt.get("enc_blocks", args.enc_blocks))
        args.num_levels = int(resume_ckpt.get("num_levels", getattr(args, "num_levels", 4)))
        args.bottleneck_blocks = int(resume_ckpt.get("bottleneck_blocks", args.bottleneck_blocks))
        args.dropout = float(resume_ckpt.get("dropout", args.dropout))
        args.drop_path_rate = float(resume_ckpt.get("drop_path_rate", getattr(args, "drop_path_rate", 0.05)))
    else:
        model = build_model(args, channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = max(0, getattr(args, "warmup_epochs", 0))
    cosine_epochs = max(1, args.epochs - warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_epochs, eta_min=args.lr * 0.05
    )
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
        )
    else:
        scheduler = cosine_scheduler
    scaler_enabled = args.amp and device.type == "cuda"
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    best_val = float("inf")
    best_psnr = float("-inf")
    best_epoch = 0
    resume_start_epoch = 0

    if resume_ckpt is not None and not args.reset_optimizer:
        if "optimizer_state" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        if "scheduler_state" in resume_ckpt:
            try:
                scheduler.load_state_dict(resume_ckpt["scheduler_state"])
            except (KeyError, ValueError):
                print("WARNING: scheduler state incompatible with current config (e.g. warmup added). Starting fresh scheduler.")

        if "scaler_state" in resume_ckpt and args.amp and device.type == "cuda":
            scaler.load_state_dict(resume_ckpt["scaler_state"])
        resume_start_epoch = int(resume_ckpt.get("epoch", 0))
        best_val = float(resume_ckpt.get("best_val_loss", best_val))
        best_psnr = float(resume_ckpt.get("best_val_psnr", best_psnr))
        best_epoch = int(resume_ckpt.get("best_epoch", best_epoch))
    cosine_scheduler.T_max = max(resume_start_epoch + cosine_epochs, 1)

    print(f"Training on device: {device}")
    print(f"Pairs: {len(base_dataset)} | Train: {len(train_dataset)} | Val: {len(val_dataset) if val_dataset is not None else 0}")
    print(f"Channels: {channels} | Architecture: {MODEL_ARCH}")
    if resume_ckpt is not None:
        restored_parts: List[str] = ["model"]
        if args.reset_optimizer:
            restored_parts.append("fresh_optimizer")
            restored_parts.append("fresh_scheduler")
            if args.amp and device.type == "cuda":
                restored_parts.append("fresh_scaler")
        else:
            if "optimizer_state" in resume_ckpt:
                restored_parts.append("optimizer")
            if "scheduler_state" in resume_ckpt:
                restored_parts.append("scheduler")
            if "scaler_state" in resume_ckpt and args.amp and device.type == "cuda":
                restored_parts.append("scaler")
        print(
            f"Resuming from: {args.model_path} | "
            f"restored={', '.join(restored_parts)} | "
            f"completed_epochs={resume_start_epoch}"
        )

    for epoch in range(resume_start_epoch + 1, resume_start_epoch + args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        seen_items = 0
        skipped_batches = 0
        train_iter = train_loader
        if tqdm is not None:
            train_iter = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch:03d}", unit="batch")

        for noisy, clean in train_iter:
            noisy = noisy.to(device=device, dtype=torch.float32, non_blocking=(device.type == "cuda"))
            clean = clean.to(device=device, dtype=torch.float32, non_blocking=(device.type == "cuda"))

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(args.amp and device.type == "cuda")):
                pred = model(noisy)
            pred_for_loss = pred.float()
            clean_for_loss = clean.float()
            loss = combined_loss(
                pred_for_loss,
                clean_for_loss,
                args.ssim_weight,
                args.fft_weight,
                args.gradient_weight,
                args.tv_weight,
                getattr(args, "faint_signal_weight", 0.0),
            )

            if not torch.isfinite(loss):
                skipped_batches += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.item()) * noisy.shape[0]
            seen_items += noisy.shape[0]

        scheduler.step()
        mean_train_loss = epoch_loss / max(seen_items, 1)

        if val_loader is not None:
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                args.ssim_weight,
                args.fft_weight,
                args.gradient_weight,
                args.amp,
                args.tv_weight,
                getattr(args, "faint_signal_weight", 0.0),
            )
            val_loss = val_metrics["loss"]
            val_psnr = val_metrics["psnr"]
        else:
            val_metrics = {
                "loss": mean_train_loss,
                "psnr": 0.0,
                "ssim": 0.0,
                "mae": 0.0,
                "rmse": 0.0,
                "noisy_psnr": 0.0,
                "psnr_gain": 0.0,
            }
            val_loss = mean_train_loss
            val_psnr = 0.0

        is_best = val_psnr > best_psnr
        if is_best:
            best_val = val_loss
            best_psnr = val_psnr
            best_epoch = epoch

        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "channels": channels,
            "arch": LITE_ARCH if str(getattr(args, "arch", "full")) == "lite" else MODEL_ARCH,
            "width": args.width,
            "enc_blocks": args.enc_blocks,
            "num_levels": int(getattr(args, "num_levels", 4)),
            "encoder_blocks": list(encoder_blocks_from_args(args)),
            "bottleneck_blocks": args.bottleneck_blocks,
            "dropout": args.dropout,
            "drop_path_rate": float(getattr(args, "drop_path_rate", 0.05)),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "best_val_psnr": best_psnr,
            "epochs": args.epochs,
            "seed": args.seed,
        }

        args.model_path.parent.mkdir(parents=True, exist_ok=True)
        if is_best:
            torch.save(checkpoint, args.model_path)
            print(f"Saved best model to: {args.model_path}")

        print(
            f"Epoch {epoch:03d}/{resume_start_epoch + args.epochs} | "
            f"train_loss={mean_train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"val_psnr={val_psnr:.2f} dB | val_ssim={val_metrics['ssim']:.4f} | "
            f"val_mae={val_metrics['mae']:.6f} | val_rmse={val_metrics['rmse']:.6f} | "
            f"psnr_gain={val_metrics['psnr_gain']:+.2f} dB | lr={scheduler.get_last_lr()[0]:.2e} | "
            f"skipped={skipped_batches}"
        )

    print(f"Best epoch: {best_epoch} | Best validation loss: {best_val:.6f} | Best validation PSNR: {best_psnr:.2f} dB")


def load_model(model_path: Path, device: "torch.device") -> nn.Module:
    require_torch()
    ckpt: Dict[str, object] = torch.load(model_path, map_location=device)
    model = build_model_from_checkpoint(ckpt, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_model_cached(model_path: Path, device: "torch.device") -> nn.Module:
    cache_key = (str(model_path.resolve()), str(device))
    mtime = model_path.stat().st_mtime
    cached = _PYTORCH_MODEL_CACHE.get(cache_key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    model = load_model(model_path, device)
    _PYTORCH_MODEL_CACHE[cache_key] = (mtime, model)
    return model


def export_model_to_onnx(
    model_path: Path,
    output_path: Path,
    patch_size: int = 256,
    opset_version: int = 17,
) -> None:
    require_torch()
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise ImportError("ONNX export requires the 'onnx' package. Install it with: pip install onnx") from exc

    device = torch.device("cpu")
    ckpt: Dict[str, object] = torch.load(model_path, map_location=device)
    channels = int(ckpt["channels"])
    model = build_model_from_checkpoint(ckpt, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy = torch.randn(1, channels, patch_size, patch_size, device=device, dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )


def _register_cuda_dll_directories() -> None:
    """Prepend CUDA 12 bin to PATH so onnxruntime can find cublas/cuDNN DLLs on Windows."""
    import os
    import sys

    if sys.platform != "win32":
        return

    cuda_root = os.environ.get("CUDA_PATH", "")
    candidates = []
    if cuda_root:
        candidates.append(os.path.join(cuda_root, "bin"))

    # Prefer the highest CUDA 12.x install (matches onnxruntime-gpu build requirement)
    toolkit = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.isdir(toolkit):
        v12_dirs = sorted(
            (v for v in os.listdir(toolkit) if v.startswith("v12")),
            reverse=True,
        )
        for ver in v12_dirs:
            candidates.insert(0, os.path.join(toolkit, ver, "bin"))

    to_prepend = [p for p in candidates if os.path.isdir(p)]
    if not to_prepend:
        return

    current_path = os.environ.get("PATH", "")
    prefix = os.pathsep.join(to_prepend)
    os.environ["PATH"] = prefix + os.pathsep + current_path

    if hasattr(os, "add_dll_directory"):
        for p in to_prepend:
            try:
                os.add_dll_directory(p)
            except OSError:
                pass


def _resolve_onnx_providers(device_name: str) -> List[str]:
    _register_cuda_dll_directories()
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "ONNX inference requires 'onnxruntime'. Install it with: pip install onnxruntime "
            "(or use onnxruntime-gpu for CUDA execution)."
        ) from exc

    if hasattr(ort, "get_available_providers"):
        available = set(ort.get_available_providers())
    else:
        # onnxruntime < 1.8 — fall back to checking all providers
        available = set(ort.get_all_providers()) if hasattr(ort, "get_all_providers") else set()
    requested = (device_name or "").strip().lower()

    if requested.startswith("cuda"):
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDA was requested for ONNX inference, but ONNX Runtime CUDA provider is not available. "
                "Install onnxruntime-gpu or use --device cpu."
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    if requested and requested != "cpu":
        raise ValueError("ONNX inference currently supports only 'cpu' or 'cuda' in --device.")

    if not requested and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def load_onnx_session(model_path: Path, device_name: str = ""):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "ONNX inference requires 'onnxruntime'. Install it with: pip install onnxruntime "
            "(or use onnxruntime-gpu for CUDA execution)."
        ) from exc
    providers = _resolve_onnx_providers(device_name)
    return ort.InferenceSession(str(model_path), providers=providers)


def load_onnx_session_cached(model_path: Path, device_name: str = ""):
    cache_key = (str(model_path.resolve()), (device_name or "").strip().lower())
    mtime = model_path.stat().st_mtime
    cached = _ONNX_SESSION_CACHE.get(cache_key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    session = load_onnx_session(model_path, device_name=device_name)
    _ONNX_SESSION_CACHE[cache_key] = (mtime, session)
    return session


def run_onnx_tta_denoise(
    session,
    image_chw: np.ndarray,
    patch_size: int,
    stride: int,
    tta: int,
    batch_size: int = 32,
    progress_callback: ProgressCallback | None = None,
) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    def onnx_predict(patches: np.ndarray) -> np.ndarray:
        outputs = session.run([output_name], {input_name: patches.astype(np.float32, copy=False)})
        return np.asarray(outputs[0], dtype=np.float32)

    modes = list(range(max(1, min(tta, 8))))
    outputs: List[np.ndarray] = []
    total_modes = len(modes)
    for mode_index, mode in enumerate(modes, start=1):
        transformed = apply_test_time_transform(image_chw, mode)
        start = (mode_index - 1) / total_modes
        end = mode_index / total_modes
        if progress_callback is not None:
            progress_callback(start, f"Running TTA pass {mode_index}/{total_modes}")
        pred = denoise_with_patch_predictor(
            predict_batch=onnx_predict,
            image_chw=transformed,
            patch_size=patch_size,
            stride=stride,
            batch_size=batch_size,
            progress_callback=progress_callback,
            progress_range=(start, end),
            progress_label=f"TTA {mode_index}/{total_modes}",
        )
        outputs.append(invert_test_time_transform(pred, mode))
    return np.mean(outputs, axis=0).astype(np.float32)


def _unsharp_mask_chw(image: np.ndarray, amount: float, radius: float = 1.5) -> np.ndarray:
    """Unsharp mask via per-channel Gaussian blur in PIL 'F' mode."""
    if amount <= 0.0:
        return image
    out = np.empty_like(image)
    for c in range(image.shape[0]):
        pil_ch = Image.fromarray(image[c], mode="F")
        from PIL import ImageFilter as _IF
        blurred = np.asarray(pil_ch.filter(_IF.GaussianBlur(radius=radius)))
        out[c] = np.clip(image[c] + amount * (image[c] - blurred), 0.0, 1.0)
    return out


def _noise_floor_subtract_chw(image: np.ndarray, floor: float) -> np.ndarray:
    """Subtract a bias floor (like bias-frame subtraction in astro)."""
    if floor <= 0.0:
        return image
    return np.maximum(image - floor, 0.0).astype(np.float32)


def _highlight_protection_chw(original: np.ndarray, denoised: np.ndarray, threshold: float) -> np.ndarray:
    """Blend original back in bright regions to prevent ringing/clipping around stars."""
    if threshold >= 1.0:
        return denoised
    lum = original.mean(axis=0, keepdims=True)
    mask = np.clip((lum - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0).astype(np.float32)
    mask = mask ** 2
    return (denoised * (1.0 - mask) + original * mask).astype(np.float32)


def denoise_image_file(
    model_path: Path,
    input_path: Path,
    patch_size: int = 128,
    stride: int = 64,
    tta: int = 4,
    amp: bool = False,
    batch_size: int = 32,
    strength: float = 1.0,
    detail_preservation: float = 0.2,
    background_threshold: float = 0.12,
    background_strength: float | None = None,
    subject_detail_preservation: float | None = None,
    background_detail_preservation: float = 0.05,
    faint_structure_protection: float = 0.85,
    faint_signal_preservation: float = 0.50,
    faint_signal_boost: float = 0.06,
    sharpen: float = 0.0,
    noise_floor: float = 0.0,
    highlight_protection: float = 1.0,
    device_name: str = "",
    progress_callback: ProgressCallback | None = None,
) -> Tuple[np.ndarray, np.ndarray, ImageMetadata]:
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
        require_torch()
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

    if progress_callback is not None:
        progress_callback(0.18, "Starting denoising")
    backend_progress = (
        None
        if progress_callback is None
        else lambda fraction, message: progress_callback(0.18 + 0.74 * fraction, message)
    )
    if use_onnx:
        denoised = run_onnx_tta_denoise(
            session=session,
            image_chw=model_input,
            patch_size=patch_size,
            stride=stride,
            tta=tta,
            batch_size=batch_size,
            progress_callback=backend_progress,
        )
    else:
        denoised = run_tta_denoise(
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
    denoised = adapt_channels_for_output(denoised, original_channels)
    if progress_callback is not None:
        progress_callback(0.94, "Applying finishing adjustments")
    bg_strength = strength if background_strength is None else background_strength
    subj_detail = detail_preservation if subject_detail_preservation is None else subject_detail_preservation
    structure_mask = compute_faint_structure_mask(
        img_chw,
        background_threshold=background_threshold,
        protection=faint_structure_protection,
    )
    background_mask = compute_background_mask(img_chw, background_threshold)
    background_mask = np.clip(background_mask * (1.0 - structure_mask), 0.0, 1.0).astype(np.float32)
    denoised = apply_spatial_denoise_strength(
        img_chw,
        denoised,
        background_mask,
        subject_strength=strength,
        background_strength=bg_strength,
    )
    denoised = preserve_detail_residual_spatial(
        img_chw,
        denoised,
        background_mask,
        subject_detail_preservation=subj_detail,
        background_detail_preservation=background_detail_preservation,
    )
    denoised = preserve_faint_signal(
        img_chw,
        denoised,
        structure_mask,
        preservation=faint_signal_preservation,
        boost=faint_signal_boost,
    )
    denoised = _highlight_protection_chw(img_chw, denoised, highlight_protection)
    denoised = _noise_floor_subtract_chw(denoised, noise_floor)
    denoised = _unsharp_mask_chw(denoised, sharpen)
    if progress_callback is not None:
        progress_callback(1.0, "Denoising complete")
    return img_hwc, denoised, meta


