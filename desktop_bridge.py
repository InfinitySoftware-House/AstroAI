#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from gradient.inference import run_gradient_inference
from noise.core import denoise_image_file, read_image, write_image
from noise.star_reducer import run_star_reducer_inference


def emit_progress(value: float, message: str, request_id: str | None = None) -> None:
    payload = {
        "progress": max(0.0, min(1.0, float(value))),
        "message": message,
    }
    if request_id is not None:
        payload["request_id"] = request_id
    print(f"__PROGRESS__{json.dumps(payload)}", file=sys.stderr, flush=True)


def chw_to_image(image: np.ndarray) -> Image.Image:
    image = np.clip(image, 0.0, 1.0)
    if image.ndim == 3 and image.shape[0] in {1, 3}:
        hwc = image[0] if image.shape[0] == 1 else np.transpose(image, (1, 2, 0))
    elif image.ndim == 3 and image.shape[2] in {1, 3}:
        hwc = image[:, :, 0] if image.shape[2] == 1 else image
    else:
        raise ValueError(f"Unsupported preview shape: {image.shape}")

    arr = (hwc * 255.0).round().astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(arr, mode="RGB")


def image_to_base64_png(image: Image.Image) -> str:
    preview = image.copy()
    preview.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    preview.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_default_output(input_path: Path, mode: str = "denoise") -> Path:
    suffix = input_path.suffix if input_path.suffix else ".tif"
    if mode == "gradient":
        suffix_tag = "_gradient_removed"
    elif mode == "star":
        suffix_tag = "_stars_reduced"
    else:
        suffix_tag = "_denoised"
    return input_path.with_name(f"{input_path.stem}{suffix_tag}{suffix}")


def find_default_model(mode: str = "denoise") -> Path | None:
    candidates = list_models(mode)
    if not candidates:
        return None
    return candidates[0]


def list_models(mode: str = "denoise") -> list[Path]:
    models_dir = Path("models")
    candidates = list(models_dir.glob("*.pt")) + list(models_dir.glob("*.pth")) + list(models_dir.glob("*.onnx"))
    if mode == "gradient":
        candidates = [path for path in candidates if "gradient" in path.stem.lower()]
    elif mode == "star":
        candidates = [path for path in candidates if "star" in path.stem.lower() or "seeing" in path.stem.lower()]
    elif mode == "sharpen":
        candidates = [path for path in candidates if "sharpen" in path.stem.lower() or "sharp" in path.stem.lower()]
    else:
        candidates = [
            path
            for path in candidates
            if "gradient" not in path.stem.lower()
            and "star" not in path.stem.lower()
            and "seeing" not in path.stem.lower()
            and "sharpen" not in path.stem.lower()
            and "sharp" not in path.stem.lower()
        ]
    suffix_rank = {
        ".pt": 0,
        ".pth": 1,
        ".onnx": 2,
    }
    return sorted(candidates, key=lambda path: (suffix_rank.get(path.suffix.lower(), 99), path.name.lower()))


def command_get_default_model(args: argparse.Namespace) -> dict[str, object]:
    model = find_default_model(args.mode)
    return {"model_path": str(model) if model else ""}


def command_list_models(args: argparse.Namespace) -> dict[str, object]:
    models = list_models(args.mode)
    return {
        "models": [
            {
                "name": model.name,
                "path": str(model),
            }
            for model in models
        ]
    }


def command_preview(args: argparse.Namespace) -> dict[str, object]:
    image_hwc, _ = read_image(args.input)
    preview = chw_to_image(np.transpose(image_hwc, (2, 0, 1)))
    return {"preview_base64": image_to_base64_png(preview)}


def command_denoise(
    args: argparse.Namespace,
    progress_callback: Callable[[float, str], None] = lambda value, message: emit_progress(value, message),
) -> dict[str, object]:
    output_path = args.output if args.output else build_default_output(args.input, args.mode)
    if args.mode == "gradient":
        original_hwc, denoised_chw, meta = run_gradient_inference(
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
            gradient_blur_sigma=getattr(args, "gradient_blur_sigma", 3.0),
            device_name=args.device,
            progress_callback=progress_callback,
        )
    elif args.mode == "star":
        original_hwc, denoised_chw, meta = run_star_reducer_inference(
            model_path=args.model_path,
            input_path=args.input,
            patch_size=args.patch_size,
            stride=args.stride,
            tta=args.tta,
            amp=args.amp,
            batch_size=args.batch_size,
            strength=args.strength,
            device_name=args.device,
            progress_callback=progress_callback,
        )
    elif args.mode == "sharpen":
        # Deconvolution model: the network IS the enhancement. Disable denoise-oriented
        # blends that would dilute the sharpened result (detail-preservation re-mixes the
        # soft original; faint-signal/background cleanup don't apply). Keep highlight
        # protection on to guard star cores from ringing.
        original_hwc, denoised_chw, meta = denoise_image_file(
            model_path=args.model_path,
            input_path=args.input,
            patch_size=args.patch_size,
            stride=args.stride,
            tta=args.tta,
            amp=args.amp,
            batch_size=args.batch_size,
            strength=args.strength,
            detail_preservation=0.0,
            background_threshold=0.0,
            background_strength=1.0,
            subject_detail_preservation=0.0,
            background_detail_preservation=0.0,
            faint_structure_protection=0.0,
            faint_signal_preservation=0.0,
            faint_signal_boost=0.0,
            sharpen=0.0,
            noise_floor=0.0,
            highlight_protection=0.9,
            device_name=args.device,
            progress_callback=progress_callback,
        )
    else:
        original_hwc, denoised_chw, meta = denoise_image_file(
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
            device_name=args.device,
            progress_callback=progress_callback,
        )
    progress_callback(0.97, "Writing output file")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_image(output_path, denoised_chw, meta)
    progress_callback(0.99, "Preparing previews")
    original_preview = chw_to_image(np.transpose(original_hwc, (2, 0, 1)))
    denoised_preview = chw_to_image(denoised_chw)
    progress_callback(1.0, "Done")
    return {
        "output_path": str(output_path),
        "original_preview_base64": image_to_base64_png(original_preview),
        "denoised_preview_base64": image_to_base64_png(denoised_preview),
    }


def run_denoise_worker() -> None:
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        request_id = ""
        try:
            request = json.loads(line)
            request_id = str(request.get("id", ""))
            args = argparse.Namespace(
                model_path=Path(request["model_path"]),
                input=Path(request["input"]),
                output=Path(request["output"]),
                mode=str(request.get("mode", "denoise")),
                patch_size=int(request.get("patch_size", 128)),
                stride=int(request.get("stride", 64)),
                tta=int(request.get("tta", 4)),
                batch_size=int(request.get("batch_size", 32)),
                amp=bool(request.get("amp", False)),
                strength=float(request.get("strength", 1.0)),
                detail_preservation=float(request.get("detail_preservation", 0.2)),
                background_threshold=float(request.get("background_threshold", 0.12)),
                background_strength=float(request.get("background_strength", 1.2)),
                subject_detail_preservation=float(request.get("subject_detail_preservation", 0.2)),
                background_detail_preservation=float(request.get("background_detail_preservation", 0.05)),
                gradient_blur_sigma=float(request.get("gradient_blur_sigma", 3.0)),
                device=str(request.get("device", "")),
            )
            result = command_denoise(
                args,
                progress_callback=lambda value, message, req_id=request_id: emit_progress(value, message, req_id),
            )
            response = {"id": request_id, "ok": True, "result": result}
        except Exception as exc:
            response = {"id": request_id, "ok": False, "error": str(exc)}

        print(json.dumps(response), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bridge between Electron and the Python denoiser.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_default = subparsers.add_parser("get-default-model")
    p_default.add_argument("--mode", choices=["denoise", "gradient", "star", "sharpen"], default="denoise")
    p_default.set_defaults(func=command_get_default_model)

    p_models = subparsers.add_parser("list-models")
    p_models.add_argument("--mode", choices=["denoise", "gradient", "star", "sharpen"], default="denoise")
    p_models.set_defaults(func=command_list_models)

    p_preview = subparsers.add_parser("preview")
    p_preview.add_argument("--input", type=Path, required=True)
    p_preview.set_defaults(func=command_preview)

    p_denoise = subparsers.add_parser("denoise")
    p_denoise.add_argument("--model-path", type=Path, required=True)
    p_denoise.add_argument("--input", type=Path, required=True)
    p_denoise.add_argument("--output", type=Path, required=True)
    p_denoise.add_argument("--mode", choices=["denoise", "gradient", "star", "sharpen"], default="denoise")
    p_denoise.add_argument("--patch-size", type=int, default=128)
    p_denoise.add_argument("--stride", type=int, default=64)
    p_denoise.add_argument("--tta", type=int, default=4)
    p_denoise.add_argument("--batch-size", type=int, default=32)
    p_denoise.add_argument("--amp", action="store_true")
    p_denoise.add_argument("--strength", type=float, default=1.0)
    p_denoise.add_argument("--detail-preservation", type=float, default=0.2)
    p_denoise.add_argument("--background-threshold", type=float, default=0.12)
    p_denoise.add_argument("--background-strength", type=float, default=1.2)
    p_denoise.add_argument("--subject-detail-preservation", type=float, default=0.2)
    p_denoise.add_argument("--background-detail-preservation", type=float, default=0.05)
    p_denoise.add_argument("--gradient-blur-sigma", type=float, default=3.0)
    p_denoise.add_argument("--device", type=str, default="")
    p_denoise.set_defaults(func=command_denoise)

    p_worker = subparsers.add_parser("serve-denoise")
    p_worker.set_defaults(func=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "serve-denoise":
        run_denoise_worker()
        return
    payload = args.func(args)
    json.dump(payload, sys.stdout)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
