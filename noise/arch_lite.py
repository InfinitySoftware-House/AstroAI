"""
AstroNAFLite — Lightweight astronomical image denoiser.

Design goals vs AstroUNet v2:
  ~4× fewer parameters  (~2.5M vs ~10M)
  ~10 MB model file     (~40 MB original)
  ~2-3× faster inference on CPU; ~2× on GPU
  Same patch-based / TTA / ONNX pipeline compatibility

Key architectural ideas
───────────────────────
1. LiteNAFBlock  — single-path (spatial only, no FFN sub-block)
   Original NAFBlock has two sequential sub-blocks (spatial + FFN).
   Each sub-block has its own LayerNorm + residual.  LiteNAFBlock
   keeps only the spatial path, halving parameter count per block
   while preserving the gating (SimpleGate) and depthwise conv that
   drive quality in NAFNet.

2. Dilated depthwise convolution  — bottleneck blocks alternate
   dilation=1 and dilation=3, giving a 9× larger receptive field
   at the bottleneck for free (zero extra parameters).

3. Bottleneck SE attention  — squeeze-excitation is applied only
   inside the bottleneck where channels are widest and global
   context matters most.  Excluded from shallow stages to save ops.

4. Reduced width + channel cap  — width=32, 3 encoder stages
   32→64→128→256.  Original is 48→96→192→384.
   Since most parameters live in the bottleneck (channel² scaling),
   halving the bottleneck width saves ~(384/256)² ≈ 2.25× there.

5. Fewer blocks per stage  — enc_blocks (2,2,3), bottleneck 4
   (vs 3,3,4 + 6 in the original).

6. Fully compatible checkpoint format  — same keys as AstroUNet v2
   checkpoints (model_state, channels, width, enc_blocks,
   bottleneck_blocks, dropout, arch).  The arch key is
   "astro_unet_lite" so load_model / build_model_from_checkpoint
   can be extended via patch_core_registry().
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Sequence, Tuple

# Heavy imports are guarded so the module can be imported without PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

LITE_ARCH = "astro_unet_lite"


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

if _TORCH_AVAILABLE:
    class LayerNorm2d(nn.Module):
        """Channel-wise layer norm for NCHW tensors (same as core.py)."""

        def __init__(self, channels: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            mean = x.mean(dim=1, keepdim=True)
            var = (x - mean).pow(2).mean(dim=1, keepdim=True)
            return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


    class SimpleGate(nn.Module):
        """Split channels in half, multiply element-wise.  Zero parameters."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x.chunk(2, dim=1)
            return x1 * x2


    class LiteNAFBlock(nn.Module):
        """
        Single-path variant of NAFBlock.

        Original NAFBlock flow:
            [LN → PW → DW → Gate → SCA → PW → β·skip]   (spatial sub-block)
            [LN → PW → Gate → PW → γ·skip]               (FFN sub-block)

        LiteNAFBlock drops the FFN sub-block entirely and makes the
        spatial path slightly richer to compensate:
            LN → PW(c→2c) → DW(2c, 3×3, dil) → Gate(→c) → [SE] → PW(c→c) → β·skip

        Parameters
        ----------
        channels : int
            Number of input/output channels.
        dw_expand : int
            Channel multiplier before the depthwise conv.  After SimpleGate,
            channels return to ``channels``.  Default 2 (same as original).
        use_attn : bool
            If True, apply squeeze-excitation after the gate.  Use for
            bottleneck blocks where global context matters.
        dilation : int
            Dilation factor for the depthwise conv.  Use dilation=3 in
            deep blocks to enlarge the receptive field without extra params.
        dropout : float
            Spatial dropout applied before the residual addition.
        """

        def __init__(
            self,
            channels: int,
            dw_expand: int = 2,
            use_attn: bool = False,
            dilation: int = 1,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            dw_ch = channels * dw_expand  # width before SimpleGate

            self.norm = LayerNorm2d(channels)

            # Pointwise expansion
            self.pw1 = nn.Conv2d(channels, dw_ch, kernel_size=1)

            # Depthwise spatial mixing with optional dilation
            padding = dilation  # keeps spatial size identical for 3×3
            self.dw = nn.Conv2d(
                dw_ch, dw_ch,
                kernel_size=3, padding=padding,
                groups=dw_ch, dilation=dilation,
            )

            # Free gating: halves channels back to `channels`
            self.gate = SimpleGate()

            # Optional squeeze-excitation (bottleneck only)
            se_mid = max(channels // 8, 4)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, se_mid, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(se_mid, channels, kernel_size=1),
                nn.Sigmoid(),
            ) if use_attn else None

            # Pointwise projection back to input width
            self.pw2 = nn.Conv2d(channels, channels, kernel_size=1)

            self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

            # Learnable residual scale, initialised small so early training
            # is stable (same initialisation as original NAFBlock)
            self.beta = nn.Parameter(torch.full((1, channels, 1, 1), 1e-2))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.norm(x)
            y = self.pw1(y)
            y = self.dw(y)
            y = self.gate(y)          # 2c → c
            if self.se is not None:
                y = y * self.se(y)
            y = self.pw2(y)
            y = self.drop(y)
            return x + y * self.beta


    # ─────────────────────────────────────────────────────────────────────────
    # Down / Up  (identical to core.py for clean parameter comparison)
    # ─────────────────────────────────────────────────────────────────────────

    class Downsample(nn.Module):
        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.body(x)


    class Upsample(nn.Module):
        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            # PixelShuffle(2) needs 4× the output channels as input
            self.body = nn.Sequential(
                nn.Conv2d(in_ch, out_ch * 4, kernel_size=1),
                nn.PixelShuffle(2),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.body(x)


    # ─────────────────────────────────────────────────────────────────────────
    # Main model
    # ─────────────────────────────────────────────────────────────────────────

    class AstroUNetLite(nn.Module):
        """
        Lightweight U-Net for astronomical image denoising / restoration.

        Architecture overview (default width=32, enc_blocks=(2,2,3), bottleneck=4)
        ──────────────────────────────────────────────────────────────────────
        intro   : Conv2d(channels → 32)
        enc-1   : 2 × LiteNAFBlock(32)           → skip-1
        down-1  : Downsample(32  → 64)
        enc-2   : 2 × LiteNAFBlock(64)           → skip-2
        down-2  : Downsample(64  → 128)
        enc-3   : 3 × LiteNAFBlock(128)          → skip-3
        down-3  : Downsample(128 → 256)
        middle  : 2 × LiteNAFBlock(256, SE, dil=1)
                + 2 × LiteNAFBlock(256, SE, dil=3)   ← dilated for large RF
        up-3    : Upsample(256 → 128) + skip-3 + 3 × LiteNAFBlock(128)
        up-2    : Upsample(128 → 64)  + skip-2 + 2 × LiteNAFBlock(64)
        up-1    : Upsample(64  → 32)  + skip-1 + 2 × LiteNAFBlock(32)
        ending  : Conv2d(32 → channels)
        output  : inp − residual_noise   (residual learning)
        ──────────────────────────────────────────────────────────────────────

        Bottleneck detail
        ─────────────────
        Half the bottleneck blocks use dilation=3 in the depthwise conv.
        This triples the effective kernel span (from 3×3 to 7×7 effective
        field) without adding a single parameter, improving the model's
        ability to suppress large-scale gradients and glow artefacts.

        Compatibility
        ─────────────
        Checkpoint format is identical to AstroUNet v2 except arch="astro_unet_lite".
        The same patch-based inference, TTA, ONNX export, and denoise_image_file()
        pipeline works once the arch registry is extended via patch_core_registry().
        """

        def __init__(
            self,
            channels: int,
            width: int = 32,
            encoder_blocks: Sequence[int] = (2, 2, 3),
            bottleneck_blocks: int = 4,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()

            self.intro = nn.Conv2d(channels, width, kernel_size=3, padding=1)
            self.ending = nn.Conv2d(width, channels, kernel_size=3, padding=1)

            self.encoders: nn.ModuleList = nn.ModuleList()
            self.downs: nn.ModuleList = nn.ModuleList()
            self.decoders: nn.ModuleList = nn.ModuleList()
            self.ups: nn.ModuleList = nn.ModuleList()

            chan = width
            skip_channels: List[int] = []

            for n_blocks in encoder_blocks:
                self.encoders.append(
                    nn.Sequential(*[
                        LiteNAFBlock(chan, dropout=dropout)
                        for _ in range(n_blocks)
                    ])
                )
                skip_channels.append(chan)
                self.downs.append(Downsample(chan, chan * 2))
                chan *= 2

            # Bottleneck: alternate dilation=1 and dilation=3 blocks.
            # All bottleneck blocks use SE attention (use_attn=True).
            bottleneck_blocks_list: List[nn.Module] = []
            for i in range(bottleneck_blocks):
                dilation = 3 if (i % 2 == 1) else 1
                bottleneck_blocks_list.append(
                    LiteNAFBlock(chan, use_attn=True, dilation=dilation, dropout=dropout)
                )
            self.middle = nn.Sequential(*bottleneck_blocks_list)

            for n_blocks, skip_chan in zip(reversed(encoder_blocks), reversed(skip_channels)):
                self.ups.append(Upsample(chan, skip_chan))
                chan = skip_chan
                self.decoders.append(
                    nn.Sequential(*[
                        LiteNAFBlock(chan, dropout=dropout)
                        for _ in range(n_blocks)
                    ])
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            inp = x
            x = self.intro(x)

            skips: List[torch.Tensor] = []
            for encoder, down in zip(self.encoders, self.downs):
                x = encoder(x)
                skips.append(x)
                x = down(x)

            x = self.middle(x)

            for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
                x = up(x)
                # Guard against off-by-one from odd spatial sizes
                if x.shape[-2:] != skip.shape[-2:]:
                    x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                x = x + skip
                x = decoder(x)

            residual_noise = self.ending(x)
            return inp - residual_noise

        def param_count(self) -> int:
            """Return total trainable parameter count."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

        def size_mb(self) -> float:
            """Estimated float32 weight size in MB."""
            return self.param_count() * 4 / (1024 ** 2)

else:
    class AstroUNetLite:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError("PyTorch is required for AstroUNetLite.")


# ─────────────────────────────────────────────────────────────────────────────
# Factory helpers  (mirror of core.py's build_model / build_model_from_checkpoint)
# ─────────────────────────────────────────────────────────────────────────────

def build_lite_model(args: argparse.Namespace, channels: int) -> "nn.Module":
    """Construct an AstroUNetLite from parsed CLI args."""
    return AstroUNetLite(
        channels=channels,
        width=args.width,
        encoder_blocks=(args.enc_blocks, args.enc_blocks, args.enc_blocks + 1),
        bottleneck_blocks=args.bottleneck_blocks,
        dropout=args.dropout,
    )


def build_lite_model_from_checkpoint(
    ckpt: Dict[str, object],
    device: "torch.device",
) -> "nn.Module":
    """Reconstruct AstroUNetLite from a saved checkpoint dict."""
    channels = int(ckpt["channels"])
    enc_blocks = int(ckpt.get("enc_blocks", 2))
    model = AstroUNetLite(
        channels=channels,
        width=int(ckpt.get("width", 32)),
        encoder_blocks=(enc_blocks, enc_blocks, enc_blocks + 1),
        bottleneck_blocks=int(ckpt.get("bottleneck_blocks", 4)),
        dropout=float(ckpt.get("dropout", 0.0)),
    ).to(device)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Core registry patcher
# ─────────────────────────────────────────────────────────────────────────────

def patch_core_registry() -> None:
    """
    Extend noise.core so it recognises 'astro_unet_lite' checkpoints.

    Call this once before using load_model / denoise_image_file with a
    lite checkpoint.  Safe to call multiple times (idempotent).

    How it works
    ────────────
    noise.core.build_model_from_checkpoint is monkey-patched with a wrapper
    that intercepts the 'astro_unet_lite' arch key and delegates to
    build_lite_model_from_checkpoint, forwarding everything else to the
    original implementation unchanged.
    """
    import noise.core as _core

    if getattr(_core, "_lite_arch_patched", False):
        return  # already patched

    _orig_build = _core.build_model_from_checkpoint

    def _patched_build(ckpt: Dict[str, object], device: "torch.device") -> "nn.Module":
        if str(ckpt.get("arch", "")) == LITE_ARCH:
            model = build_lite_model_from_checkpoint(ckpt, device)
            model.load_state_dict(ckpt["model_state"])  # type: ignore[arg-type]
            model.eval()
            return model
        return _orig_build(ckpt, device)

    _core.build_model_from_checkpoint = _patched_build
    _core._lite_arch_patched = True  # type: ignore[attr-defined]
