"""
Class-Conditional U-Net for DDPM
==================================
Architecture overview:
  • Sinusoidal time embedding → 2-layer MLP projection.
  • Class label embedding → added to time embedding (FiLM-style conditioning).
  • Encoder: 4 resolution levels, each with 2 ResBlocks + optional Attention.
  • Bottleneck: 2 ResBlocks + Self-Attention.
  • Decoder: mirrors encoder with skip connections (U-Net).
  • Output: linear conv → same spatial dims as input.

Resolution trace for 224×224 input:
  Enc L0: 224×224  ch=128   ResBlock ×2
  Enc L1: 112×112  ch=256   ResBlock ×2
  Enc L2:  56×56   ch=256   ResBlock ×2 + Attn
  Enc L3:  28×28   ch=512   ResBlock ×2 + Attn
  Bottleneck: 14×14  ch=512  ResBlock ×2 + Attn
  (mirrored decoder with up-sampling)

Design notes:
  • GroupNorm (groups=32) throughout — stable for small batches.
  • SiLU activation — smoother gradient than ReLU.
  • Attention uses multi-head scaled dot-product (built-in Flash-Attention
    pathway when running on CUDA with PyTorch ≥ 2.0).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ═══════════════════════════════════════════════════════════════════════════
#  HELPER LAYERS
# ═══════════════════════════════════════════════════════════════════════════

class Swish(nn.Module):
    """SiLU / Swish activation."""
    def forward(self, x: Tensor) -> Tensor:
        return x * x.sigmoid()


def norm(channels: int, groups: int = 32) -> nn.GroupNorm:
    """GroupNorm with automatic group-count reduction if channels < 32."""
    g = min(groups, channels)
    while channels % g != 0:
        g //= 2
    return nn.GroupNorm(g, channels)


# ─────────────────────────────────────────────────────────────────────────────
#  SINUSOIDAL TIME EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal positional encoding for diffusion timesteps (Vaswani et al.).
    Produces a (B, dim) embedding for integer timesteps.
    """

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "Embedding dim must be even."
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        """
        Parameters
        ----------
        t : (B,) integer tensor of timesteps in [0, T-1]
        """
        device      = t.device
        half_dim    = self.dim // 2
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=device) / (half_dim - 1)
        )                                              # (half_dim,)
        angles = t[:, None].float() * frequencies[None, :]  # (B, half_dim)
        emb    = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (B, dim)
        return emb


class TimeEmbedMLP(nn.Module):
    """
    Project raw sinusoidal time + class embeddings into a rich feature vector.
    """

    def __init__(self, base_dim: int, proj_dim: int, num_classes: int):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(base_dim)
        self.class_embed = nn.Embedding(num_classes + 1, base_dim,
                                         padding_idx=num_classes)  # +1 for CFG null token
        self.mlp = nn.Sequential(
            nn.Linear(base_dim, proj_dim),
            Swish(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, t: Tensor, class_label: Tensor) -> Tensor:
        """
        Parameters
        ----------
        t           : (B,)   integer timesteps
        class_label : (B,)   integer class indices [0, num_classes-1]
                             OR ``num_classes`` for the null (unconditioned) token.
        Returns
        -------
        emb : (B, proj_dim)
        """
        t_emb = self.time_embed(t)              # (B, base_dim)
        c_emb = self.class_embed(class_label)   # (B, base_dim)
        combined = t_emb + c_emb               # additive conditioning (FiLM-lite)
        return self.mlp(combined)              # (B, proj_dim)


# ─────────────────────────────────────────────────────────────────────────────
#  RESNET BLOCK
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """
    ResNet block with time/class embedding injection (added to hidden features).

    Structure:
        x → Norm → Swish → Conv3×3
          → [+time_emb projection] → Norm → Swish → Dropout → Conv3×3
          → [+shortcut if channels differ]
    """

    def __init__(
        self,
        in_ch:   int,
        out_ch:  int,
        emb_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block1 = nn.Sequential(
            norm(in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        # Time + class embedding projection (scale & shift)
        self.emb_proj = nn.Sequential(
            Swish(),
            nn.Linear(emb_dim, out_ch * 2),  # for scale + shift
        )
        self.block2 = nn.Sequential(
            norm(out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        h = self.block1(x)

        # Scale-shift modulation from time + class embedding
        scale_shift = self.emb_proj(emb)[:, :, None, None]  # (B, 2C, 1, 1)
        scale, shift = scale_shift.chunk(2, dim=1)
        h = h * (1.0 + scale) + shift

        h = self.block2(h)
        return h + self.shortcut(x)


# ─────────────────────────────────────────────────────────────────────────────
#  SELF-ATTENTION BLOCK
# ─────────────────────────────────────────────────────────────────────────────

class SelfAttention(nn.Module):
    """
    Multi-head self-attention applied to spatial feature maps.
    Uses PyTorch 2.0 F.scaled_dot_product_attention for Flash-Attention on CUDA.
    """

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        assert channels % num_heads == 0, \
            f"channels {channels} must be divisible by num_heads {num_heads}"
        self.num_heads   = num_heads
        self.head_dim    = channels // num_heads
        self.norm        = norm(channels)
        self.qkv         = nn.Conv1d(channels, channels * 3, 1, bias=False)
        self.out_proj    = nn.Conv1d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        res = x
        x   = self.norm(x).reshape(B, C, H * W)   # (B, C, HW)

        qkv = self.qkv(x)                           # (B, 3C, HW)
        q, k, v = qkv.chunk(3, dim=1)               # each (B, C, HW)

        # Rearrange to (B, heads, HW, head_dim)
        def split_heads(t: Tensor) -> Tensor:
            return t.reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Flash attention (PyTorch ≥ 2.0) or manual fallback
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

        # (B, heads, HW, head_dim) → (B, C, HW)
        attn_out = attn_out.permute(0, 1, 3, 2).reshape(B, C, H * W)
        out      = self.out_proj(attn_out).reshape(B, C, H, W)
        return res + out


# ─────────────────────────────────────────────────────────────────────────────
#  DOWN / UP SAMPLING
# ─────────────────────────────────────────────────────────────────────────────

class Downsample(nn.Module):
    """Strided convolution 2× spatial down-sampling."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Bilinear 2× up-sampling followed by conv to avoid checkerboard artifacts."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
#  ENCODER LEVEL
# ─────────────────────────────────────────────────────────────────────────────

class EncoderLevel(nn.Module):
    def __init__(
        self,
        in_ch:        int,
        out_ch:       int,
        emb_dim:      int,
        num_res:      int  = 2,
        use_attn:     bool = False,
        attn_heads:   int  = 8,
        dropout:      float = 0.1,
        downsample:   bool = True,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        self.attns   = nn.ModuleList()

        for i in range(num_res):
            self.resnets.append(
                ResBlock(in_ch if i == 0 else out_ch, out_ch, emb_dim, dropout)
            )
            self.attns.append(
                SelfAttention(out_ch, attn_heads) if use_attn else nn.Identity()
            )

        self.down = Downsample(out_ch) if downsample else nn.Identity()

    def forward(self, x: Tensor, emb: Tensor) -> tuple[Tensor, list[Tensor]]:
        skips = []
        for res, attn in zip(self.resnets, self.attns):
            x = res(x, emb)
            x = attn(x)
            skips.append(x)
        x = self.down(x)
        return x, skips


# ─────────────────────────────────────────────────────────────────────────────
#  DECODER LEVEL
# ─────────────────────────────────────────────────────────────────────────────

class DecoderLevel(nn.Module):
    def __init__(
        self,
        in_ch:        int,
        skip_ch:      int,
        out_ch:       int,
        emb_dim:      int,
        num_res:      int  = 2,
        use_attn:     bool = False,
        attn_heads:   int  = 8,
        dropout:      float = 0.1,
        upsample:     bool = True,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        self.attns   = nn.ModuleList()

        for i in range(num_res):
            # first block fuses incoming features with skip connection
            res_in = (in_ch + skip_ch) if i == 0 else out_ch
            self.resnets.append(
                ResBlock(res_in, out_ch, emb_dim, dropout)
            )
            self.attns.append(
                SelfAttention(out_ch, attn_heads) if use_attn else nn.Identity()
            )

        self.up = Upsample(out_ch) if upsample else nn.Identity()

    def forward(
        self, x: Tensor, skips: list[Tensor], emb: Tensor
    ) -> Tensor:
        for i, (res, attn) in enumerate(zip(self.resnets, self.attns)):
            skip = skips.pop()          # LIFO — matches encoder order
            x    = torch.cat([x, skip], dim=1)
            x    = res(x, emb)
            x    = attn(x)
        x = self.up(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════
#  CLASS-CONDITIONAL U-NET
# ═══════════════════════════════════════════════════════════════════════════

class ClassConditionalUNet(nn.Module):
    """
    U-Net noise-prediction network for class-conditional DDPM.

    Architecture
    ------------
    Input : (B, 3, 224, 224)  noisy image x_t
    Output: (B, 3, 224, 224)  predicted noise ε_θ(x_t, t, c)

    Conditioning:
      Time  → sinusoidal embed → MLP
      Class → learned embed   → sums with time embed → scale-shift in ResBlocks

    Channel widths  : [128, 256, 256, 512]  (4 encoder levels)
    Bottleneck      : 512 (2 ResBlocks + Self-Attention)
    Attention at    : levels 2, 3 and bottleneck (56×56 and smaller)

    Parameters
    ----------
    in_channels  : int   input image channels (3 for RGB).
    num_classes  : int   number of conditioning classes.
    base_ch      : int   base channel count (multiplied per level).
    ch_mults     : tuple channel multipliers per level.
    num_res      : int   ResBlocks per encoder/decoder level.
    attn_levels  : set   levels (0-indexed) where self-attention is applied.
    dropout      : float dropout probability inside ResBlocks.
    """

    def __init__(
        self,
        in_channels: int   = 3,
        num_classes: int   = 7,
        base_ch:     int   = 128,
        ch_mults:    tuple = (1, 2, 2, 4),
        num_res:     int   = 2,
        attn_levels: set   = frozenset({2, 3}),
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        channels         = [base_ch * m for m in ch_mults]  # e.g. [128,256,256,512]
        emb_dim          = base_ch * 4                       # 512

        # ── Embedding ──────────────────────────────────────────────────
        self.time_class_emb = TimeEmbedMLP(
            base_dim=base_ch,
            proj_dim=emb_dim,
            num_classes=num_classes,
        )

        # ── Input projection ───────────────────────────────────────────
        self.input_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # ── Encoder ────────────────────────────────────────────────────
        self.encoders = nn.ModuleList()
        in_ch = channels[0]
        for i, ch in enumerate(channels):
            use_attn = (i in attn_levels)
            # Last level: no downsample (goes to bottleneck directly)
            is_last  = (i == len(channels) - 1)
            self.encoders.append(
                EncoderLevel(
                    in_ch=in_ch, out_ch=ch,
                    emb_dim=emb_dim,
                    num_res=num_res,
                    use_attn=use_attn,
                    dropout=dropout,
                    downsample=not is_last,
                )
            )
            in_ch = ch

        # ── Bottleneck ─────────────────────────────────────────────────
        bn_ch = channels[-1]
        self.mid_res1 = ResBlock(bn_ch, bn_ch, emb_dim, dropout)
        self.mid_attn = SelfAttention(bn_ch)
        self.mid_res2 = ResBlock(bn_ch, bn_ch, emb_dim, dropout)

        # ── Decoder ────────────────────────────────────────────────────
        self.decoders = nn.ModuleList()
        rev_channels = list(reversed(channels))
        for i, ch in enumerate(rev_channels):
            # Each level consumes skip connections from encoder
            # skip_ch: channels used by the *same* encoder level
            skip_ch  = ch
            out_ch   = rev_channels[i + 1] if i + 1 < len(rev_channels) else channels[0]
            use_attn = ((len(channels) - 1 - i) in attn_levels)
            is_last  = (i == len(rev_channels) - 1)
            self.decoders.append(
                DecoderLevel(
                    in_ch=ch,
                    skip_ch=skip_ch,
                    out_ch=out_ch,
                    emb_dim=emb_dim,
                    num_res=num_res,
                    use_attn=use_attn,
                    dropout=dropout,
                    upsample=not is_last,
                )
            )

        # ── Output head ────────────────────────────────────────────────
        self.out = nn.Sequential(
            norm(channels[0]),
            Swish(),
            nn.Conv2d(channels[0], in_channels, 3, padding=1),
        )

    # ─────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x:     Tensor,
        t:     Tensor,
        label: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        x     : (B, 3, H, W)  noisy image x_t
        t     : (B,)  integer timesteps in [0, T-1]
        label : (B,)  integer class labels in [0, num_classes-1]
                      (use num_classes for null/unconditioned token in CFG)

        Returns
        -------
        eps_pred : (B, 3, H, W)  predicted noise
        """
        # Time + class conditioning
        emb = self.time_class_emb(t, label)    # (B, emb_dim)

        # Input projection
        x = self.input_conv(x)                 # (B, C0, H, W)

        # Encoder
        all_skips: list[list[Tensor]] = []
        for enc in self.encoders:
            x, skips = enc(x, emb)
            all_skips.append(skips)

        # Bottleneck
        x = self.mid_res1(x, emb)
        x = self.mid_attn(x)
        x = self.mid_res2(x, emb)

        # Decoder (reverse order)
        for dec, skips in zip(self.decoders, reversed(all_skips)):
            x = dec(x, list(skips), emb)

        return self.out(x)

    # ─────────────────────────────────────────────────────────────────────

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        M = self.num_parameters / 1e6
        return (
            f"ClassConditionalUNet("
            f"classes={self.num_classes}, "
            f"params={M:.1f}M)"
        )
