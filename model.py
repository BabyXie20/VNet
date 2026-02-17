from __future__ import annotations
from typing import Literal, Optional
from typing import Sequence, Tuple
import torch
from torch import nn
import torch.nn.functional as F
import math

NormType = Optional[Literal["batchnorm", "groupnorm", "instancenorm", "none"]]


def predictive_entropy_from_logits(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    logits: [B, K, D, H, W]
    return: [B, 1, D, H, W] in [0, 1] (normalized entropy)
    """
    p = torch.softmax(logits, dim=1)
    ent = -(p * torch.log(p.clamp_min(eps))).sum(dim=1, keepdim=True)  # [B,1,D,H,W]
    ent = ent / math.log(p.shape[1])  # normalize by log(K)
    return ent


def _best_group_count(num_channels: int, preferred: int = 16) -> int:
    g = min(preferred, num_channels)
    while g > 1 and (num_channels % g) != 0:
        g -= 1
    return g


def norm3d(norm: NormType, num_channels: int, affine: bool = True) -> nn.Module:
    if norm == "batchnorm":
        return nn.BatchNorm3d(num_channels)
    if norm == "groupnorm":
        num_groups = _best_group_count(num_channels, preferred=16)
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    if norm == "instancenorm":
        return nn.InstanceNorm3d(num_channels, affine=affine)
    if norm in ("none", None):
        return nn.Identity()
    raise ValueError(f"Unknown normalization: {norm}")


def _haar_pair():
    s2 = math.sqrt(2.0)
    L = torch.tensor([1., 1.]) / s2
    H = torch.tensor([1., -1.]) / s2
    return L, H


def _kron3(a, b, c):
    return torch.einsum('i,j,k->ijk', a, b, c)


def _build_haar_3d_kernels(device):
    L, H = _haar_pair()
    taps = []
    for zf in [L, H]:
        for yf in [L, H]:
            for xf in [L, H]:
                taps.append(_kron3(zf, yf, xf))  # [2, 2, 2]
    k = torch.stack(taps, dim=0)[:, None, ...]  # [8, 1, 2, 2, 2]
    stride = (2, 2, 2)
    ksize = (2, 2, 2)
    return k.to(device), stride, ksize


# =========================
# (1) DWT/IDWT: cache weight in __init__
# =========================
class DWT3D(nn.Module):
    def __init__(self, channels: Optional[int] = None):
        """
        channels: if provided, pre-cache the grouped conv weight [8*C,1,2,2,2] in __init__
                  to avoid repeat(C,...) every forward. Logic identical.
        """
        super().__init__()
        k, stride, ksize = _build_haar_3d_kernels(device=torch.device('cpu'))
        # k: [8, 1, 2, 2, 2]
        self.register_buffer("kernels", k)
        self.stride = stride
        self.ksize = ksize

        self.channels = channels
        if channels is not None:
            weight = self.kernels.repeat(channels, 1, 1, 1, 1)  # [8*C,1,2,2,2]
            self.register_buffer("weight", weight)
        else:
            self.weight = None  # type: ignore[assignment]

    def forward(self, x: torch.Tensor):
        B, C, D, H, W = x.shape
        S = self.kernels.shape[0]  # 8

        if self.weight is None:
            # fallback (keeps original behavior if channels not provided)
            weight = self.kernels.repeat(C, 1, 1, 1, 1)
        else:
            if self.channels is not None and C != self.channels:
                raise ValueError(f"DWT3D channels mismatch: got C={C}, expected {self.channels}")
            weight = self.weight

        y = F.conv3d(x, weight, stride=self.stride, padding=0, groups=C)
        y = y.view(B, C, S, *y.shape[-3:])  # [B, C, 8, D/2, H/2, W/2]
        low = y[:, :, :1, ...].squeeze(2)   # [B, C, D/2, H/2, W/2]
        highs = y[:, :, 1:, ...]            # [B, C, 7, D/2, H/2, W/2]
        return low, highs


class IDWT3D(nn.Module):
    def __init__(self, channels: Optional[int] = None):
        """
        channels: if provided, pre-cache the grouped deconv weight [8*C,1,2,2,2] in __init__
                  to avoid repeat(C,...) every forward. Logic identical.
        """
        super().__init__()
        k, stride, ksize = _build_haar_3d_kernels(device=torch.device('cpu'))
        self.register_buffer("kernels", k)   # [8, 1, 2, 2, 2]
        self.stride = stride
        self.ksize = ksize

        self.channels = channels
        if channels is not None:
            weight = self.kernels.repeat(channels, 1, 1, 1, 1)  # [8*C,1,2,2,2]
            self.register_buffer("weight", weight)
        else:
            self.weight = None  # type: ignore[assignment]

    def forward(self, low: torch.Tensor, highs: torch.Tensor):
        B, C, Dp, Hp, Wp = low.shape
        S = self.kernels.shape[0]  # 8

        y = torch.cat([low.unsqueeze(2), highs], dim=2)  # [B, C, 8, Dp, Hp, Wp]
        y = y.view(B, C * S, Dp, Hp, Wp)                 # [B, 8*C, Dp, Hp, Wp]

        if self.weight is None:
            weight = self.kernels.repeat(C, 1, 1, 1, 1)
        else:
            if self.channels is not None and C != self.channels:
                raise ValueError(f"IDWT3D channels mismatch: got C={C}, expected {self.channels}")
            weight = self.weight

        x = F.conv_transpose3d(
            y, weight,
            stride=self.stride,
            padding=0,
            output_padding=0,
            groups=C
        )
        return x


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages - 1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


def recommended_window_attn_config(patch_size: int | Sequence[int] = (96, 96, 96)):
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size, patch_size)

    if tuple(patch_size) == (96, 96, 96):
        return {
            "skip1": {"num_heads": 4, "window_size": (6, 6, 6)},
            "skip2": {"num_heads": 4, "window_size": (6, 6, 6)},
            "attn_dropout": 0.0,
            "proj_dropout": 0.1,
        }

    return {
        "skip1": {"num_heads": 2, "window_size": (4, 4, 4)},
        "skip2": {"num_heads": 4, "window_size": (4, 4, 4)},
        "attn_dropout": 0.0,
        "proj_dropout": 0.1,
    }


class DW_PW_1x1x1(nn.Module):
    """
    Depthwise 1x1x1 -> Pointwise 1x1x1
    (channel-wise)      (channel-mixing)
    """
    def __init__(self, channels: int, bias: bool = True):
        super().__init__()
        self.dw = nn.Conv3d(channels, channels, kernel_size=1, groups=channels, bias=bias)
        self.pw = nn.Conv3d(channels, channels, kernel_size=1, groups=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class CrossAttention3D(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=8,
        window_size: Tuple[int, int, int] = (6, 6, 6),
        attn_dropout=0.0,
        proj_dropout=0.0
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.c = channels
        self.h = num_heads
        self.dh = channels // num_heads
        self.window_size = tuple(window_size)

        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)

        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_dropout)
        self.attn_drop = attn_dropout

    # =========================
    # (3) window partition: share pad/meta across q/k/v
    # =========================
    def _compute_pad_meta(self, D: int, H: int, W: int):
        wd, wh, ww = self.window_size
        pd = (wd - D % wd) % wd
        ph = (wh - H % wh) % wh
        pw = (ww - W % ww) % ww
        Dp, Hp, Wp = D + pd, H + ph, W + pw
        meta = (D, H, W, Dp, Hp, Wp)
        pad = (pd, ph, pw)
        return meta, pad

    def _window_partition_shared(self, x: torch.Tensor, meta, pad):
        # x: [B, D, H, W, C]
        B, D, H, W, C = x.shape
        D0, H0, W0, Dp, Hp, Wp = meta
        pd, ph, pw = pad
        assert (D, H, W) == (D0, H0, W0), "meta mismatch with x spatial shape"

        wd, wh, ww = self.window_size
        if pd > 0 or ph > 0 or pw > 0:
            x = F.pad(x, (0, 0, 0, pw, 0, ph, 0, pd))

        x = x.view(B, Dp // wd, wd, Hp // wh, wh, Wp // ww, ww, C)
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, wd * wh * ww, C)
        return windows

    def _window_reverse(self, windows: torch.Tensor, shape_meta):
        D, H, W, Dp, Hp, Wp = shape_meta
        wd, wh, ww = self.window_size
        B = windows.shape[0] // ((Dp // wd) * (Hp // wh) * (Wp // ww))

        x = windows.view(B, Dp // wd, Hp // wh, Wp // ww, wd, wh, ww, self.c)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, Dp, Hp, Wp, self.c)
        return x[:, :D, :H, :W, :]

    def forward(self, q_map, k_map, v_map):
        q_map = q_map.permute(0, 2, 3, 4, 1).contiguous()  # [B,D,H,W,C]
        k_map = k_map.permute(0, 2, 3, 4, 1).contiguous()
        v_map = v_map.permute(0, 2, 3, 4, 1).contiguous()

        B0, D, H, W, C = q_map.shape
        meta, pad = self._compute_pad_meta(D, H, W)

        q = self._window_partition_shared(q_map, meta, pad)
        k = self._window_partition_shared(k_map, meta, pad)
        v = self._window_partition_shared(v_map, meta, pad)

        q = self.norm_q(q)
        k = self.norm_kv(k)
        v = self.norm_kv(v)

        B, Nq, C = q.shape
        Nk = k.shape[1]

        q = q.view(B, Nq, self.h, self.dh).transpose(1, 2)  # (BnW,h,Nq,dh)
        k = k.view(B, Nk, self.h, self.dh).transpose(1, 2)  # (BnW,h,Nk,dh)
        v = v.view(B, Nk, self.h, self.dh).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False
        )

        out = out.transpose(1, 2).contiguous().view(B, Nq, C)  # (BnW,Nq,C)
        out = self.proj_drop(self.proj(out))
        out = self._window_reverse(out, meta)
        return out.permute(0, 4, 1, 2, 3).contiguous()


class GatedFreqCrossAttn3D(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        num_high: int = 7,
        window_size: Tuple[int, int, int] = (6, 6, 6),
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.1,
        hf_embed_init_std: float = 0.02,
    ):
        super().__init__()
        self.num_high = num_high
        self.channels = channels

        # ---- (A) HF subband learnable embedding: [num_high, C]
        self.hf_embed = nn.Parameter(torch.zeros(num_high, channels))
        nn.init.normal_(self.hf_embed, mean=0.0, std=hf_embed_init_std)

        # ---- (B) DW 1x1x1 + PW 1x1x1 projections
        self.q_low = DW_PW_1x1x1(channels)
        self.k_low = DW_PW_1x1x1(channels)
        self.v_low = DW_PW_1x1x1(channels)

        self.q_high = nn.ModuleList([DW_PW_1x1x1(channels) for _ in range(num_high)])
        self.k_f = DW_PW_1x1x1(channels)
        self.v_f = DW_PW_1x1x1(channels)

        self.fuse_hi = nn.Conv3d(num_high * channels, channels, kernel_size=1, bias=False)

        self.attn_low = CrossAttention3D(
            channels, num_heads=num_heads, window_size=window_size,
            attn_dropout=attn_dropout, proj_dropout=proj_dropout
        )
        self.attn_high = nn.ModuleList([
            CrossAttention3D(
                channels, num_heads=num_heads, window_size=window_size,
                attn_dropout=attn_dropout, proj_dropout=proj_dropout
            )
            for _ in range(num_high)
        ])

        self.gate_low = nn.Sequential(nn.Conv3d(2 * channels, channels, 1, bias=True), nn.Sigmoid())
        self.gate_high = nn.Sequential(nn.Conv3d(2 * channels, channels, 1, bias=True), nn.Sigmoid())

        # ---- (C) uncertainty-guided HF boost strength (learnable, starts neutral)
        self.beta_hf = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        low: torch.Tensor,
        highs: torch.Tensor,
        gate_dec: torch.Tensor,
        unc: Optional[torch.Tensor] = None,
    ):
        """
        low:   [B, C, D, H, W]
        highs: [B, C, num_high, D, H, W]
        gate_dec: [B, C, D, H, W]
        unc:  [B, 1, D, H, W]  (normalized entropy in [0,1]), optional
        """
        B, C, D, H, W = low.shape
        assert highs.shape[:3] == (B, C, self.num_high), f"highs shape mismatch: {highs.shape}"

        # =========================
        # (2) remove highs_list loop + cat (keep identical logic)
        # =========================
        # add per-subband embedding in a vectorized way
        # hf_embed: [num_high, C] -> broadcast to [B, C, num_high, D, H, W]
        hf = self.hf_embed.view(1, self.num_high, C, 1, 1, 1).permute(0, 2, 1, 3, 4, 5)
        highs_emb = highs + hf  # [B, C, num_high, D, H, W]

        # fuse HF for LF query (same as cat(highs_list, dim=1) then 1x1)
        f_in = highs_emb.permute(0, 2, 1, 3, 4, 5).reshape(B, self.num_high * C, D, H, W)
        f = self.fuse_hi(f_in)  # [B, C, D, H, W]

        # --- projections
        q_low = self.q_low(low)
        k_low = self.k_low(low)
        v_low = self.v_low(low)

        k_f = self.k_f(f)
        v_f = self.v_f(f)

        # --- LF <- HF (cross-attn)
        delta_low = self.attn_low(q_low, k_f, v_f)
        alpha_low = self.gate_low(torch.cat([low, gate_dec], dim=1))
        low_out = low + alpha_low * delta_low

        # --- uncertainty-guided HF scale (broadcastable)
        if unc is None:
            scale = 1.0
        else:
            unc = unc.clamp(0.0, 1.0)
            beta = torch.sigmoid(self.beta_hf)
            scale = 1.0 + beta * unc

        highs_out = []
        for i in range(self.num_high):
            hi = highs_emb[:, :, i, ...]  # [B,C,D,H,W] (equiv to highs_list[i])
            q_hi = self.q_high[i](hi)
            delta_hi = self.attn_high[i](q_hi, k_low, v_low)
            alpha_hi = self.gate_high(torch.cat([hi, gate_dec], dim=1))  # [B,C,D,H,W]
            highs_out.append(hi + (alpha_hi * scale) * delta_hi)

        highs_out = torch.stack(highs_out, dim=2)  # [B, C, num_high, D, H, W]
        return low_out, highs_out


class CrossDomainBlcok(nn.Module):
    def __init__(self, c: int, norm: str = "instancenorm"):
        super().__init__()
        inter = c
        self.spa_mlp = nn.Sequential(
            nn.Conv3d(2 * c, inter, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter, c, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.fre_mlp = nn.Sequential(
            nn.Conv3d(2 * c, inter, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter, c, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.s2f = nn.Sequential(nn.Conv3d(c, c, 1, bias=False), norm3d(norm, c), nn.Sigmoid())
        self.f2s = nn.Sequential(nn.Conv3d(c, c, 1, bias=False), norm3d(norm, c), nn.Sigmoid())

        self.proj = nn.Sequential(
            nn.Conv3d(2 * c, c, kernel_size=1, bias=False),
            norm3d(norm, c),
            nn.ReLU(inplace=True),
        )

    def _dual_stats(self, x: torch.Tensor) -> torch.Tensor:
        avg_stat = F.adaptive_avg_pool3d(x, 1)
        max_stat = F.adaptive_max_pool3d(x, 1)
        return torch.cat([avg_stat, max_stat], dim=1)

    def forward(self, x_spa: torch.Tensor, x_fre: torch.Tensor) -> torch.Tensor:
        w_spa = self.spa_mlp(self._dual_stats(x_spa))
        w_fre = self.fre_mlp(self._dual_stats(x_fre))

        x_spa_cal = x_spa * w_spa
        x_fre_cal = x_fre * w_fre

        y_spa = x_spa_cal * self.f2s(x_fre_cal)
        y_fre = x_fre_cal * self.s2f(x_spa_cal)

        y = torch.cat([y_spa, y_fre], dim=1)
        return self.proj(y)


class MFEC(nn.Module):
    def __init__(
        self,
        channels: int,
        normalization: NormType = "instancenorm",
        num_experts: int = 6,
        gate_hidden_ratio: float = 0.25,
        spatial_gate_hidden: Optional[int] = None,
    ):
        super().__init__()
        self.channels = channels
        self.num_experts = num_experts

        def _dw_pw_expert(kernel_size, padding, dilation=1):
            return nn.Sequential(
                nn.Conv3d(
                    channels, channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                    groups=channels,
                    bias=False
                ),
                nn.Conv3d(channels, channels, kernel_size=1, bias=False),
                norm3d(normalization, channels),
                nn.ReLU(inplace=True),
            )

        exp1 = _dw_pw_expert(kernel_size=3, padding=1)                          # 3x3x3
        exp2 = _dw_pw_expert(kernel_size=5, padding=2)                          # 5x5x5
        exp3 = _dw_pw_expert(kernel_size=(1, 3, 3), padding=(0, 1, 1))           # 1x3x3
        exp4 = _dw_pw_expert(kernel_size=(5, 1, 1), padding=(2, 0, 0))           # 5x1x1
        exp5 = _dw_pw_expert(kernel_size=3, padding=2, dilation=2)               # dilated 3x3x3 (d=2)
        exp6 = _dw_pw_expert(kernel_size=(3, 5, 5), padding=(1, 2, 2))           # 3x5x5

        self.experts = nn.ModuleList([exp1, exp2, exp3, exp4, exp5, exp6])

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.maxpool = nn.AdaptiveMaxPool3d(1)

        hidden = max(int(channels * gate_hidden_ratio), 8)
        self.gate_global = nn.Sequential(
            nn.Conv3d(3 * channels + 2, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, self.num_experts, kernel_size=1, bias=True),
        )

        if spatial_gate_hidden is None:
            spatial_gate_hidden = max(channels // 2, 16)

        self.gate_spatial = nn.Sequential(
            nn.Conv3d(channels + 1, spatial_gate_hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(spatial_gate_hidden, spatial_gate_hidden, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(spatial_gate_hidden, self.num_experts, kernel_size=1, bias=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv3d(self.num_experts * channels, channels, kernel_size=1, bias=False),
            norm3d(normalization, channels),
            nn.ReLU(inplace=True),
        )

    def _global_stats(self, x: torch.Tensor, unc_in: torch.Tensor) -> torch.Tensor:
        """
        x      : [B,C,D,H,W]
        unc_in : [B,1,D,H,W]
        return : [B,3C+2,1,1,1]
        """
        avg = self.avgpool(x)
        mx = self.maxpool(x)
        mn = -self.maxpool(-x)

        unc_avg = self.avgpool(unc_in)
        unc_max = self.maxpool(unc_in)

        return torch.cat([avg, mx, mn, unc_avg, unc_max], dim=1)

    def forward(self, x: torch.Tensor, unc_in: torch.Tensor) -> torch.Tensor:
        """
        x      : [B,C,D,H,W]
        unc_in : [B,1,D,H,W]
        """
        if unc_in.dim() != 5 or unc_in.shape[1] != 1:
            raise ValueError(f"unc_in must be [B,1,D,H,W], got {tuple(unc_in.shape)}")

        unc_in = unc_in.to(dtype=x.dtype, device=x.device).clamp_(0.0, 1.0)

        g_global = self.gate_global(self._global_stats(x, unc_in))               # [B,E,1,1,1]
        g_spatial = self.gate_spatial(torch.cat([x, unc_in], dim=1))             # [B,E,D,H,W]
        gate_logits = g_spatial + g_global                                       # broadcast

        weights = torch.softmax(gate_logits, dim=1)                               # [B,E,D,H,W]

        outs = []
        for e, expert in enumerate(self.experts):
            outs.append(expert(x) * weights[:, e:e + 1])

        y = torch.cat(outs, dim=1)                                                # [B,E*C,D,H,W]
        y = self.fuse(y)                                                          # [B,C,D,H,W]
        return x + y


class SkipRefinement(nn.Module):
    def __init__(self, c: int, num_heads: int, g2_channels: int, normalization: NormType = "instancenorm",
                 window_size: Tuple[int, int, int] = (6, 6, 6), attn_dropout: float = 0.0, proj_dropout: float = 0.1,
                 gate_hidden_ratio: float = 0.25):
        super().__init__()
        self.dwt = DWT3D(channels=c)      # 你缓存 weight 的版本
        self.idwt = IDWT3D(channels=c)    # 你缓存 weight 的版本

        self.spa = MFEC(c, normalization=normalization, gate_hidden_ratio=gate_hidden_ratio)
        self.fre = GatedFreqCrossAttn3D(
            c, num_heads=num_heads, num_high=7, window_size=window_size,
            attn_dropout=attn_dropout, proj_dropout=proj_dropout
        )
        self.fuse = CrossDomainBlcok(c, norm=normalization)

        self.g2_proj = nn.Sequential(
            nn.Conv3d(g2_channels, c, kernel_size=1, bias=False),
            norm3d(normalization, c),
        )

    @staticmethod
    def _upsample_unc_to_x(unc: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # unc: [B,1,Du,Hu,Wu] -> [B,1,Dx,Hx,Wx]
        if unc.shape[-3:] == x.shape[-3:]:
            return unc
        # entropy map 是连续值：trilinear 更合理
        return F.interpolate(unc, size=x.shape[-3:], mode="trilinear", align_corners=False)

    def forward(self, x: torch.Tensor, g2: torch.Tensor, unc: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x   : [B,C,D,H,W]          (skip feature, e.g. 48^3 or 96^3)
        g2  : decoder guidance     (same as your original)
        unc : [B,1,D/2,H/2,W/2]    (matches DWT(low) scale by design: 24^3 or 48^3)
        """
        # --- spatial branch: needs unc at x scale
        if unc is None:
            unc_spa = torch.zeros((x.shape[0], 1, *x.shape[-3:]), device=x.device, dtype=x.dtype)
        else:
            unc_spa = unc.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)
            unc_spa = self._upsample_unc_to_x(unc_spa, x)

        x_spa = self.spa(x, unc_spa)

        # --- frequency branch: keep unc at low scale (NO interpolation)
        low, highs = self.dwt(x)  # low: [B,C,D/2,H/2,W/2]
        g2 = self.g2_proj(g2)

        if unc is None:
            unc_fre = None
        else:
            unc_fre = unc.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)
            # 强约束：按你的设定，unc 必须匹配 low 尺度
            if unc_fre.shape[-3:] != low.shape[-3:]:
                raise RuntimeError(
                    f"unc_fre must match low scale. got {tuple(unc_fre.shape[-3:])}, "
                    f"expected {tuple(low.shape[-3:])}"
                )

        low2, highs2 = self.fre(low, highs, g2, unc=unc_fre)
        x_fre = self.idwt(low2, highs2)

        out = self.fuse(x_spa, x_fre)
        return out


class Encoder(nn.Module):
    def __init__(self, n_channels=1, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]
        return res


class Decoder(nn.Module):
    def __init__(
        self,
        n_classes=14,
        n_filters=16,
        normalization='none',
        has_dropout=False,
        has_residual=False,
        attn_cfg: Optional[dict] = None
    ):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.aux_skip2 = nn.Conv3d(n_filters * 4, n_classes, 1, padding=0)  # from x6_up (24^3)
        self.aux_skip1 = nn.Conv3d(n_filters * 2, n_classes, 1, padding=0)  # from x7_up (48^3)

        if attn_cfg is None:
            attn_cfg = recommended_window_attn_config((96, 96, 96))

        skip1_cfg = attn_cfg.get("skip1", {})
        skip2_cfg = attn_cfg.get("skip2", {})
        attn_dropout = attn_cfg.get("attn_dropout", 0.0)
        proj_dropout = attn_cfg.get("proj_dropout", 0.1)

        self.skip1 = SkipRefinement(
            n_filters,
            num_heads=skip1_cfg.get("num_heads", 4),
            g2_channels=n_filters * 2,
            normalization=normalization,
            window_size=skip1_cfg.get("window_size", (6, 6, 6)),
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            gate_hidden_ratio=0.25
        )
        self.skip2 = SkipRefinement(
            n_filters * 2,
            num_heads=skip2_cfg.get("num_heads", 4),
            g2_channels=n_filters * 4,
            normalization=normalization,
            window_size=skip2_cfg.get("window_size", (6, 6, 6)),
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            gate_hidden_ratio=0.5
        )
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features, return_aux: bool = False):
        x1, x2, x3, x4, x5 = features

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        # --- aux @ 24^3 (supervised + for uncertainty)
        logits_s2 = self.aux_skip2(x6_up)  # [B,K,24,24,24]
        with torch.no_grad():
            unc_s2 = predictive_entropy_from_logits(logits_s2)  # [B,1,24,24,24]

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)  # [B,32,48,48,48]

        # --- aux @ 48^3 (supervised + for uncertainty)
        logits_s1 = self.aux_skip1(x7_up)  # [B,K,48,48,48]
        with torch.no_grad():
            unc_s1 = predictive_entropy_from_logits(logits_s1)  # [B,1,48,48,48]

        # --- refine skip2 using unc_s2
        x2 = self.skip2(x2, x6_up, unc_s2)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)

        # --- refine skip1 using unc_s1
        x1 = self.skip1(x1, x7_up, unc_s1)
        x8_up = x8_up + x1

        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)

        out_seg = self.out_conv(x9)  # [B,K,96,96,96]

        if return_aux:
            return out_seg, logits_s1, logits_s2
        return out_seg


class VNet(nn.Module):
    def __init__(
        self,
        n_channels=1,
        n_classes=14,
        patch_size=96,
        n_filters=16,
        normalization='instancenorm',
        has_dropout=False,
        has_residual=False,
        input_layout: str = "NCHWD"
    ):
        super(VNet, self).__init__()
        self.num_classes = n_classes
        self.input_layout = input_layout
        self.encoder = Encoder(n_channels, n_filters, normalization, has_dropout, has_residual)
        self.decoder = Decoder(n_classes, n_filters, normalization, has_dropout, has_residual)

    def _to_ncdhw(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_layout.upper() == "NCHWD":
            return x.permute(0, 1, 4, 2, 3).contiguous()
        return x

    def _to_nchwd(self, y: torch.Tensor) -> torch.Tensor:
        if self.input_layout.upper() == "NCHWD":
            return y.permute(0, 1, 3, 4, 2).contiguous()
        return y

    def forward(self, input: torch.Tensor, return_aux: bool = False):
        x = self._to_ncdhw(input)
        features = self.encoder(x)
        out = self.decoder(features, return_aux=return_aux)
        if return_aux:
            out_seg, logits_s1, logits_s2 = out
            out_seg = self._to_nchwd(out_seg)
            logits_s1 = self._to_nchwd(logits_s1)
            logits_s2 = self._to_nchwd(logits_s2)
            return out_seg, logits_s1, logits_s2

        out_seg = out
        out_seg = self._to_nchwd(out_seg)
        return out_seg
