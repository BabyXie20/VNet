from __future__ import annotations
from typing import Literal, Optional
from typing import Sequence, Tuple
import torch
from torch import nn
import torch.nn.functional as F
import math

NormType = Optional[Literal["batchnorm", "groupnorm", "instancenorm", "none"]]


def _best_group_count(num_channels: int, preferred: int = 16) -> int:
    g = min(preferred, num_channels)
    while g > 1 and (num_channels % g) != 0:
        g -= 1
    return g


def norm3d(norm: NormType, num_channels: int, affine: bool = True) -> nn.Module:
    if norm in ("batchnorm", "groupnorm", "instancenorm"):
        num_groups = _best_group_count(num_channels, preferred=8)
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
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


class DWT3D(nn.Module):
    def __init__(self):
        super().__init__()
        k, stride, ksize = _build_haar_3d_kernels(device=torch.device('cpu'))
        # k: [8, 1, 2, 2, 2]
        self.register_buffer("kernels", k)  
        self.stride = stride                 
        self.ksize = ksize                   

    def forward(self, x: torch.Tensor):
        B, C, D, H, W = x.shape
        S = self.kernels.shape[0]           # 8
        # [C*8, 1, 2,2,2]
        weight = self.kernels.repeat(C, 1, 1, 1, 1)

        y = F.conv3d(x, weight, stride=self.stride, padding=0, groups=C)

        y = y.view(B, C, S, *y.shape[-3:])  # [B, C, 8, D/2, H/2, W/2]
        low = y[:, :, :1, ...].squeeze(2)   # [B, C, D/2, H/2, W/2]
        highs = y[:, :, 1:, ...]            # [B, C, 7, D/2, H/2, W/2]
        return low, highs


class IDWT3D(nn.Module):
    def __init__(self):
        super().__init__()
        k, stride, ksize = _build_haar_3d_kernels(device=torch.device('cpu'))
        self.register_buffer("kernels", k)   # [8, 1, 2, 2, 2]
        self.stride = stride
        self.ksize = ksize

    def forward(self, low: torch.Tensor, highs: torch.Tensor):
        B, C, Dp, Hp, Wp = low.shape
        S = self.kernels.shape[0]        

        y = torch.cat([low.unsqueeze(2), highs], dim=2)  
        y = y.view(B, C * S, Dp, Hp, Wp)                 

        weight = self.kernels.repeat(C, 1, 1, 1, 1)      

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
            if normalization in ('batchnorm', 'groupnorm', 'instancenorm'):
                ops.append(nn.GroupNorm(num_groups=_best_group_count(n_filters_out, preferred=8), num_channels=n_filters_out))
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
            if normalization in ('batchnorm', 'groupnorm', 'instancenorm'):
                ops.append(nn.GroupNorm(num_groups=_best_group_count(n_filters_out, preferred=8), num_channels=n_filters_out))
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
            if normalization in ('batchnorm', 'groupnorm', 'instancenorm'):
                ops.append(nn.GroupNorm(num_groups=_best_group_count(n_filters_out, preferred=8), num_channels=n_filters_out))
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
            if normalization in ('batchnorm', 'groupnorm', 'instancenorm'):
                ops.append(nn.GroupNorm(num_groups=_best_group_count(n_filters_out, preferred=8), num_channels=n_filters_out))
            else:
                assert False
        else:

            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class DWC1x1x1(nn.Module):
    def __init__(self, channels, bias=True):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=1, groups=channels, bias=bias)

    def forward(self, x):
        return self.conv(x)


class CrossAttention3D(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=8,
        window_size: Tuple[int, int, int] = (6, 8, 8),
        shift_size: Tuple[int, int, int] = (3, 4, 4),
        attn_dropout=0.0,
        proj_dropout=0.0,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.c = channels
        self.h = num_heads
        self.dh = channels // num_heads
        self.window_size = tuple(window_size)
        self.shift_size = tuple(shift_size)

        wd, wh, ww = self.window_size
        sd, sh, sw = self.shift_size
        assert 0 <= sd < wd and 0 <= sh < wh and 0 <= sw < ww, "shift_size must be in [0, window_size)"

        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)

        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_dropout)
        self.attn_drop = float(attn_dropout)

        # 简单缓存：按 (Dp,Hp,Wp,device,dtype) 缓存 mask
        self._mask_cache = {}

    def _pad_to_window(self, x: torch.Tensor):
        # x: [B, D, H, W, C]
        B, D, H, W, C = x.shape
        wd, wh, ww = self.window_size
        pd = (wd - D % wd) % wd
        ph = (wh - H % wh) % wh
        pw = (ww - W % ww) % ww
        if pd or ph or pw:
            x = F.pad(x, (0, 0, 0, pw, 0, ph, 0, pd))
        return x, (D, H, W, D + pd, H + ph, W + pw, pd, ph, pw)

    def _window_partition_no_pad(self, x: torch.Tensor):
        # x: [B, Dp, Hp, Wp, C] where divisible by window_size
        B, Dp, Hp, Wp, C = x.shape
        wd, wh, ww = self.window_size
        x = x.view(B, Dp // wd, wd, Hp // wh, wh, Wp // ww, ww, C)
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        windows = windows.view(-1, wd * wh * ww, C)  # [B*nW, N, C]
        return windows

    def _window_reverse_no_pad(self, windows: torch.Tensor, B: int, Dp: int, Hp: int, Wp: int):
        # windows: [B*nW, N, C]
        wd, wh, ww = self.window_size
        C = windows.shape[-1]
        nW = (Dp // wd) * (Hp // wh) * (Wp // ww)
        assert windows.shape[0] == B * nW
        x = windows.view(B, Dp // wd, Hp // wh, Wp // ww, wd, wh, ww, C)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, Dp, Hp, Wp, C)
        return x

    def _get_attn_mask(self, Dp: int, Hp: int, Wp: int, device, dtype):
        # 返回 shape: [nW, N, N] (单个样本的所有窗口)
        key = (Dp, Hp, Wp, device.type, device.index, str(dtype))
        if key in self._mask_cache:
            return self._mask_cache[key]

        wd, wh, ww = self.window_size
        sd, sh, sw = self.shift_size
        N = wd * wh * ww

        # img_mask: [1, Dp, Hp, Wp, 1]，按 3x3x3 区域打标签（Swin 标准做法的 3D 扩展）
        img_mask = torch.zeros((1, Dp, Hp, Wp, 1), device=device, dtype=torch.int32)
        cnt = 0

        d_slices = (slice(0, -wd), slice(-wd, -sd), slice(-sd, None)) if sd > 0 else (slice(0, None),)
        h_slices = (slice(0, -wh), slice(-wh, -sh), slice(-sh, None)) if sh > 0 else (slice(0, None),)
        w_slices = (slice(0, -ww), slice(-ww, -sw), slice(-sw, None)) if sw > 0 else (slice(0, None),)

        for ds in d_slices:
            for hs in h_slices:
                for ws in w_slices:
                    img_mask[:, ds, hs, ws, :] = cnt
                    cnt += 1

        mask_windows = self._window_partition_no_pad(img_mask).view(-1, N)  # [nW, N]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)    # [nW, N, N]

        neg = torch.finfo(dtype).min
        attn_mask = attn_mask.to(device=device)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, neg).masked_fill(attn_mask == 0, 0.0)
        attn_mask = attn_mask.to(dtype=dtype)  # 与 q 同 dtype 更省显存

        self._mask_cache[key] = attn_mask
        return attn_mask

    def forward(self, q_map, k_map, v_map):
        # 输入: [B,C,D,H,W]
        q_map = q_map.permute(0, 2, 3, 4, 1).contiguous()  # [B,D,H,W,C]
        k_map = k_map.permute(0, 2, 3, 4, 1).contiguous()
        v_map = v_map.permute(0, 2, 3, 4, 1).contiguous()

        B0, D, H, W, C = q_map.shape

        # 1) pad（先 pad，再 shift）
        q_pad, meta = self._pad_to_window(q_map)
        _, _, _, Dp, Hp, Wp, pd, ph, pw = meta

        if pd or ph or pw:
            k_pad = F.pad(k_map, (0, 0, 0, pw, 0, ph, 0, pd))
            v_pad = F.pad(v_map, (0, 0, 0, pw, 0, ph, 0, pd))
        else:
            k_pad, v_pad = k_map, v_map

        # 2) cyclic shift
        sd, sh, sw = self.shift_size
        use_shift = (sd != 0) or (sh != 0) or (sw != 0)
        if use_shift:
            q_pad = torch.roll(q_pad, shifts=(-sd, -sh, -sw), dims=(1, 2, 3))
            k_pad = torch.roll(k_pad, shifts=(-sd, -sh, -sw), dims=(1, 2, 3))
            v_pad = torch.roll(v_pad, shifts=(-sd, -sh, -sw), dims=(1, 2, 3))

        # 3) window partition（此时 Dp/Hp/Wp 都能整除 window_size）
        q = self._window_partition_no_pad(q_pad)  # [B0*nW, N, C]
        k = self._window_partition_no_pad(k_pad)
        v = self._window_partition_no_pad(v_pad)

        # LN
        q = self.norm_q(q)
        k = self.norm_kv(k)
        v = self.norm_kv(v)

        Bwin, Nq, _ = q.shape  # Bwin = B0*nW
        Nk = k.shape[1]
        assert Nq == Nk, "Swin window attention expects same token count per window."

        # reshape to SDPA
        q = q.view(Bwin, Nq, self.h, self.dh).transpose(1, 2)  # [Bwin,h,N,dh]
        k = k.view(Bwin, Nk, self.h, self.dh).transpose(1, 2)
        v = v.view(Bwin, Nk, self.h, self.dh).transpose(1, 2)

        # 4) attention mask（只在 shift 时需要）
        attn_mask = None
        if use_shift:
            # 单样本 mask: [nW, N, N]，扩展到 batch: [B0*nW, N, N]
            mask_1 = self._get_attn_mask(Dp, Hp, Wp, device=q.device, dtype=q.dtype)
            nW = mask_1.shape[0]
            mask = mask_1.repeat(B0, 1, 1)  # [B0*nW, N, N]
            attn_mask = mask.unsqueeze(1)   # [B0*nW, 1, N, N] -> broadcast 到 heads

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False
        )  # [Bwin,h,N,dh]

        out = out.transpose(1, 2).contiguous().view(Bwin, Nq, C)  # [Bwin,N,C]
        out = self.proj_drop(self.proj(out))

        # 5) reverse windows -> padded feature
        out = self._window_reverse_no_pad(out, B0, Dp, Hp, Wp)  # [B0,Dp,Hp,Wp,C]

        # 6) reverse shift + unpad
        if use_shift:
            out = torch.roll(out, shifts=(sd, sh, sw), dims=(1, 2, 3))

        out = out[:, :D, :H, :W, :]  
        return out.permute(0, 4, 1, 2, 3).contiguous()  # [B,C,D,H,W]


def recommended_window_attn_config(patch_size: int | Sequence[int] = (96, 96, 96)):
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size, patch_size)

    if tuple(patch_size) == (96, 96, 96):
        return {
            "skip1": {"num_heads": 4, "window_size": (6, 8, 8)},
            "skip2": {"num_heads": 4, "window_size": (6, 8, 8)},
            "attn_dropout": 0.0,
            "proj_dropout": 0.1,
        }

    return {
        "skip1": {"num_heads": 2, "window_size": (4, 4, 4)},
        "skip2": {"num_heads": 4, "window_size": (4, 4, 4)},
        "attn_dropout": 0.0,
        "proj_dropout": 0.1,
    }


class GatedFreqCrossAttn3D(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8, num_high: int = 7, window_size: Tuple[int, int, int] = (6, 8, 8), attn_dropout: float = 0.0, proj_dropout: float = 0.1):
        super().__init__()
        self.num_high = num_high

        self.q_low = DWC1x1x1(channels)
        self.k_low = DWC1x1x1(channels)
        self.v_low = DWC1x1x1(channels)

        self.q_high = nn.ModuleList([DWC1x1x1(channels) for _ in range(num_high)])
        self.k_f = DWC1x1x1(channels)
        self.v_f = DWC1x1x1(channels)

        self.fuse_hi = nn.Conv3d(num_high * channels, channels, kernel_size=1, bias=False)

        ws = window_size
        ss = (ws[0] // 2, ws[1] // 2, ws[2] // 2)

        self.attn_low = CrossAttention3D(
            channels, num_heads=num_heads,
            window_size=ws, shift_size=ss,
            attn_dropout=attn_dropout, proj_dropout=proj_dropout
        )
        self.attn_high = nn.ModuleList([
            CrossAttention3D(
                channels, num_heads=num_heads,
                window_size=ws, shift_size=ss,
                attn_dropout=attn_dropout, proj_dropout=proj_dropout
            )
            for _ in range(num_high)
        ])

        self.gate_low = nn.Sequential(nn.Conv3d(2 * channels, channels, 1, bias=True), nn.Sigmoid())
        self.gate_high = nn.Sequential(nn.Conv3d(2 * channels, channels, 1, bias=True), nn.Sigmoid())

    def forward(self, low: torch.Tensor, highs: torch.Tensor, gate_dec: torch.Tensor):
        B, C, D, H, W = low.shape

        highs_list = [highs[:, :, i, ...] for i in range(self.num_high)]  

        f = self.fuse_hi(torch.cat(highs_list, dim=1))  

        q_low = self.q_low(low)
        k_low = self.k_low(low)
        v_low = self.v_low(low)

        k_f = self.k_f(f)
        v_f = self.v_f(f)

        delta_low = self.attn_low(q_low, k_f, v_f)
        alpha_low = self.gate_low(torch.cat([low, gate_dec], dim=1))
        low_out = low + alpha_low * delta_low

        highs_out = []
        for i in range(self.num_high):
            q_hi = self.q_high[i](highs_list[i])
            delta_hi = self.attn_high[i](q_hi, k_low, v_low)
            alpha_hi = self.gate_high(torch.cat([highs_list[i], gate_dec], dim=1))
            highs_out.append(highs_list[i] + alpha_hi * delta_hi)

        highs_out = torch.stack(highs_out, dim=2)  
        return low_out, highs_out
    

class CrossDomainBlcok(nn.Module):
    def __init__(self, c: int, norm: str = "groupnorm"):
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
    def __init__(self, channels: int, normalization: NormType = "groupnorm", num_experts: int = 5):
        super().__init__()
        if num_experts != 5:
            raise ValueError("MFEC is configured for 5 organ-aware experts.")

        # Expert-1: isotropic local context (kidney/pancreas boundary details)
        exp1 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            norm3d(normalization, channels),
            nn.ReLU(inplace=True),
        )
        # Expert-2: large field for big organs (liver/spleen/stomach)
        exp2 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            norm3d(normalization, channels),
            nn.ReLU(inplace=True),
        )
        # Expert-3: anisotropic in-plane (handles thicker z-spacing common in CT)
        exp3 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), groups=channels, bias=False),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            norm3d(normalization, channels),
            nn.ReLU(inplace=True),
        )
        # Expert-4: superior-inferior continuity (vessels / elongated structures)
        exp4 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=channels, bias=False),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            norm3d(normalization, channels),
            nn.ReLU(inplace=True),
        )
        # Expert-5: dilated context for small tubular targets (aorta/IVC/esophagus)
        exp5 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels, bias=False),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            norm3d(normalization, channels),
            nn.ReLU(inplace=True),
        )
        self.experts = nn.ModuleList([exp1, exp2, exp3, exp4, exp5])

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, max(channels // 4, 8), kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(max(channels // 4, 8), num_experts, kernel_size=1, bias=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv3d(num_experts * channels, channels, kernel_size=1, bias=False),
            norm3d(normalization, channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.gate(x), dim=1)  # [B, E, 1, 1, 1]

        weighted_expert_outs = []
        for idx, expert in enumerate(self.experts):
            y = expert(x)
            y = y * weights[:, idx:idx + 1]
            weighted_expert_outs.append(y)

        y = torch.cat(weighted_expert_outs, dim=1)
        y = self.fuse(y)
        return x + y


class SkipRefinement(nn.Module):
    def __init__(self, c: int, num_heads: int, g2_channels: int, normalization: NormType = "groupnorm",
                 window_size: Tuple[int, int, int] = (6, 6, 6), attn_dropout: float = 0.0, proj_dropout: float = 0.1):
        super().__init__()
        self.dwt = DWT3D()
        self.idwt = IDWT3D()

        self.spa = MFEC(c, normalization=normalization)
        self.fre = GatedFreqCrossAttn3D(
            c, num_heads=num_heads, num_high=7, window_size=window_size, attn_dropout=attn_dropout, proj_dropout=proj_dropout
        )
        self.fuse = CrossDomainBlcok(c, norm=normalization)

        self.g2_proj = nn.Sequential(
            nn.Conv3d(g2_channels, c, kernel_size=1, bias=False),
            norm3d(normalization, c),
        )

    def forward(self, x: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
        x_spa = self.spa(x)

        low, highs = self.dwt(x)

        g2 = self.g2_proj(g2)

        low2, highs2 = self.fre(low, highs, g2)
        x_fre = self.idwt(low2, highs2)

        out = self.fuse(x_spa, x_fre)
        return out

    
class Encoder(nn.Module):
    def __init__(self, n_channels=1,n_filters=16, normalization='none', has_dropout=False,has_residual=False):
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
    def __init__(self,n_classes=14, n_filters=16, normalization='none', has_dropout=False, has_residual=False,
                 attn_cfg: Optional[dict] = None):
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

        if attn_cfg is None:
            attn_cfg = recommended_window_attn_config((96, 96, 96))

        skip1_cfg = attn_cfg.get("skip1", {})
        skip2_cfg = attn_cfg.get("skip2", {})
        attn_dropout = attn_cfg.get("attn_dropout", 0.0)
        proj_dropout = attn_cfg.get("proj_dropout", 0.1)

        self.skip1 = SkipRefinement(
            n_filters,
            num_heads=skip1_cfg.get("num_heads", 4),
            g2_channels=n_filters*2,
            normalization=normalization,
            window_size=skip1_cfg.get("window_size", (6, 8, 8)),
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )
        self.skip2 = SkipRefinement(
            n_filters*2,
            num_heads=skip2_cfg.get("num_heads", 4),
            g2_channels=n_filters*4,
            normalization=normalization,
            window_size=skip2_cfg.get("window_size", (6, 8, 8)),
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )
        self.skip1_mfec = MFEC(n_filters, normalization=normalization)
        self.skip2_mfec = MFEC(n_filters * 2, normalization=normalization)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)

        x2=self.skip2(x2,x6_up)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)

        x1=self.skip1(x1,x7_up)
        x8_up = x8_up + x1

        x9 = self.block_nine(x8_up)

        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)
        return out_seg


class VNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=14, patch_size=96, n_filters=16,
                 normalization='groupnorm', has_dropout=False, has_residual=False,
                 input_layout: str = "NCHWD"):   
        super(VNet, self).__init__()
        self.num_classes = n_classes
        self.input_layout = input_layout
        self.encoder = Encoder(
        n_channels=n_channels,
        n_filters=n_filters,
        normalization=normalization,
        has_dropout=has_dropout,
        has_residual=has_residual
        )

        self.decoder = Decoder(n_classes, n_filters, normalization, has_dropout, has_residual)

    def _to_ncdhw(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_layout.upper() == "NCHWD":
            return x.permute(0, 1, 4, 2, 3).contiguous()  
        return x 

    def _to_nchwd(self, y: torch.Tensor) -> torch.Tensor:
        if self.input_layout.upper() == "NCHWD":
            return y.permute(0, 1, 3, 4, 2).contiguous()  
        return y

    def forward(self, input: torch.Tensor):
        x = self._to_ncdhw(input)  
        features = self.encoder(x)
        out_seg  = self.decoder(features) 
        out_seg  = self._to_nchwd(out_seg) 
        return out_seg
