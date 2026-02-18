from __future__ import annotations
from typing import Literal, Optional
from typing import Sequence, Tuple
import torch
from torch import nn
import torch.nn.functional as F
import math

NormType = Optional[Literal["batchnorm", "groupnorm", "instancenorm", "none"]]


def predictive_entropy_from_logits(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = torch.softmax(logits, dim=1)
    ent = -(p * torch.log(p.clamp_min(eps))).sum(dim=1, keepdim=True)
    ent = ent / math.log(p.shape[1])
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
                taps.append(_kron3(zf, yf, xf))
    k = torch.stack(taps, dim=0)[:, None, ...]
    stride = (2, 2, 2)
    ksize = (2, 2, 2)
    return k.to(device), stride, ksize


class DWT3D(nn.Module):
    def __init__(self):
        super().__init__()
        k, stride, ksize = _build_haar_3d_kernels(device=torch.device('cpu'))

        self.register_buffer("kernels", k)
        self.stride = stride
        self.ksize = ksize

    def forward(self, x: torch.Tensor):
        # in: x [B,C,D,H,W]; out: low [B,C,D/2,H/2,W/2], highs [B,C,7,D/2,H/2,W/2]
        B, C, D, H, W = x.shape
        S = self.kernels.shape[0]

        weight = self.kernels.repeat(C, 1, 1, 1, 1)

        y = F.conv3d(x, weight, stride=self.stride, padding=0, groups=C)

        y = y.view(B, C, S, *y.shape[-3:])
        low = y[:, :, :1, ...].squeeze(2)
        highs = y[:, :, 1:, ...]
        return low, highs


class IDWT3D(nn.Module):
    def __init__(self):
        super().__init__()
        k, stride, ksize = _build_haar_3d_kernels(device=torch.device('cpu'))
        self.register_buffer("kernels", k)
        self.stride = stride
        self.ksize = ksize

    def forward(self, low: torch.Tensor, highs: torch.Tensor):
        # in: low [B,C,D/2,H/2,W/2], highs [B,C,7,D/2,H/2,W/2]; out: x [B,C,D,H,W]
        B, C, Dp, Hp, Wp = low.shape
        S = self.kernels.shape[0]

        y = torch.cat([low.unsqueeze(2), highs], dim=2)
        y = y.view(B, C * S, Dp, Hp, Wp)

        weight = self.kernels.repeat(C, 1, 1, 1, 1)

        x = F.conv_transpose3d(y, weight,stride=self.stride,padding=0,output_padding=0,groups=C)
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
        # in: x [B,Cin,D,H,W]; out: y [B,Cout,D,H,W]
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
        # in: x [B,Cin,D,H,W]; out: y [B,Cout,D,H,W]
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
        # in: x [B,Cin,D,H,W]; out: y [B,Cout,D/stride,H/stride,W/stride]
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
        # in: x [B,Cin,D,H,W]; out: y [B,Cout,D*stride,H*stride,W*stride]
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
    def __init__(self, channels: int, bias: bool = True):
        super().__init__()
        self.dw = nn.Conv3d(channels, channels, kernel_size=1, groups=channels, bias=bias)
        self.pw = nn.Conv3d(channels, channels, kernel_size=1, groups=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # in: x [B,C,D,H,W]; out: y [B,C,D,H,W]
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
        self.c = channels
        self.h = num_heads
        self.dh = channels // num_heads
        self.window_size = tuple(window_size)

        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)

        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_dropout)
        self.attn_drop = attn_dropout

    def _window_partition(self, x: torch.Tensor):

        B, D, H, W, C = x.shape
        wd, wh, ww = self.window_size

        pd = (wd - D % wd) % wd
        ph = (wh - H % wh) % wh
        pw = (ww - W % ww) % ww
        if pd > 0 or ph > 0 or pw > 0:
            x = F.pad(x, (0, 0, 0, pw, 0, ph, 0, pd))

        Dp, Hp, Wp = D + pd, H + ph, W + pw
        x = x.view(B, Dp // wd, wd, Hp // wh, wh, Wp // ww, ww, C)
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, wd * wh * ww, C)
        return windows, (D, H, W, Dp, Hp, Wp)

    def _window_reverse(self, windows: torch.Tensor, shape_meta):
        D, H, W, Dp, Hp, Wp = shape_meta
        wd, wh, ww = self.window_size
        B = windows.shape[0] // ((Dp // wd) * (Hp // wh) * (Wp // ww))

        x = windows.view(B, Dp // wd, Hp // wh, Wp // ww, wd, wh, ww, self.c)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, Dp, Hp, Wp, self.c)
        return x[:, :D, :H, :W, :]

    def forward(self, q_map, k_map, v_map):
        # in: q_map/k_map/v_map [B,C,D,H,W]; out: y [B,C,D,H,W]
        q_map = q_map.permute(0, 2, 3, 4, 1).contiguous()
        k_map = k_map.permute(0, 2, 3, 4, 1).contiguous()
        v_map = v_map.permute(0, 2, 3, 4, 1).contiguous()

        q, shp = self._window_partition(q_map)
        k, _ = self._window_partition(k_map)
        v, _ = self._window_partition(v_map)

        q = self.norm_q(q)
        k = self.norm_kv(k)
        v = self.norm_kv(v)

        B, Nq, C = q.shape
        Nk = k.shape[1]

        q = q.view(B, Nq, self.h, self.dh).transpose(1, 2)
        k = k.view(B, Nk, self.h, self.dh).transpose(1, 2)
        v = v.view(B, Nk, self.h, self.dh).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False
        )

        out = out.transpose(1, 2).contiguous().view(B, Nq, C)
        out = self.proj_drop(self.proj(out))
        out = self._window_reverse(out, shp)
        return out.permute(0, 4, 1, 2, 3).contiguous()


class UncGuidedWindowFFTEenhance3D(nn.Module):
    def __init__(
        self,
        channels: int,
        window_size: Tuple[int, int, int] = (6, 6, 6),
        num_bases: int = 8,
        norm: str = "ortho",
        hidden: Optional[int] = None,
        eps: float = 1e-8,
        init_gamma: float = 0.0,
        rbf_min_sigma: float = 0.02,
        rbf_init_sigma: float = 0.15,
    ):
        super().__init__()
        self.c = int(channels)
        self.window_size = tuple(int(x) for x in window_size)
        self.num_bases = int(num_bases)
        self.norm = norm
        self.eps = float(eps)
        self.rbf_min_sigma = float(rbf_min_sigma)

        wd, wh, ww = self.window_size
        fd = torch.fft.fftfreq(wd, d=1.0).abs()
        fh = torch.fft.fftfreq(wh, d=1.0).abs()
        fw = torch.fft.rfftfreq(ww, d=1.0).abs()
        rr = (fd[:, None, None] ** 2 + fh[None, :, None] ** 2 + fw[None, None, :] ** 2).sqrt()
        rr = rr / (rr.max() + self.eps)
        self.register_buffer("rr", rr[None].float(), persistent=False)

        mu0 = torch.linspace(0.05, 0.95, steps=self.num_bases).clamp(1e-4, 1 - 1e-4)
        self.mu_raw = nn.Parameter(torch.log(mu0 / (1.0 - mu0)))
        sig0 = torch.full((self.num_bases,), float(rbf_init_sigma))
        self.sigma_raw = nn.Parameter(torch.log(torch.exp(sig0) - 1.0))

        in_dim = 3 * self.c + 1
        hidden = max(64, self.c) if hidden is None else int(hidden)
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.head_basis = nn.Linear(hidden, self.num_bases)
        self.head_strength = nn.Linear(hidden, 1)
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def _pad_to_window(self, x: torch.Tensor):
        _, _, D, H, W = x.shape
        wd, wh, ww = self.window_size
        pd, ph, pw = (wd - D % wd) % wd, (wh - H % wh) % wh, (ww - W % ww) % ww
        if pd or ph or pw:
            x = F.pad(x, (0, pw, 0, ph, 0, pd))
        return x, (D, H, W, D + pd, H + ph, W + pw)

    def _window_partition_padded(self, x_pad: torch.Tensor, meta):
        D, H, W, Dp, Hp, Wp = meta
        B, C, *_ = x_pad.shape
        wd, wh, ww = self.window_size
        x = x_pad.view(B, C, Dp // wd, wd, Hp // wh, wh, Wp // ww, ww)
        return x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous().view(-1, C, wd, wh, ww)

    def _window_reverse(self, windows: torch.Tensor, meta):
        D, H, W, Dp, Hp, Wp = meta
        wd, wh, ww = self.window_size
        BnW, C, *_ = windows.shape
        B = BnW // ((Dp // wd) * (Hp // wh) * (Wp // ww))
        x = windows.view(B, Dp // wd, Hp // wh, Wp // ww, C, wd, wh, ww)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(B, C, Dp, Hp, Wp)
        return x[:, :, :D, :H, :W]

    def _rbf_basis_partition_of_unity(self) -> torch.Tensor:
        rr = self.rr
        mu = torch.sigmoid(self.mu_raw).view(-1, 1, 1, 1)
        sigma = (F.softplus(self.sigma_raw) + self.rbf_min_sigma).view(-1, 1, 1, 1)
        basis = torch.exp(-0.5 * ((rr - mu) / (sigma + self.eps)) ** 2).squeeze(1)
        return basis / basis.sum(dim=0, keepdim=True).clamp_min(self.eps)

    def forward(self, low: torch.Tensor, gate_dec: torch.Tensor, unc: Optional[torch.Tensor] = None) -> torch.Tensor:
        # in: low/gate_dec [B,C,D,H,W], unc [B,1,D,H,W]|None; out: low_fft [B,C,D,H,W]
        B, C, D, H, W = low.shape
        unc = torch.zeros((B, 1, D, H, W), device=low.device, dtype=low.dtype) if unc is None else unc.clamp(0.0, 1.0)

        combo, meta = self._pad_to_window(torch.cat([low, gate_dec, unc], dim=1))
        combo_w = self._window_partition_padded(combo, meta)
        low_w, gate_w, unc_w = combo_w[:, :C].float(), combo_w[:, C:2 * C].float(), combo_w[:, 2 * C:2 * C + 1].float()

        gate_vec = gate_w.mean(dim=(2, 3, 4))
        low_mean = low_w.mean(dim=(2, 3, 4))
        low_std = low_w.var(dim=(2, 3, 4), unbiased=False).clamp_min(self.eps).sqrt()
        u_vec = unc_w.mean(dim=(2, 3, 4))
        cond = torch.cat([gate_vec, low_mean, low_std, u_vec], dim=1)

        h = self.trunk(cond)
        w = torch.softmax(self.head_basis(h), dim=1)
        strength = torch.sigmoid(self.head_strength(h)).view(-1, 1)

        basis = self._rbf_basis_partition_of_unity()
        M = torch.einsum("bk,kdhw->bdhw", w, basis).unsqueeze(1)

        X = torch.fft.rfftn(low_w, dim=(-3, -2, -1), norm=self.norm)
        scale = 1.0 + (torch.tanh(self.gamma) * strength.view(-1, 1, 1, 1, 1)) * M
        low_fft_w = torch.fft.irfftn(X * scale.to(dtype=X.dtype), s=self.window_size, dim=(-3, -2, -1), norm=self.norm)

        return self._window_reverse(low_fft_w, meta).to(dtype=low.dtype)


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
        fft_band_edges: Tuple[float, float] = (0.20, 0.45),
        hidden: int= 32
    ):
        super().__init__()
        self.num_high = num_high
        self.channels = channels

        self.fft_enh = UncGuidedWindowFFTEenhance3D(
            channels=channels,
            window_size=window_size,
            band_edges=fft_band_edges,
            hidden=hidden,
            init_gamma=0.0,
        )

        self.hf_embed = nn.Parameter(torch.zeros(num_high, channels))
        nn.init.normal_(self.hf_embed, mean=0.0, std=hf_embed_init_std)

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

        self.beta_hf = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        low: torch.Tensor,
        highs: torch.Tensor,
        gate_dec: torch.Tensor,
        unc: Optional[torch.Tensor] = None,
    ):
        # in: low [B,C,D,H,W], highs [B,C,7,D,H,W], gate_dec [B,C,D,H,W], unc [B,1,D,H,W]|None; out: low_out [B,C,D,H,W], highs_out [B,C,7,D,H,W]
        B, C, D, H, W = low.shape

        low = self.fft_enh(low, gate_dec=gate_dec, unc=unc)

        highs_list = []
        for i in range(self.num_high):
            hi = highs[:, :, i, ...] + self.hf_embed[i].view(1, C, 1, 1, 1)
            highs_list.append(hi)

        f = self.fuse_hi(torch.cat(highs_list, dim=1))

        q_low = self.q_low(low)
        k_low = self.k_low(low)
        v_low = self.v_low(low)

        k_f = self.k_f(f)
        v_f = self.v_f(f)

        delta_low = self.attn_low(q_low, k_f, v_f)
        alpha_low = self.gate_low(torch.cat([low, gate_dec], dim=1))
        low_out = low + alpha_low * delta_low

        if unc is None:
            scale = 1.0
        else:
            unc_ = unc.clamp(0.0, 1.0)
            beta = torch.sigmoid(self.beta_hf)
            scale = 1.0 + beta * unc_

        highs_out = []
        for i in range(self.num_high):
            q_hi = self.q_high[i](highs_list[i])
            delta_hi = self.attn_high[i](q_hi, k_low, v_low)
            alpha_hi = self.gate_high(torch.cat([highs_list[i], gate_dec], dim=1))
            highs_out.append(highs_list[i] + (alpha_hi * scale) * delta_hi)

        highs_out = torch.stack(highs_out, dim=2)
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
        # in: x_spa/x_fre [B,C,D,H,W]; out: y [B,C,D,H,W]
        w_spa = self.spa_mlp(self._dual_stats(x_spa))
        w_fre = self.fre_mlp(self._dual_stats(x_fre))

        x_spa_cal = x_spa * w_spa
        x_fre_cal = x_fre * w_fre

        y_spa = x_spa_cal * self.f2s(x_fre_cal)
        y_fre = x_fre_cal * self.s2f(x_spa_cal)

        y = torch.cat([y_spa, y_fre], dim=1)
        return self.proj(y)


class MFEC(nn.Module):
    def __init__(self, channels: int, normalization: NormType = "instancenorm", num_experts: int = 5):
        super().__init__()

        exp1 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            norm3d(normalization, channels),
            nn.ReLU(inplace=True),
        )
        exp2 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            norm3d(normalization, channels),
            nn.ReLU(inplace=True),
        )
        exp3 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), groups=channels, bias=False),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            norm3d(normalization, channels),
            nn.ReLU(inplace=True),
        )
        exp4 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=channels, bias=False),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            norm3d(normalization, channels),
            nn.ReLU(inplace=True),
        )
        exp5 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels, bias=False),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            norm3d(normalization, channels),
            nn.ReLU(inplace=True),
        )
        self.experts = nn.ModuleList([exp1, exp2, exp3, exp4, exp5])

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.maxpool = nn.AdaptiveMaxPool3d(1)

        hidden = max(channels // 4, 8)
        self.gate = nn.Sequential(
            nn.Conv3d(3 * channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, num_experts, kernel_size=1, bias=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv3d(num_experts * channels, channels, kernel_size=1, bias=False),
            norm3d(normalization, channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # in: x [B,C,D,H,W]; out: y [B,C,D,H,W]
        avg = self.avgpool(x)
        mx  = self.maxpool(x)
        mn  = -self.maxpool(-x)

        g = torch.cat([avg, mx, mn], dim=1)
        weights = torch.softmax(self.gate(g), dim=1)

        weighted_expert_outs = []
        for idx, expert in enumerate(self.experts):
            y = expert(x)
            y = y * weights[:, idx:idx + 1]
            weighted_expert_outs.append(y)

        y = torch.cat(weighted_expert_outs, dim=1)
        y = self.fuse(y)
        return x + y


class SkipRefinement(nn.Module):
    def __init__(self, c: int, num_heads: int, g2_channels: int, normalization: NormType = "instancenorm",
                 window_size: Tuple[int, int, int] = (6, 6, 6), attn_dropout: float = 0.0, proj_dropout: float = 0.1,hidden: int=32):
        super().__init__()
        self.dwt = DWT3D()
        self.idwt = IDWT3D()

        self.spa = MFEC(c, normalization=normalization)
        self.fre = GatedFreqCrossAttn3D(
            c, num_heads=num_heads, num_high=7, window_size=window_size, attn_dropout=attn_dropout, proj_dropout=proj_dropout,hidden=hidden
        )
        self.fuse = CrossDomainBlcok(c, norm=normalization)

        self.g2_proj = nn.Sequential(
            nn.Conv3d(g2_channels, c, kernel_size=1, bias=False),
            norm3d(normalization, c),
        )

    def forward(self, x: torch.Tensor, g2: torch.Tensor, unc: Optional[torch.Tensor] = None) -> torch.Tensor:
        # in: x [B,C,D,H,W], g2 [B,Cg,2D,2H,2W], unc [B,1,D,H,W]|None; out: y [B,C,D,H,W]
        x_spa = self.spa(x)
        low, highs = self.dwt(x)
        g2 = self.g2_proj(g2)

        low2, highs2 = self.fre(low, highs, g2, unc=unc)
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
        # in: input [B,C,D,H,W]; out: [x1,x2,x3,x4,x5]=[[B,f,D,H,W],[B,2f,D/2,H/2,W/2],[B,4f,D/4,H/4,W/4],[B,8f,D/8,H/8,W/8],[B,16f,D/16,H/16,W/16]]
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

        self.aux_skip2 = nn.Conv3d(n_filters * 4, n_classes, 1, padding=0)
        self.aux_skip1 = nn.Conv3d(n_filters * 2, n_classes, 1, padding=0)

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
            window_size=skip1_cfg.get("window_size", (6, 6, 6)),
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            hidden=32,
        )
        self.skip2 = SkipRefinement(
            n_filters*2,
            num_heads=skip2_cfg.get("num_heads", 4),
            g2_channels=n_filters*4,
            normalization=normalization,
            window_size=skip2_cfg.get("window_size", (6, 6, 6)),
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            hidden=64,
        )
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features, return_aux: bool = False):
        # in: features=[x1,x2,x3,x4,x5]; out: out_seg [B,K,D,H,W] | (out_seg [B,K,D,H,W], logits_s1 [B,K,D/2,H/2,W/2], logits_s2 [B,K,D/4,H/4,W/4])
        x1, x2, x3, x4, x5 = features

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        logits_s2 = self.aux_skip2(x6_up)
        with torch.no_grad():
            unc_s2 = predictive_entropy_from_logits(logits_s2)

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)

        logits_s1 = self.aux_skip1(x7_up)
        with torch.no_grad():
            unc_s1 = predictive_entropy_from_logits(logits_s1)

        x2 = self.skip2(x2, x6_up, unc_s2)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)

        x1 = self.skip1(x1, x7_up, unc_s1)
        x8_up = x8_up + x1

        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)

        out_seg = self.out_conv(x9)

        if return_aux:
            return out_seg, logits_s1, logits_s2
        return out_seg


class VNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=14, patch_size=96, n_filters=16,
                 normalization='instancenorm', has_dropout=False, has_residual=False,
                 input_layout: str = "NCHWD"):
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
        # in: input [B,C,H,W,D] | [B,C,D,H,W]; out: out_seg [B,K,H,W,D] | [B,K,D,H,W] | (out_seg, logits_s1, logits_s2)
        x = self._to_ncdhw(input)
        features = self.encoder(x)
        out = self.decoder(features, return_aux=return_aux)
        if return_aux:
            out_seg, logits_s1, logits_s2 = out
            out_seg  = self._to_nchwd(out_seg)
            logits_s1 = self._to_nchwd(logits_s1)
            logits_s2 = self._to_nchwd(logits_s2)
            return out_seg, logits_s1, logits_s2

        out_seg = out
        out_seg = self._to_nchwd(out_seg)
        return out_seg
