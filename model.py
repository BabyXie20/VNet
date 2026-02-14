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


class LayerNormChannelFirst(nn.Module):
    """LayerNorm on channel dimension for (B, C, D, H, W)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        return x.permute(0, 4, 1, 2, 3).contiguous()


class Mlp3d(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Conv3d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Conv3d(hidden_dim, dim, kernel_size=1)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def window_partition(x: torch.Tensor, window_size: Tuple[int, int, int]) -> torch.Tensor:
    # x: (B, D, H, W, C)
    b, d, h, w, c = x.shape
    wd, wh, ww = window_size
    x = x.view(b, d // wd, wd, h // wh, wh, w // ww, ww, c)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return windows.view(-1, wd * wh * ww, c)


def window_reverse(windows: torch.Tensor, window_size: Tuple[int, int, int], b: int, d: int, h: int, w: int, c: int) -> torch.Tensor:
    wd, wh, ww = window_size
    x = windows.view(b, d // wd, h // wh, w // ww, wd, wh, ww, c)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(b, d, h, w, c)


def compute_mask(d: int, h: int, w: int, window_size: Tuple[int, int, int], shift_size: Tuple[int, int, int], device: torch.device) -> torch.Tensor:
    wd, wh, ww = window_size
    sd, sh, sw = shift_size
    mask = torch.zeros((1, d, h, w, 1), device=device)
    d_slices = (slice(0, -wd), slice(-wd, -sd), slice(-sd, None))
    h_slices = (slice(0, -wh), slice(-wh, -sh), slice(-sh, None))
    w_slices = (slice(0, -ww), slice(-ww, -sw), slice(-sw, None))
    cnt = 0
    for ds in d_slices:
        for hs in h_slices:
            for ws in w_slices:
                mask[:, ds, hs, ws, :] = cnt
                cnt += 1
    mask_windows = window_partition(mask, window_size).squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class RelPosBias3D(nn.Module):
    def __init__(self, window_size: Tuple[int, int, int], num_heads: int):
        super().__init__()
        self.window_size = window_size
        wd, wh, ww = window_size
        size = (2 * wd - 1) * (2 * wh - 1) * (2 * ww - 1)
        self.table = nn.Parameter(torch.zeros(size, num_heads))

        coords = torch.stack(torch.meshgrid(
            torch.arange(wd), torch.arange(wh), torch.arange(ww), indexing="ij"
        ))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += wd - 1
        relative_coords[:, :, 1] += wh - 1
        relative_coords[:, :, 2] += ww - 1
        relative_coords[:, :, 0] *= (2 * wh - 1) * (2 * ww - 1)
        relative_coords[:, :, 1] *= (2 * ww - 1)
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.table, std=0.02)

    def forward(self) -> torch.Tensor:
        wd, wh, ww = self.window_size
        n = wd * wh * ww
        bias = self.table[self.relative_position_index.view(-1)].view(n, n, -1)
        return bias.permute(2, 0, 1).contiguous()


class CrossWindowAttention3D(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, window_size: Tuple[int, int, int] = (6, 6, 6),
                 qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0, use_rel_pos_bias: bool = True):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Conv3d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.k = nn.Conv3d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.v = nn.Conv3d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.proj = nn.Conv3d(dim, dim, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rel_pos = RelPosBias3D(window_size, num_heads) if use_rel_pos_bias else None

    def _pad_to_window(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        _, _, d, h, w = x.shape
        wd, wh, ww = self.window_size
        pd = (wd - d % wd) % wd
        ph = (wh - h % wh) % wh
        pw = (ww - w % ww) % ww
        if pd or ph or pw:
            x = F.pad(x, (0, pw, 0, ph, 0, pd))
        return x, (pd, ph, pw)

    def forward(self, q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor, use_shift: bool = False) -> torch.Tensor:
        b, c, d, h, w = q_in.shape
        q = self.q(q_in)
        k = self.k(k_in)
        v = self.v(v_in)

        q, pad = self._pad_to_window(q)
        k, _ = self._pad_to_window(k)
        v, _ = self._pad_to_window(v)
        pd, ph, pw = pad
        dp, hp, wp = d + pd, h + ph, w + pw

        shift_size = tuple(ws // 2 for ws in self.window_size) if use_shift else (0, 0, 0)
        if use_shift:
            q = torch.roll(q, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(2, 3, 4))
            k = torch.roll(k, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(2, 3, 4))
            v = torch.roll(v, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(2, 3, 4))

        q = window_partition(q.permute(0, 2, 3, 4, 1).contiguous(), self.window_size)
        k = window_partition(k.permute(0, 2, 3, 4, 1).contiguous(), self.window_size)
        v = window_partition(v.permute(0, 2, 3, 4, 1).contiguous(), self.window_size)

        nw, n, _ = q.shape
        q = q.view(nw, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(nw, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(nw, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.rel_pos is not None:
            attn = attn + self.rel_pos().unsqueeze(0)

        if use_shift:
            mask = compute_mask(dp, hp, wp, self.window_size, shift_size, q_in.device)
            nW = mask.shape[0]
            attn = attn.view(b, nW, self.num_heads, n, n)
            attn = attn + mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(nw, n, c)

        out = window_reverse(out, self.window_size, b, dp, hp, wp, c)
        if use_shift:
            out = torch.roll(out, shifts=shift_size, dims=(1, 2, 3))
        out = out[:, :d, :h, :w, :].permute(0, 4, 1, 2, 3).contiguous()
        out = self.proj_drop(self.proj(out))
        return out


class FreCrossAttBlock(nn.Module):
    """Frequency-domain cross-attention skip fusion block for 3D VNet."""

    def __init__(self, dim: int, dwt3d: nn.Module, idwt3d: nn.Module, num_heads: int = 4,
                 window_size: Tuple[int, int, int] = (6, 6, 6), use_shift: bool = False,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.use_shift = use_shift
        self.dwt3d = dwt3d
        self.idwt3d = idwt3d

        hidden = int(dim * mlp_ratio)
        self.q_proj = nn.ModuleList([nn.Conv3d(dim, dim, kernel_size=1) for _ in range(7)])
        self.q_norm = nn.ModuleList([LayerNormChannelFirst(dim) for _ in range(7)])
        self.q_mlp = nn.ModuleList([Mlp3d(dim, hidden_dim=hidden, dropout=dropout) for _ in range(7)])

        self.k_proj = nn.Conv3d(dim, dim, kernel_size=1)
        self.k_norm = LayerNormChannelFirst(dim)
        self.k_mlp = Mlp3d(dim, hidden_dim=hidden, dropout=dropout)

        self.v_fuse = nn.Conv3d(3 * dim, dim, kernel_size=1)
        self.v_norm = LayerNormChannelFirst(dim)
        self.v_mlp = Mlp3d(dim, hidden_dim=hidden, dropout=dropout)

        self.cross_attn = CrossWindowAttention3D(dim=dim, num_heads=num_heads, window_size=window_size, proj_drop=dropout)
        self.rec_norm = nn.ModuleList([LayerNormChannelFirst(dim) for _ in range(7)])
        self.rec_mlp = nn.ModuleList([Mlp3d(dim, hidden_dim=hidden, dropout=dropout) for _ in range(7)])
        self.low_norm = LayerNormChannelFirst(dim)
        self.low_mlp = Mlp3d(dim, hidden_dim=hidden, dropout=dropout)

    @staticmethod
    def _residual_ffn(x: torch.Tensor, norm: nn.Module, mlp: nn.Module) -> torch.Tensor:
        return x + mlp(norm(x))

    def _split_subbands(self, e_l: torch.Tensor) -> list[torch.Tensor]:
        parts = self.dwt3d(e_l)
        if isinstance(parts, (tuple, list)) and len(parts) == 2:
            low, highs = parts
            return [low] + [highs[:, :, i, ...] for i in range(highs.shape[2])]
        if isinstance(parts, (tuple, list)) and len(parts) == 8:
            return list(parts)
        raise RuntimeError("DWT3D output format must be (low, highs[7]) or 8 sub-bands.")

    def _merge_subbands(self, subbands: list[torch.Tensor]) -> torch.Tensor:
        low = subbands[0]
        highs = torch.stack(subbands[1:], dim=2)
        try:
            return self.idwt3d(low, highs)
        except TypeError:
            return self.idwt3d(subbands)

    def forward(self, e_l: torch.Tensor, u_l: torch.Tensor, use_shift: Optional[bool] = None) -> torch.Tensor:
        shift = self.use_shift if use_shift is None else use_shift
        subbands = self._split_subbands(e_l)
        p_lll = subbands[0]
        highs = subbands[1:]

        k = self._residual_ffn(self.k_proj(p_lll), self.k_norm, self.k_mlp)
        v_in = torch.cat([u_l, p_lll], dim=1)
        v = self._residual_ffn(self.v_fuse(v_in), self.v_norm, self.v_mlp)

        high_hat = []
        for i, p_h in enumerate(highs):
            q = self._residual_ffn(self.q_proj[i](p_h), self.q_norm[i], self.q_mlp[i])
            z = self.cross_attn(q, k, v, use_shift=shift)
            r = self.rec_mlp[i](self.rec_norm[i](z))
            high_hat.append(p_h + r)

        p_lll_hat = p_lll + self.low_mlp(self.low_norm(v))
        return self._merge_subbands([p_lll_hat] + high_hat)
    

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


class DWC1x1x1(nn.Module):
    def __init__(self, channels, bias=True):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=1, groups=channels, bias=bias)

    def forward(self, x):
        return self.conv(x)


class CrossAttention3D(nn.Module):
    def __init__(self, channels, num_heads=8, window_size: Tuple[int, int, int] = (6, 6, 6), attn_dropout=0.0, proj_dropout=0.0):
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

    def _window_partition(self, x: torch.Tensor):
        # x: [B, D, H, W, C]
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
        q_map = q_map.permute(0, 2, 3, 4, 1).contiguous()  # [B,D,H,W,C]
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
        out = self._window_reverse(out, shp)
        return out.permute(0, 4, 1, 2, 3).contiguous()


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


class GatedFreqCrossAttn3D(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8, num_high: int = 7, window_size: Tuple[int, int, int] = (6, 6, 6), attn_dropout: float = 0.0, proj_dropout: float = 0.1):
        super().__init__()
        self.num_high = num_high

        self.q_low = DWC1x1x1(channels)
        self.k_low = DWC1x1x1(channels)
        self.v_low = DWC1x1x1(channels)

        self.q_high = nn.ModuleList([DWC1x1x1(channels) for _ in range(num_high)])
        self.k_f = DWC1x1x1(channels)
        self.v_f = DWC1x1x1(channels)

        self.fuse_hi = nn.Conv3d(num_high * channels, channels, kernel_size=1, bias=False)

        self.attn_low = CrossAttention3D(channels, num_heads=num_heads, window_size=window_size, attn_dropout=attn_dropout, proj_dropout=proj_dropout)
        self.attn_high = nn.ModuleList([
            CrossAttention3D(channels, num_heads=num_heads, window_size=window_size, attn_dropout=attn_dropout, proj_dropout=proj_dropout)
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
    def __init__(self, channels: int, normalization: NormType = "instancenorm", num_experts: int = 5):
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
    def __init__(self, c: int, g2_channels: int, window_size: Tuple[int, int, int] = (6, 8, 8),
                 use_shift: bool = True, dropout: float = 0.0):
        super().__init__()
        if g2_channels != 2 * c:
            raise ValueError(f"Expected g2_channels == 2*c, got {g2_channels} and c={c}.")
        if c == 16:
            num_heads = 2
        elif c == 32:
            num_heads = 4
        else:
            raise ValueError(f"SkipRefinement expects c in {{16, 32}}, but got {c}.")

        self.fre = FreCrossAttBlock(
            dim=c,
            dwt3d=DWT3D(),
            idwt3d=IDWT3D(),
            num_heads=num_heads,
            window_size=window_size,
            use_shift=use_shift,
            dropout=dropout,
        )

    def forward(self, enc_skip: torch.Tensor, dec_half: torch.Tensor) -> torch.Tensor:
        if dec_half.shape[1] == 2 * enc_skip.shape[1]:
            dec_in = dec_half
        else:
            raise ValueError(
                f"Decoder half-scale feature channels must be 2x skip channels, got {dec_half.shape[1]} vs {enc_skip.shape[1]}."
            )
        return self.fre(enc_skip, dec_in)

    
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

        # patch_size=96, n_filters=16: use FreCrossAttBlock with fixed window_size and heads.
        self.skip1 = SkipRefinement(
            n_filters,
            g2_channels=n_filters * 2,
            window_size=(6, 8, 8),
            use_shift=True,
        )
        self.skip2 = SkipRefinement(
            n_filters * 2,
            g2_channels=n_filters * 4,
            window_size=(6, 8, 8),
            use_shift=True,
        )

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
    def __init__(self, n_channels=1, n_classes=14, patch_size=96, n_filters=16, normalization='instancenorm', has_dropout=False, has_residual=False):
        super(VNet, self).__init__()
        self.num_classes = n_classes
        self.encoder = Encoder(n_channels,n_filters, normalization, has_dropout, has_residual)
        attn_cfg = recommended_window_attn_config(patch_size)
        self.decoder = Decoder(n_classes, n_filters, normalization, has_dropout, has_residual, attn_cfg=attn_cfg)

    def forward_encoder(self, x):
        return self.encoder(x)

    def forward_decoder(self, feat_list):
        return self.decoder(feat_list)

    def forward(self, input):
        features = self.encoder(input)
        out_seg = self.decoder(features)
        return out_seg
    
