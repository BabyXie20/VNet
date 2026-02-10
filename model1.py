from __future__ import annotations
import math
from typing import Literal, Optional, Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from monai.networks.nets.swin_unetr import SwinTransformerBlock
from monai.networks.nets import swin_unetr


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
    """
    3D 离散小波变换 (Discrete Wavelet Transform) - Haar
    输入:  x: [B, C, D, H, W]
    输出:  low:   [B, C, D/2, H/2, W/2]
           highs: [B, C, 7,   D/2, H/2, W/2]
    """
    def __init__(self):
        super().__init__()
        k, stride, ksize = _build_haar_3d_kernels(device=torch.device('cpu'))
        # k: [8, 1, 2, 2, 2]
        self.register_buffer("kernels", k)  
        self.stride = stride                 
        self.ksize = ksize                   

    def forward(self, x: torch.Tensor):
        """
        x: [B, C, D, H, W]
        """
        B, C, D, H, W = x.shape
        S = self.kernels.shape[0]           # 8
        # [C*8, 1, 2,2,2]
        weight = self.kernels.repeat(C, 1, 1, 1, 1)

        # groups=C 的分组卷积
        y = F.conv3d(x, weight, stride=self.stride, padding=0, groups=C)
        # y: [B, C*8, D/2, H/2, W/2]

        y = y.view(B, C, S, *y.shape[-3:])  # [B, C, 8, D/2, H/2, W/2]
        low = y[:, :, :1, ...].squeeze(2)   # [B, C, D/2, H/2, W/2]
        highs = y[:, :, 1:, ...]            # [B, C, 7, D/2, H/2, W/2]
        return low, highs


class IDWT3D(nn.Module):
    """
    3D 逆离散小波变换 (Inverse DWT) - Haar
    输入:
        low:   [B, C, D', H', W']
        highs: [B, C, 7, D', H', W']
    输出:
        x: [B, C, 2D', 2H', 2W']
    """
    def __init__(self):
        super().__init__()
        k, stride, ksize = _build_haar_3d_kernels(device=torch.device('cpu'))
        self.register_buffer("kernels", k)   # [8, 1, 2, 2, 2]
        self.stride = stride
        self.ksize = ksize

    def forward(self, low: torch.Tensor, highs: torch.Tensor):
        """
        low:   [B, C, D', H', W']
        highs: [B, C, 7, D', H', W']
        """
        B, C, Dp, Hp, Wp = low.shape
        S = self.kernels.shape[0]           # 8

        y = torch.cat([low.unsqueeze(2), highs], dim=2)  # [B, C, 8, D', H', W']
        y = y.view(B, C * S, Dp, Hp, Wp)                 # [B, C*8, D', H', W']

        weight = self.kernels.repeat(C, 1, 1, 1, 1)      # [C*8, 1, 2, 2, 2]

        x = F.conv_transpose3d(
            y, weight,
            stride=self.stride,
            padding=0,
            output_padding=0,
            groups=C
        )
        return x  # [B, C, 2D', 2H', 2W']
    

class WindowAttention(nn.Module):
    def __init__(
        self,
        C: int,
        window_size=(4, 4, 4),
        shift_size=(0, 0, 0),       
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.window_size = tuple(window_size)
        self.shift_size = tuple(shift_size)

        self.block = SwinTransformerBlock(
            dim=C,
            num_heads=num_heads,
            window_size=self.window_size,
            shift_size=self.shift_size,       
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            use_checkpoint=False,
        )

        self._mask_cache = {}  #

    def _get_attn_mask(self, d: int, h: int, w: int, device: torch.device):
        win, sh = swin_unetr.get_window_size((d, h, w), self.window_size, self.shift_size)

        if not any(i > 0 for i in sh):
            return None

        dp = ((d + win[0] - 1) // win[0]) * win[0]
        hp = ((h + win[1] - 1) // win[1]) * win[1]
        wp = ((w + win[2] - 1) // win[2]) * win[2]

        key = (dp, hp, wp, device)
        if key not in self._mask_cache:
            self._mask_cache[key] = swin_unetr.compute_mask([dp, hp, wp], win, sh, device)
        return self._mask_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,D,H,W]
        b, c, d, h, w = x.shape
        x_cl = x.permute(0, 2, 3, 4, 1).contiguous()  # [B,D,H,W,C]

        attn_mask = self._get_attn_mask(d, h, w, x.device)
        x_cl = self.block(x_cl, mask_matrix=attn_mask)

        return x_cl.permute(0, 4, 1, 2, 3).contiguous()


class LowFreEnhanceBlock(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        c:int,
        window_size=(4, 4, 4),
        num_heads=4,
    ):
        super().__init__()
    
        ws = tuple(window_size)
        ss = tuple(i // 2 for i in ws)  

        blocks = []
        for i in range(int(n_blocks)):
            shift = (0, 0, 0) if (i % 2 == 0) else ss

            blocks.append(WindowAttention(C=c, window_size=ws, shift_size=shift, num_heads=num_heads))
            blocks.append(nn.Sequential(
                nn.Conv3d(c, c, kernel_size=3, padding=1, bias=False),
                norm3d('instancenorm', c, affine=True),
                nn.GELU(),
            ))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        x = self.proj(x)
        res = x
        for m in self.blocks:
            x = m(x)
        return x + res

    
class FreqEnhanceBlock(nn.Module):
    """
    Input : x  [B,C,D,H,W] 
    Output: x_rec [B,C,D,H,W]  
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels

        self.dwt = DWT3D()
        self.idwt = IDWT3D()

        mid = max(1, channels // reduction)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.global_mlp = nn.Sequential(
            nn.Conv3d(channels, mid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, channels, 1, bias=True),
        )
        self.local_mlp = nn.Sequential(
            nn.Conv3d(channels, mid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, channels, 1, bias=True),
        )

    def forward(self, x: torch.Tensor):
        B, C, D, H, W = x.shape
        assert C == self.channels
        assert (D % 2 == 0) and (H % 2 == 0) and (W % 2 == 0)

        low, highs = self.dwt(x)            # low: [B,C,D',H',W'], highs: [B,C,7,D',H',W']
        high = highs.sum(dim=2)             # [B,C,D',H',W']
        x_sum = low + high

        wg = self.global_mlp(self.gap(x_sum))   # [B,C,1,1,1]
        wl = self.local_mlp(x_sum)              # [B,C,D',H',W']
        w = torch.sigmoid(wg + wl)              # [B,C,D',H',W']

        highs_mod = highs * w.unsqueeze(2)      # [B,C,7,D',H',W']

        x_rec = self.idwt(low, highs_mod)       # [B,C,D,H,W]

        return x_rec


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


class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear', align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
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
