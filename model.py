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
    """1x1x1 depthwise conv: groups=C"""
    def __init__(self, channels, bias=True):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=1, groups=channels, bias=bias)

    def forward(self, x):
        return self.conv(x)


class CrossAttention3D(nn.Module):
    def __init__(self, channels, num_heads=8, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        assert channels % num_heads == 0
        self.c = channels
        self.h = num_heads
        self.dh = channels // num_heads

        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)

        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_dropout)
        self.attn_drop = attn_dropout

    def _to_tokens(self, x):
        # (B,C,D,H,W) -> (B,N,C)
        B, C, D, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return t, (D, H, W)

    def _to_map(self, t, shape):
        D, H, W = shape
        # (B,N,C) -> (B,C,D,H,W)
        return t.transpose(1, 2).reshape(t.size(0), self.c, D, H, W)

    def forward(self, q_map, k_map, v_map):
        q, shp = self._to_tokens(q_map)
        k, _   = self._to_tokens(k_map)
        v, _   = self._to_tokens(v_map)

        q = self.norm_q(q)
        k = self.norm_kv(k)
        v = self.norm_kv(v)

        B, Nq, C = q.shape
        Nk = k.shape[1]

        q = q.view(B, Nq, self.h, self.dh).transpose(1, 2)  # (B,h,Nq,dh)
        k = k.view(B, Nk, self.h, self.dh).transpose(1, 2)  # (B,h,Nk,dh)
        v = v.view(B, Nk, self.h, self.dh).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False
        )  

        out = out.transpose(1, 2).contiguous().view(B, Nq, C)  # (B,Nq,C)
        out = self.proj_drop(self.proj(out))
        return self._to_map(out, shp)


class GatedFreqCrossAttn3D(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8, num_high: int = 7):
        super().__init__()
        self.num_high = num_high

        self.q_low = DWC1x1x1(channels)
        self.k_low = DWC1x1x1(channels)
        self.v_low = DWC1x1x1(channels)

        self.q_high = nn.ModuleList([DWC1x1x1(channels) for _ in range(num_high)])
        self.k_f = DWC1x1x1(channels)
        self.v_f = DWC1x1x1(channels)

        self.fuse_hi = nn.Conv3d(num_high * channels, channels, kernel_size=1, bias=False)

        self.attn_low = CrossAttention3D(channels, num_heads=num_heads)
        self.attn_high = nn.ModuleList([CrossAttention3D(channels, num_heads=num_heads) for _ in range(num_high)])

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
    

class SpatialGate3D(nn.Module):
    def __init__(self, c: int, inter: int | None = None, norm: str = "instancenorm"):
        super().__init__()
        inter = inter or max(c // 2, 8)

        self.wx = nn.Conv3d(c, inter, 1, bias=False)
        self.wg = nn.Conv3d(c, inter, 1, bias=False)
        self.psi = nn.Conv3d(inter, 1, 1, bias=True)

        self.nx = norm3d(norm, inter)
        self.ng = norm3d(norm, inter)

        self.refine = nn.Sequential(
            nn.Conv3d(c, c, 3, padding=1, bias=False),
            norm3d(norm, c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    
        a = F.relu(self.nx(self.wx(x)) + self.ng(self.wg(g)), inplace=True)
        alpha = torch.sigmoid(self.psi(a))  
        x = x * alpha
        return self.refine(x) + x  


class AdaptiveFuse2B3D(nn.Module):
    def __init__(self, c: int, norm: str = "instancenorm"):
        super().__init__()
        self.logit = nn.Sequential(
            nn.Conv3d(3 * c, c, 1, bias=False),
            norm3d(norm, c),
            nn.ReLU(inplace=True),
            nn.Conv3d(c, 2, 1, bias=True),  # 2 logits: spa vs fre
        )
        self.out = nn.Sequential(
            nn.Conv3d(c, c, 1, bias=False),
            norm3d(norm, c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_spa: torch.Tensor, x_fre: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        logits = self.logit(torch.cat([x_spa, x_fre, g], dim=1))  # [B,2,D,H,W]
        w = torch.softmax(logits, dim=1)
        y = w[:, 0:1] * x_spa + w[:, 1:2] * x_fre
        return self.out(y)


class SkipRefinement(nn.Module):
    def __init__(self, c: int, num_heads: int, g2_channels: int, normalization: NormType = "instancenorm"):
        super().__init__()
        self.dwt = DWT3D()
        self.idwt = IDWT3D()

        self.spa = SpatialGate3D(c, norm=normalization)
        self.fre = GatedFreqCrossAttn3D(c, num_heads=num_heads, num_high=7)
        self.fuse = AdaptiveFuse2B3D(c, norm=normalization)

        self.g2_proj = nn.Sequential(
            nn.Conv3d(g2_channels, c, kernel_size=1, bias=False),
            norm3d(normalization, c),
        )

    def forward(self, x: torch.Tensor, g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
        x_spa = self.spa(x, g1)

        low, highs = self.dwt(x)

        g2 = self.g2_proj(g2)

        low2, highs2 = self.fre(low, highs, g2)
        x_fre = self.idwt(low2, highs2)

        out = self.fuse(x_spa, x_fre, g1)
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
    def __init__(self,n_classes=14, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
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

        self.skip1 = SkipRefinement(n_filters,   num_heads=1, g2_channels=n_filters*2, normalization=normalization)
        self.skip2 = SkipRefinement(n_filters*2, num_heads=2, g2_channels=n_filters*4, normalization=normalization)

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

        x2=self.skip2(x2,x7_up,x6_up)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)

        x1=self.skip1(x1,x8_up,x7_up)
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
        self.decoder = Decoder(n_classes, n_filters, normalization, has_dropout, has_residual)

    def forward_encoder(self, x):
        return self.encoder(x)

    def forward_decoder(self, feat_list):
        return self.decoder(feat_list)

    def forward(self, input):
        features = self.encoder(input)
        out_seg = self.decoder(features)
        return out_seg
    

