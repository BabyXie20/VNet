import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math
import numpy as np

def _gn_groups(C: int) -> int:
    for g in [16, 8, 4, 2]:
        if C % g == 0:
            return g
    return 1


class SEFuse2Branch3D(nn.Module):
    """
    SE-style fusion for two same-shape feature maps (B,C,D,H,W).
    - squeeze: GAP over DHW
    - excite: produce branch-wise channel weights
    - use softmax across branches for stable competition (SK-style branch selection) :contentReference[oaicite:2]{index=2}
      while still being SE-like channel recalibration :contentReference[oaicite:3]{index=3}
    """
    def __init__(self, channels: int, reduction: int = 16, use_softmax: bool = True):
        super().__init__()
        self.use_softmax = use_softmax
        hidden = max((2 * channels) // reduction, 1)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(2 * channels, hidden, 1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(hidden, 2 * channels, 1, bias=True)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a,b: (B,C,D,H,W)
        z = torch.cat([a, b], dim=1)               # (B,2C,D,H,W)
        s = self.pool(z)                           # (B,2C,1,1,1)
        w = self.fc2(self.act(self.fc1(s)))        # (B,2C,1,1,1)

        B, _, _, _, _ = w.shape
        C = a.shape[1]
        w = w.view(B, 2, C, 1, 1, 1)               # (B,2,C,1,1,1)

        if self.use_softmax:
            w = torch.softmax(w, dim=1)            # branch-competition weights
        else:
            w = torch.sigmoid(w)
            w = w / (w.sum(dim=1, keepdim=True) + 1e-6)

        return w[:, 0] * a + w[:, 1] * b


class SpatialEnhance3D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        k = kernel_size
        pad = k // 2
        g = channels  # depthwise

        self.dw_large = nn.Conv3d(channels, channels, kernel_size=k, padding=pad, groups=g, bias=False)
        self.dw_strip_d = nn.Conv3d(channels, channels, kernel_size=(k, 1, 1), padding=(pad, 0, 0), groups=g, bias=False)
        self.dw_strip_h = nn.Conv3d(channels, channels, kernel_size=(1, k, 1), padding=(0, pad, 0), groups=g, bias=False)
        self.dw_strip_w = nn.Conv3d(channels, channels, kernel_size=(1, 1, k), padding=(0, 0, pad), groups=g, bias=False)

        self.pw = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=_gn_groups(channels), num_channels=channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw_large(x) + self.dw_strip_d(x) + self.dw_strip_h(x) + self.dw_strip_w(x)
        y = self.pw(y)
        y = self.norm(y)
        y = self.act(y)
        return x + y


# -------- 3D Haar DWT filters (fixed) --------
def get_wav3d(in_channels: int, pool: bool = True, stride: Tuple[int, int, int] = (2, 2, 2)):
    L = (1 / np.sqrt(2)) * np.array([1.0, 1.0], dtype=np.float32)
    H = (1 / np.sqrt(2)) * np.array([-1.0, 1.0], dtype=np.float32)

    def outer3(a, b, c):
        return np.einsum("i,j,k->ijk", a, b, c)

    filters = [
        outer3(L, L, L), outer3(L, L, H), outer3(L, H, L), outer3(L, H, H),
        outer3(H, L, L), outer3(H, L, H), outer3(H, H, L), outer3(H, H, H),
    ]
    filters = [torch.from_numpy(f).unsqueeze(0).unsqueeze(0) for f in filters]  # (1,1,2,2,2)
    net = nn.Conv3d if pool else nn.ConvTranspose3d

    convs = []
    for _ in range(8):
        convs.append(
            net(in_channels, in_channels, kernel_size=2, stride=stride, padding=0, bias=False, groups=in_channels)
        )

    for conv, f in zip(convs, filters):
        conv.weight.requires_grad_(False)
        conv.weight.data = f.float().expand(in_channels, 1, 2, 2, 2).clone()

    return convs


class WavePool3D(nn.Module):
    def __init__(self, in_channels: int, stride: Tuple[int, int, int] = (2, 2, 2)):
        super().__init__()
        convs = get_wav3d(in_channels, pool=True, stride=stride)
        (self.LLL, self.LLH, self.LHL, self.LHH,
         self.HLL, self.HLH, self.HHL, self.HHH) = convs

    def forward(self, x: torch.Tensor):
        return (self.LLL(x), self.LLH(x), self.LHL(x), self.LHH(x),
                self.HLL(x), self.HLH(x), self.HHL(x), self.HHH(x))


class MSCM(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.local_attn = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, d_model))
        self.global_attn = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, d_model))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = x.mean(dim=1, keepdim=True)
        w = self.local_attn(x) + self.global_attn(g)
        return self.sigmoid(w)


class WaveletMSCMFusion3D(nn.Module):
    """
    Frequency-only KV pre.
    Return:
      fre_tokens: (B, N', C)
      (Dp,Hp,Wp): pooled size
    """
    def __init__(self, channels: int, stride=(2, 2, 2)):
        super().__init__()
        self.dwt = WavePool3D(channels, stride=stride)
        self.mscm = MSCM(d_model=channels)

    def forward(self, feat_map: torch.Tensor):
        (LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH) = self.dwt(feat_map)
        low = LLL
        high = LLH + LHL + LHH + HLL + HLH + HHL + HHH

        B, C, Dp, Hp, Wp = low.shape
        low_t = low.flatten(2).transpose(1, 2)
        high_t = high.flatten(2).transpose(1, 2)

        w = self.mscm(high_t + low_t)
        fre = (w * high_t) + low_t
        return fre, (Dp, Hp, Wp)


class PositionEmbeddingSine3D(nn.Module):
    def __init__(self, d_model: int, num_pos_feats: int = 32, temperature: int = 10000, normalize: bool = True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.proj = nn.Conv3d(6 * num_pos_feats, d_model, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, D, H, W = x.shape
        device = x.device

        z = torch.arange(D, device=device).float()
        y = torch.arange(H, device=device).float()
        x_ = torch.arange(W, device=device).float()
        zz, yy, xx = torch.meshgrid(z, y, x_, indexing="ij")

        if self.normalize:
            eps = 1e-6
            zz = zz / (D - 1 + eps) * 2 * math.pi
            yy = yy / (H - 1 + eps) * 2 * math.pi
            xx = xx / (W - 1 + eps) * 2 * math.pi

        dim_t = torch.arange(self.num_pos_feats, device=device).float()
        dim_t = self.temperature ** (dim_t / self.num_pos_feats)

        def embed(pos):
            pos = pos[..., None] / dim_t
            out = torch.stack((pos.sin(), pos.cos()), dim=-1)
            return out.flatten(-2)

        ez = embed(zz)
        ey = embed(yy)
        ex = embed(xx)

        pos = torch.cat([ez, ey, ex], dim=-1)  # (D,H,W,6F)
        pos = pos.permute(3, 0, 1, 2).unsqueeze(0).repeat(B, 1, 1, 1, 1)
        return self.proj(pos)


def _pad_to_multiple_3d(x: torch.Tensor, ws: Tuple[int, int, int]):
    B, C, D, H, W = x.shape
    wd, wh, ww = ws
    pd = (wd - D % wd) % wd
    ph = (wh - H % wh) % wh
    pw = (ww - W % ww) % ww
    if pd or ph or pw:
        x = F.pad(x, (0, pw, 0, ph, 0, pd))
    return x, (pd, ph, pw)


def _crop_pad_3d(x: torch.Tensor, pads: Tuple[int, int, int]):
    pd, ph, pw = pads
    if pd: x = x[:, :, :-pd, :, :]
    if ph: x = x[:, :, :, :-ph, :]
    if pw: x = x[:, :, :, :, :-pw]
    return x


def _window_partition_3d(x: torch.Tensor, ws: Tuple[int, int, int]):
    B, C, D, H, W = x.shape
    wd, wh, ww = ws
    assert D % wd == 0 and H % wh == 0 and W % ww == 0
    Dg, Hg, Wg = D // wd, H // wh, W // ww
    x = x.view(B, C, Dg, wd, Hg, wh, Wg, ww)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    win = x.view(B * Dg * Hg * Wg, C, wd, wh, ww)
    return win, (Dg, Hg, Wg)


def _window_reverse_3d(win: torch.Tensor, grid: Tuple[int, int, int], ws: Tuple[int, int, int], B: int):
    Dg, Hg, Wg = grid
    wd, wh, ww = ws
    C = win.shape[1]
    x = win.view(B, Dg, Hg, Wg, C, wd, wh, ww)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    return x.view(B, C, Dg * wd, Hg * wh, Wg * ww)


def _match_dhw_by_pad_or_crop(x: torch.Tensor, target: Tuple[int, int, int]):
    B, C, D, H, W = x.shape
    tD, tH, tW = target
    x = x[:, :, :min(D, tD), :min(H, tH), :min(W, tW)]
    _, _, D2, H2, W2 = x.shape
    pd, ph, pw = max(0, tD - D2), max(0, tH - H2), max(0, tW - W2)
    if pd or ph or pw:
        x = F.pad(x, (0, pw, 0, ph, 0, pd))
    return x


class WaveletWindowCrossAttention3D(nn.Module):
    """
    Windowed cross-attn:
      Q = query_map windows
      KV = frequency-only (wavelet+MSCM) from memory_map
    """
    def __init__(self, d_model: int, nhead: int,
                 kv_stride=(2, 2, 2),
                 window_size=(6, 12, 12),
                 dropout: float = 0.0):
        super().__init__()
        self.kv_stride = kv_stride
        self.window_size = window_size

        self.kv_pre = WaveletMSCMFusion3D(d_model, stride=kv_stride)
        self.pos3d = PositionEmbeddingSine3D(d_model)

        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, query_map: torch.Tensor, memory_map: torch.Tensor) -> torch.Tensor:
        """
        query_map:  (B,C,D,H,W)
        memory_map: (B,C,D,H,W)  (here it is already SpatialEnhance3D-enhanced skip, but KV uses only freq)
        """
        B, C, D, H, W = query_map.shape
        ws = self.window_size
        sd, sh, sw = self.kv_stride
        wd, wh, ww = ws

        # pad to window multiple
        q_pad, pads = _pad_to_multiple_3d(query_map, ws)
        m_pad, _ = _pad_to_multiple_3d(memory_map, ws)
        Dp, Hp, Wp = q_pad.shape[2:]

        # q pos
        qpos = self.pos3d(torch.zeros((B, 1, Dp, Hp, Wp), device=q_pad.device, dtype=q_pad.dtype))

        # freq-only KV tokens -> map
        fre_tokens, (Dk, Hk, Wk) = self.kv_pre(m_pad)              # (B,N',C)
        kv_map = fre_tokens.transpose(1, 2).contiguous().view(B, C, Dk, Hk, Wk)

        # kv pos
        kpos = self.pos3d(torch.zeros((B, 1, Dk, Hk, Wk), device=q_pad.device, dtype=q_pad.dtype))

        # window alignment between Q and KV
        assert wd % sd == 0 and wh % sh == 0 and ww % sw == 0, "window_size must be divisible by kv_stride"
        k_ws = (wd // sd, wh // sh, ww // sw)

        # number of query windows
        nD, nH, nW = Dp // wd, Hp // wh, Wp // ww
        target_k = (nD * k_ws[0], nH * k_ws[1], nW * k_ws[2])
        kv_map = _match_dhw_by_pad_or_crop(kv_map, target_k)
        kpos = _match_dhw_by_pad_or_crop(kpos, target_k)

        # partition windows
        q_win, q_grid = _window_partition_3d(q_pad, ws)          # (BNw,C,wd,wh,ww)
        qp_win, _ = _window_partition_3d(qpos, ws)

        k_win, k_grid = _window_partition_3d(kv_map, k_ws)       # (BNw,C,kwd,kwh,kww)
        kp_win, _ = _window_partition_3d(kpos, k_ws)

        assert q_grid == k_grid, f"Q grid {q_grid} != K grid {k_grid}"

        BNw = q_win.shape[0]
        Lq = wd * wh * ww
        Lk = k_ws[0] * k_ws[1] * k_ws[2]

        # to sequences (L, N, E) for MHA :contentReference[oaicite:6]{index=6}
        q_seq = q_win.flatten(2).permute(2, 0, 1).contiguous()     # (Lq, BNw, C)
        qp_seq = qp_win.flatten(2).permute(2, 0, 1).contiguous()

        k_seq = k_win.flatten(2).permute(2, 0, 1).contiguous()     # (Lk, BNw, C)
        kp_seq = kp_win.flatten(2).permute(2, 0, 1).contiguous()

        attn_out = self.attn(query=q_seq + qp_seq, key=k_seq + kp_seq, value=k_seq, need_weights=False)[0]
        out_seq = self.norm(q_seq + self.drop(attn_out))           # (Lq, BNw, C)

        # back to map
        out_win = out_seq.permute(1, 2, 0).contiguous().view(BNw, C, wd, wh, ww)
        out_pad = _window_reverse_3d(out_win, q_grid, ws, B)       # (B,C,Dp,Hp,Wp)
        out = _crop_pad_3d(out_pad, pads)                          # (B,C,D,H,W)
        return out


class WaveletWindowCrossFusion3D(nn.Module):
    """
    Fuse:
      y_attn = WaveletWindowCrossAttention3D(query, skip_enhanced)   (KV uses only frequency)
      y_spa  = query + skip_enhanced                                (pure spatial skip fusion)
      out    = SEFuse2Branch3D(y_attn, y_spa)
    """
    def __init__(self, d_model: int, nhead: int,
                 kv_stride=(2, 2, 2),
                 window_size=(6, 12, 12),
                 dropout=0.0,
                 se_reduction: int = 16):
        super().__init__()
        self.cross = WaveletWindowCrossAttention3D(
            d_model=d_model, nhead=nhead, kv_stride=kv_stride, window_size=window_size, dropout=dropout
        )
        self.se_fuse = SEFuse2Branch3D(d_model, reduction=se_reduction, use_softmax=True)

    def forward(self, query_map: torch.Tensor, skip_enhanced: torch.Tensor) -> torch.Tensor:
        y_attn = self.cross(query_map, skip_enhanced)
        y_spa = query_map + skip_enhanced
        return self.se_fuse(y_attn, y_spa)


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
    

class Encoder(nn.Module):
    def __init__(self, n_channels=1, n_classes=14, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super().__init__()
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

        self.se1 = SpatialEnhance3D(n_filters * 1, kernel_size=5)
        self.se2 = SpatialEnhance3D(n_filters * 2, kernel_size=7)
        self.se3 = SpatialEnhance3D(n_filters * 4, kernel_size=9)
        self.se4 = SpatialEnhance3D(n_filters * 8, kernel_size=11)

    def forward(self, input):
        x1 = self.se1(self.block_one(input))
        x1_dw = self.block_one_dw(x1)

        x2 = self.se2(self.block_two(x1_dw))
        x2_dw = self.block_two_dw(x2)

        x3 = self.se3(self.block_three(x2_dw))
        x3_dw = self.block_three_dw(x3)

        x4 = self.se4(self.block_four(x3_dw))
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        if self.has_dropout:
            x5 = self.dropout(x5)

        return [x1, x2, x3, x4, x5]


class Decoder(nn.Module):
    def __init__(self, n_classes=14, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super().__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        upsampling = UpsamplingDeconvBlock  ## using transposed convolution

        self.block_five_up = upsampling(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = upsampling(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = upsampling(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = upsampling(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

        self.fuse_x4 = WaveletWindowCrossFusion3D(
            d_model=n_filters * 8, nhead=8, kv_stride=(2,2,2), window_size=(4,4,4), dropout=0.0, se_reduction=16
        )
        self.fuse_x3 = WaveletWindowCrossFusion3D(
            d_model=n_filters * 4, nhead=8, kv_stride=(2,2,2), window_size=(6,6,6), dropout=0.0, se_reduction=16
        )
        self.fuse_x2 = WaveletWindowCrossFusion3D(
            d_model=n_filters * 2, nhead=8, kv_stride=(2,2,2), window_size=(6,12,12), dropout=0.0, se_reduction=16
        )
        self.fuse_x1 = WaveletWindowCrossFusion3D(
            d_model=n_filters * 1, nhead=4, kv_stride=(2,2,2), window_size=(6,12,12), dropout=0.0, se_reduction=16
        )

    def forward(self, features):
        x1, x2, x3, x4, x5 = features

        x5_up = self.block_five_up(x5)
        x5_up = self.fuse_x4(x5_up, x4)

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = self.fuse_x3(x6_up, x3)

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = self.fuse_x2(x7_up, x2)

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = self.fuse_x1(x8_up, x1)

        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)
        return out_seg


class VNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=14, patch_size=96, n_filters=16,
                 normalization='instancenorm', has_dropout=False, has_residual=False):
        super(VNet, self).__init__()
        self.num_classes = n_classes
        self.encoder = Encoder(
            n_channels=n_channels,
            n_classes=n_classes,
            n_filters=n_filters,
            normalization=normalization,
            has_dropout=has_dropout,
            has_residual=has_residual
        )
        self.decoder = Decoder(
            n_classes=n_classes,
            n_filters=n_filters,
            normalization=normalization,
            has_dropout=has_dropout,
            has_residual=has_residual
        )

    def forward(self, input):
        features = self.encoder(input)
        out_seg = self.decoder(features)
        return out_seg

