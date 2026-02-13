from __future__ import annotations
from typing import Literal, Optional
from typing import Sequence, Tuple
import torch
from torch import nn
from monai.networks.nets.swin_unetr import SwinTransformerBlock
from monai.networks.nets import swin_unetr

if __package__:
    from .DWT_IDWT_layer import DWT_3D, IDWT_3D
else:
    from DWT_IDWT_layer import DWT_3D, IDWT_3D


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


class WindowAttention(nn.Module):
    def __init__(
        self,
        C: int,
        window_size=(4, 4, 4),
        shift_size=(0, 0, 0),       
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
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


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or (not self.training):
            return x
        keep_prob = 1.0 - self.drop_prob
        # broadcast along non-batch dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rnd = x.new_empty(shape).bernoulli_(keep_prob)
        return x * rnd / keep_prob


class LayerScale3D(nn.Module):
    def __init__(self, channels: int, init_value: float = 1e-4):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1, 1) * init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class AnisoInceptionConv3D(nn.Module):
    def __init__(self, c: int, dilation: int = 1):
        super().__init__()
        d = dilation
        self.b1 = nn.Conv3d(
            c, c, kernel_size=(1, 5, 5),
            padding=(0, 2*d, 2*d),
            dilation=(1, d, d), bias=False
        )
        self.b2 = nn.Conv3d(
            c, c,
            kernel_size=(3, 1, 1),
            padding=(dilation, 0, 0),
            dilation=(dilation, 1, 1),
            groups=c,   
            bias=False
        )
        self.b3 = nn.Conv3d(
            c, c, kernel_size=(1, 1, 5),
            padding=(0, 0, 2*d),
            dilation=(1, 1, d), bias=False
        )
        self.b4 = nn.Conv3d(
            c, c, kernel_size=(1, 5, 1),
            padding=(0, 2*d, 0),
            dilation=(1, d, 1), bias=False
        )

        self.fuse = nn.Conv3d(4 * c, c, kernel_size=1, bias=False)
        self.norm = norm3d('instancenorm', c, affine=True)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        y = self.fuse(y)
        y = self.act(self.norm(y))
        return y


class MultiShapeWindowAttention(nn.Module):
    def __init__(
        self,
        c: int,
        window_sizes: Sequence[Tuple[int, int, int]],
        shift: bool,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        gate_reduction: int = 8,
    ):
        super().__init__()
        self.window_sizes = [tuple(ws) for ws in window_sizes]
        self.shift = bool(shift)
        self.num_branches = len(self.window_sizes)

        branches = []
        for ws in self.window_sizes:
            if self.shift:
                ss = tuple(max(0, w // 2) for w in ws)
            else:
                ss = (0, 0, 0)
            branches.append(
                WindowAttention(
                    C=c,
                    window_size=ws,
                    shift_size=ss,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                )
            )
        self.branches = nn.ModuleList(branches)

        # gating: x -> [B,K,1,1,1] then softmax over K
        hidden = max(1, c // gate_reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(c, hidden, 1, bias=True),
            nn.GELU(),
            nn.Conv3d(hidden, self.num_branches, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,D,H,W]
        outs = [b(x) for b in self.branches]  # list of [B,C,D,H,W]
        w = self.gate(x).softmax(dim=1)       # [B,K,1,1,1]
        y = 0
        for k, o in enumerate(outs):
            y = y + o * w[:, k:k+1]
        return y


class LowFreEnhanceBlock(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        c: int,
        window_size: Tuple[int, int, int] = (4, 4, 4),
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-4,
        conv_dilation: int = 1,
        window_shapes: Optional[Sequence[Tuple[int, int, int]]] = None,
    ):
        super().__init__()
        self.n_blocks = int(n_blocks)
        self.c = int(c)

        base_ws = tuple(window_size)

        def _clamp_ws(ws: Tuple[int, int, int], cap: int = 12) -> Tuple[int, int, int]:
            # avoid too large windows (esp. for low-res subband)
            return tuple(max(1, min(int(v), cap)) for v in ws)

        # default multi-shape set: iso + strip variants
        if window_shapes is None:
            # iso
            iso = base_ws
            # strip-ish variants (favor long-range on two axes, compact on one axis)
            strip1 = (max(1, base_ws[0] // 2), base_ws[1] * 2, base_ws[2] * 2)
            strip2 = (base_ws[0] * 2, max(1, base_ws[1] // 2), base_ws[2] * 2)
            strip3 = (base_ws[0] * 2, base_ws[1] * 2, max(1, base_ws[2] // 2))
            window_shapes = [_clamp_ws(iso), _clamp_ws(strip1), _clamp_ws(strip2), _clamp_ws(strip3)]
        else:
            window_shapes = [tuple(ws) for ws in window_shapes]

        self.window_shapes = window_shapes

        # distribute droppath across layers (optional but usually better)
        if self.n_blocks > 1:
            dp_rates = torch.linspace(0, drop_path, steps=self.n_blocks).tolist()
        else:
            dp_rates = [drop_path]

        self.attn_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        self.attn_ls = nn.ModuleList()
        self.conv_ls = nn.ModuleList()
        self.attn_dp = nn.ModuleList()
        self.conv_dp = nn.ModuleList()

        for i in range(self.n_blocks):
            use_shift = (i % 2 == 1)

            self.attn_layers.append(
                MultiShapeWindowAttention(
                    c=c,
                    window_sizes=self.window_shapes,
                    shift=use_shift,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=0.0,  # keep Swin internal droppath off; we handle outer droppath for stability
                )
            )
            self.conv_layers.append(AnisoInceptionConv3D(c=c, dilation=conv_dilation))

            # LayerScale + DropPath per sublayer
            self.attn_ls.append(LayerScale3D(c, init_value=layer_scale_init))
            self.conv_ls.append(LayerScale3D(c, init_value=layer_scale_init))
            self.attn_dp.append(DropPath(dp_rates[i]))
            self.conv_dp.append(DropPath(dp_rates[i]))

        # optional final residual scale (kept simple)
        self.final_ls = LayerScale3D(c, init_value=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,C,D',H',W'] (low subband)
        """
        res0 = x

        for attn, conv, ls_a, ls_c, dp_a, dp_c in zip(
            self.attn_layers, self.conv_layers, self.attn_ls, self.conv_ls, self.attn_dp, self.conv_dp
        ):
            # --- attention sublayer (delta residual for stability) ---
            y = attn(x)
            x = x + dp_a(ls_a(y - x))

            # --- conv sublayer (residual) ---
            y = conv(x)
            x = x + dp_c(ls_c(y))

        # global skip (keeps low-frequency identity path)
        x = self.final_ls(x) + res0
        return x


class FreqEnhanceBlock(nn.Module):
    def __init__(self, channels: int,n_blocks: int,window_size=(4,4,4),num_heads=4,reduction: int = 16):
        super().__init__()
        self.channels = channels

        self.dwt = DWT_3D('haar')
        self.idwt = IDWT_3D('haar')
        self.lowtransform= LowFreEnhanceBlock(n_blocks, channels, window_size,num_heads)
        self.subband_att = nn.Sequential(
            nn.Conv3d(channels * 7, max(1, channels // reduction), kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(max(1, channels // reduction), 7, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        B, C, D, H, W = x.shape
        assert C == self.channels
        assert (D % 2 == 0) and (H % 2 == 0) and (W % 2 == 0)

        low, llh, lhl, lhh, hll, hlh, hhl, hhh = self.dwt(x)
        highs = torch.cat([
            llh.unsqueeze(2), lhl.unsqueeze(2), lhh.unsqueeze(2),
            hll.unsqueeze(2), hlh.unsqueeze(2), hhl.unsqueeze(2), hhh.unsqueeze(2)
        ], dim=2)

        high_cat = highs.view(B, C * 7, *highs.shape[-3:])
        highs_att = highs * self.subband_att(high_cat).unsqueeze(1)
        high_agg = highs_att.sum(dim=2)

        low=self.lowtransform(low)
        x_sum = low + high_agg

        zeros = torch.zeros_like(low)
        low_rec = self.idwt(
            low,
            zeros, zeros, zeros,
            zeros, zeros, zeros, zeros,
        )
        high_rec = self.idwt(
            zeros,
            highs_att[:, :, 0, ...],
            highs_att[:, :, 1, ...],
            highs_att[:, :, 2, ...],
            highs_att[:, :, 3, ...],
            highs_att[:, :, 4, ...],
            highs_att[:, :, 5, ...],
            highs_att[:, :, 6, ...],
        )

        return low_rec, high_rec


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
            if normalization not in ('batchnorm', 'groupnorm', 'instancenorm', 'none'):
                raise ValueError(f"Unknown normalization: {normalization}")
            if normalization != 'none':
                ops.append(norm3d(normalization, n_filters_out))
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
            if normalization not in ('batchnorm', 'groupnorm', 'instancenorm', 'none'):
                raise ValueError(f"Unknown normalization: {normalization}")
            if normalization != 'none':
                ops.append(norm3d(normalization, n_filters_out))

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
            if normalization not in ('batchnorm', 'groupnorm', 'instancenorm', 'none'):
                raise ValueError(f"Unknown normalization: {normalization}")
            ops.append(norm3d(normalization, n_filters_out))
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
            if normalization not in ('batchnorm', 'groupnorm', 'instancenorm', 'none'):
                raise ValueError(f"Unknown normalization: {normalization}")
            ops.append(norm3d(normalization, n_filters_out))
        else:

            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class FreqGuidedChannelFusion3D(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        pool: str = "avgmax",
        dropout: float = 0.0,
        out_mode: str = "residual_mul",  # ['residual_mul', 'residual_add', 'mul']
    ):
        super().__init__()

        self.channels = channels
        self.pool = pool
        self.out_mode = out_mode

        hidden = max(channels // reduction, 1)
        # shared MLP (SE/CBAM style)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        self.act = nn.Sigmoid()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @staticmethod
    def _gap(x: torch.Tensor) -> torch.Tensor:
        # (B,C,D,H,W) -> (B,C)
        return x.mean(dim=(2, 3, 4))

    @staticmethod
    def _gmp(x: torch.Tensor) -> torch.Tensor:
        # (B,C,D,H,W) -> (B,C)
        return x.amax(dim=(2, 3, 4))

    def forward(self, fs: torch.Tensor, ff: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = fs.shape

        if self.pool == "avg":
            desc = self._gap(ff)
            attn = self.mlp(desc)
        elif self.pool == "max":
            desc = self._gmp(ff)
            attn = self.mlp(desc)
        else:  # 'avgmax' (CBAM style)
            attn = self.mlp(self._gap(ff)) + self.mlp(self._gmp(ff))

        w_ch = self.act(attn).view(b, c, 1, 1, 1)
        w_ch = self.drop(w_ch)

        # fuse
        if self.out_mode == "mul":
            return fs * w_ch
        elif self.out_mode == "residual_add":
            # fs + fs*w  (explicit form)
            return fs + fs * w_ch
        else:  # 'residual_mul'
            return fs * (1.0 + w_ch)


class WaveletDown(nn.Module):
    def __init__(self,c_in,c_out,n_stages,n_blocks,window_size,reduction,num_heads,normalization: NormType = "instancenorm"):
        super().__init__()

        self.reduction=reduction
        self.n_blocks=n_blocks
        self.window_size=window_size
        self.num_heads=num_heads
        self.conv = ConvBlock(
            n_stages=n_stages,
            n_filters_in=c_in,
            n_filters_out=c_in,
            normalization=normalization
        )
        self.fre=FreqEnhanceBlock(
            channels=c_in,n_blocks=n_blocks,window_size=window_size,num_heads=num_heads,reduction=reduction
        )

        self.fuse = FreqGuidedChannelFusion3D(channels=c_in, reduction=reduction)

        
        self.down = DownsamplingConvBlock(c_in, c_out, stride=2, normalization=normalization)

    def forward(self, x_spatial: torch.Tensor):
      
        x_spatial = self.conv(x_spatial) 
        x_fre=self.fre(x_spatial)
        x_fuse=self.fuse(x_spatial,x_fre)
        skip_enc=x_fuse
        x_next= self.down(x_fuse)               
        
        return x_next, skip_enc
    

class Encoder(nn.Module):
    def __init__(self, n_channels=1, n_classes=14, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock
        self.stem = convBlock(1, n_channels, n_filters, normalization=normalization)

        self.down1 = WaveletDown(n_filters,      n_filters * 2,  n_stages=1,num_heads=1,window_size=(3,4,4),reduction=4,n_blocks=1,normalization=normalization)
        self.down2 = WaveletDown(n_filters * 2,  n_filters * 4,  n_stages=2,num_heads=2,window_size=(6,6,6),reduction=8,n_blocks=1,normalization=normalization)
        self.down3 = WaveletDown(n_filters * 4,  n_filters * 8,  n_stages=3,num_heads=4,window_size=(6,6,6),reduction=8,n_blocks=2,normalization=normalization)
        self.down4 = WaveletDown(n_filters * 8,  n_filters * 16, n_stages=3,num_heads=8,window_size=(3,3,3),reduction=16,n_blocks=2,normalization=normalization)

        self.bottleneck = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, input):
        x = self.stem(input)

        x1,s1 = self.down1(x)
        x2,s2 = self.down2(x1)
        x3,s3 = self.down3(x2)
        x4,s4 = self.down4(x3)

        z = self.bottleneck(x4)

        return {
            'skips': [s1, s2, s3, s4],
            'bottleneck': z,
        }
    

class Decoder(nn.Module):
    def __init__(self,n_classes=14, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
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
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features):
        s1, s2, s3, s4 = features['skips']
        z = features['bottleneck']

        x5_up = self.block_five_up(z)
        x5_up = x5_up + s4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + s3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + s2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + s1
        x9 = self.block_nine(x8_up)
        
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)
        return out_seg


class VNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=14, patch_size=96, n_filters=16,
                 normalization='instancenorm', has_dropout=False, has_residual=False,
                 input_layout: str = "NCHWD"):   
        super(VNet, self).__init__()
        self.num_classes = n_classes
        self.input_layout = input_layout
        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
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
