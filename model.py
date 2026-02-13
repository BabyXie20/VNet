from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for 3D medical image segmentation with windowed attention."""

    patch_size: tuple[int, int, int] = (96, 96, 96)
    in_channels: int = 1
    out_channels: int = 14
    feature_size: int = 48
    depths: tuple[int, int, int, int] = (2, 2, 2, 2)
    num_heads: tuple[int, int, int, int] = (3, 6, 12, 24)
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    dropout_path_rate: float = 0.1
    normalize: bool = True
    use_checkpoint: bool = False
    spatial_dims: int = 3


DEFAULT_CONFIG = ModelConfig()


class VNet(nn.Module):
    """
    Windowed-attention segmentation model based on MONAI SwinUNETR.

    Note:
        MONAI is imported lazily so this module can still be imported in environments
        where MONAI is not preinstalled.
    """

    def __init__(self, config: ModelConfig = DEFAULT_CONFIG):
        super().__init__()
        self.config = config
        self.model = self._build_monai_model(config)

    @staticmethod
    def _build_monai_model(config: ModelConfig) -> nn.Module:
        try:
            from monai.networks.nets import SwinUNETR
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "MONAI is required for this model. Please install it with: pip install monai"
            ) from exc

        return SwinUNETR(
            img_size=config.patch_size,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            feature_size=config.feature_size,
            depths=config.depths,
            num_heads=config.num_heads,
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            dropout_path_rate=config.dropout_path_rate,
            normalize=config.normalize,
            use_checkpoint=config.use_checkpoint,
            spatial_dims=config.spatial_dims,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def get_model_config() -> dict[str, Any]:
    """Export default training/inference config for patch size (96,96,96)."""

    cfg = asdict(DEFAULT_CONFIG)
    cfg.update(
        {
            "model_name": "SwinUNETR",
            "attention_type": "windowed_attention",
            "recommended_train_batch_size": 2,
            "recommended_infer_batch_size": 1,
            "optimizer": {"name": "AdamW", "lr": 1e-4, "weight_decay": 1e-5},
            "scheduler": {"name": "CosineAnnealingLR", "T_max": 500},
            "loss": {"name": "DiceCELoss", "include_background": True, "to_onehot_y": True, "softmax": True},
            "post_process": {"name": "AsDiscrete", "argmax": True},
        }
    )
    return cfg


def build_model(config: ModelConfig = DEFAULT_CONFIG) -> VNet:
    return VNet(config)
