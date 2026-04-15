from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Hyperparameters sized for ~10–15M total parameters (float32)."""

    image_size: int = 224
    vision_width: int = 96
    vision_dim: int = 384

    task_vocab_size: int = 256
    task_max_len: int = 64

    d_model: int = 384
    nhead: int = 6
    num_fusion_layers: int = 4
    dim_feedforward: int = 1536
    dropout: float = 0.1

    thought_len: int = 24
    thought_vocab_size: int = 512

    num_object_types: int = 32
    bbox_bins: int = 16

    grid_h: int = 32
    grid_w: int = 32
    num_action_types: int = 8
    num_key_ids: int = 32

    max_param_budget: int = 15_000_000


def grid_cells(cfg: ModelConfig) -> int:
    return cfg.grid_h * cfg.grid_w


def total_action_logits(cfg: ModelConfig) -> int:
    """Flat index: type * (grid+keys) + offset — we use separate heads instead."""
    return cfg.num_action_types + cfg.grid_cells(cfg) + cfg.num_key_ids
