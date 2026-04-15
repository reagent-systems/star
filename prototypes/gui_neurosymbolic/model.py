from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .config import ModelConfig, grid_cells


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pw(x)
        x = self.bn2(x)
        x = self.act(x)
        return x


class TinyVisionBackbone(nn.Module):
    """Lightweight CNN (~3–5M params) with depthwise-separable stacks."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        c = cfg.vision_width
        self.stem = nn.Sequential(
            nn.Conv2d(3, c // 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c // 2),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv2d(c // 2, c, stride=1),
            DepthwiseSeparableConv2d(c, c, stride=1),
        )
        self.stage2 = DepthwiseSeparableConv2d(c, int(c * 1.25), stride=2)
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv2d(int(c * 1.25), int(c * 1.5), stride=1),
            DepthwiseSeparableConv2d(int(c * 1.5), int(c * 1.5), stride=1),
        )
        self.stage4 = DepthwiseSeparableConv2d(int(c * 1.5), c, stride=2)
        self.stage5 = nn.Sequential(
            DepthwiseSeparableConv2d(c, c, stride=1),
            DepthwiseSeparableConv2d(c, c, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c, cfg.vision_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class TaskEncoder(nn.Module):
    """Embed discrete task token ids (char-level or hashed words)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(cfg.task_vocab_size, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, cfg.task_max_len, cfg.d_model))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, task_token_ids: torch.Tensor) -> torch.Tensor:
        b, l = task_token_ids.shape
        x = self.embed(task_token_ids.clamp(0, self.embed.num_embeddings - 1))
        x = x + self.pos[:, :l, :]
        return x


class FusionTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        enc = nn.TransformerEncoder(layer, num_layers=cfg.num_fusion_layers)
        enc.enable_nested_tensor = False
        for lyr in getattr(enc, "layers", []):
            if hasattr(lyr, "enable_nested_tensor"):
                lyr.enable_nested_tensor = False
        self.encoder = enc
        self.vision_proj = nn.Linear(cfg.vision_dim, cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, vision_vec: torch.Tensor, task_hidden: torch.Tensor) -> torch.Tensor:
        v = self.vision_proj(vision_vec).unsqueeze(1)
        h = torch.cat([v, task_hidden], dim=1)
        h = self.norm(h)
        return self.encoder(h)


class GUILiteNeuroSymbolicModel(nn.Module):
    """
    Multimodal stack: vision + task -> fusion -> thought / symbol logits / action heads.
    Action decomposition: action_type (CE) + click_cell (CE if click) + key_id (CE if key).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision = TinyVisionBackbone(cfg)
        self.task_enc = TaskEncoder(cfg)
        self.fusion = FusionTransformer(cfg)
        d = cfg.d_model
        self.fusion_pool = nn.Linear(d, d)
        self.thought_head = nn.Linear(d, cfg.thought_len * cfg.thought_vocab_size)
        self.object_type_head = nn.Linear(d, cfg.num_object_types)
        self.bbox_heads = nn.ModuleList([nn.Linear(d, cfg.bbox_bins) for _ in range(4)])
        self.action_type_head = nn.Linear(d, cfg.num_action_types)
        self.click_head = nn.Linear(d, grid_cells(cfg))
        self.key_head = nn.Linear(d, cfg.num_key_ids)

    def forward(
        self,
        image: torch.Tensor,
        task_token_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        vis = self.vision(image)
        tok = self.task_enc(task_token_ids)
        fused = self.fusion(vis, tok)
        pooled = torch.relu(self.fusion_pool(fused[:, 0, :]))

        tl, tv = self.cfg.thought_len, self.cfg.thought_vocab_size
        thought_logits = self.thought_head(pooled).view(-1, tl, tv)
        out: Dict[str, torch.Tensor] = {
            "thought_logits": thought_logits,
            "object_type_logits": self.object_type_head(pooled),
            "bbox_logits": torch.stack([h(pooled) for h in self.bbox_heads], dim=1),
            "action_type_logits": self.action_type_head(pooled),
            "click_logits": self.click_head(pooled),
            "key_logits": self.key_head(pooled),
        }
        return out

    def predict_symbol_json(
        self,
        logits: Dict[str, torch.Tensor],
        batch_idx: int = 0,
    ) -> Dict[str, object]:
        """Turn argmax symbol heads into a JSON-friendly dict (for the planner)."""
        cfg = self.cfg
        ot = int(logits["object_type_logits"][batch_idx].argmax().item())
        bb = [int(logits["bbox_logits"][batch_idx, i].argmax().item()) for i in range(4)]
        cx = (bb[0] + 0.5) / cfg.bbox_bins
        cy = (bb[1] + 0.5) / cfg.bbox_bins
        w = (bb[2] + 1) / cfg.bbox_bins
        h = (bb[3] + 1) / cfg.bbox_bins
        return {
            "object_type_id": ot,
            "object_type": f"type_{ot}",
            "bbox_norm": [cx, cy, w, h],
            "bbox_bins": bb,
        }


def param_count(module: nn.Module, trainable_only: bool = True) -> int:
    ps = module.parameters()
    if trainable_only:
        return sum(p.numel() for p in ps if p.requires_grad)
    return sum(p.numel() for p in ps)


def build_model(cfg: ModelConfig) -> Tuple[GUILiteNeuroSymbolicModel, int]:
    m = GUILiteNeuroSymbolicModel(cfg)
    n = param_count(m)
    return m, n
