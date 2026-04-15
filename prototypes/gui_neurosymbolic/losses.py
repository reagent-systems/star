from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .symbolic_planner import SymbolicPlanner, neural_logits_to_action_dict


def multi_task_loss(
    logits: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg: ModelConfig,
    *,
    thought_weight: float = 0.08,
    bbox_weight: float = 0.35,
    action_weight: float = 1.8,
    actions_only: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Weighted sum of CE losses. Thought/bbox are down-weighted so action heads get gradient
    (24-slot thought CE otherwise dominates).
    """
    ce = nn.CrossEntropyLoss()

    thought_logits = logits["thought_logits"]
    b, tl, tv = thought_logits.shape
    t_flat = thought_logits.view(b * tl, tv)
    tgt = batch["thought_ids"].view(b * tl)
    loss_thought = ce(t_flat, tgt)

    loss_obj = ce(logits["object_type_logits"], batch["object_type"])
    loss_bbox = sum(
        ce(logits["bbox_logits"][:, i, :], batch["bbox_bins"][:, i]) for i in range(4)
    ) / 4.0

    loss_at = ce(logits["action_type_logits"], batch["action_type"])
    loss_click = ce(logits["click_logits"], batch["click_cell"])
    loss_key = ce(logits["key_logits"], batch["key_id"])

    if actions_only:
        total = action_weight * loss_at + action_weight * loss_click
    else:
        total = (
            thought_weight * loss_thought
            + loss_obj
            + bbox_weight * loss_bbox
            + action_weight * loss_at
            + action_weight * loss_click
            + loss_key
        )
    parts: Dict[str, float] = {
        "thought": float(loss_thought.detach()),
        "object": float(loss_obj.detach()),
        "bbox": float(loss_bbox.detach()),
        "action_type": float(loss_at.detach()),
        "click": float(loss_click.detach()),
        "key": float(loss_key.detach()),
    }

    return total, parts


def planner_consistency_loss(
    logits: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg: ModelConfig,
    planner: SymbolicPlanner,
    task_texts: List[str],
    neural_symbols: List[Dict[str, Any]],
) -> torch.Tensor:
    """
    When a symbolic rule fires and proposes a click, add CE toward grid cell derived from
    neural bbox center (rules that set prefer=center_of_bbox). Otherwise 0.
    """
    device = logits["click_logits"].device
    total = torch.tensor(0.0, device=device)
    b = logits["action_type_logits"].shape[0]
    n = 0
    for i in range(b):
        raw = neural_logits_to_action_dict(
            logits["action_type_logits"],
            logits["click_logits"],
            logits["key_logits"],
            cfg.grid_h,
            cfg.grid_w,
            batch_idx=i,
        )
        planned = planner.plan(task_texts[i], neural_symbols[i], neural_action=raw)
        if not planned.get("rule_name"):
            continue
        action = planned.get("action") or {}
        if action.get("type") != "click":
            continue
        x = float(action.get("x", 0.5))
        y = float(action.get("y", 0.5))
        col = min(cfg.grid_w - 1, max(0, int(x * cfg.grid_w)))
        row = min(cfg.grid_h - 1, max(0, int(y * cfg.grid_h)))
        target_cell = row * cfg.grid_w + col
        total = total + F.cross_entropy(
            logits["click_logits"][i : i + 1],
            torch.tensor([target_cell], device=device, dtype=torch.long),
        )
        n += 1
    if n == 0:
        return total
    return total / n
