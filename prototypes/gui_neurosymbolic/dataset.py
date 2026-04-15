from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .config import ModelConfig, grid_cells


def task_text_to_ids(text: str, max_len: int, vocab_size: int) -> torch.Tensor:
    """Char-level token ids (0..vocab_size-1)."""
    b = text.encode("utf-8", errors="ignore")[:max_len]
    ids = [c % vocab_size for c in b]
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


class SyntheticGUIDataset(Dataset):
    """
    Random tensors + labels for sanity / overfit checks (no real screenshots).
    Each sample: noise image, task string, parallel targets for all heads.
    """

    def __init__(self, cfg: ModelConfig, n: int, seed: int = 0) -> None:
        super().__init__()
        self.cfg = cfg
        self.n = n
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rng = self.rng
        cfg = self.cfg
        g = grid_cells(cfg)
        # Deterministic per index for reproducibility
        rng.seed(idx * 7919 + 13)

        img = torch.randn(3, cfg.image_size, cfg.image_size)
        task = rng.choice(
            [
                "click submit to search",
                "open menu cancel dialog",
                "scroll down and wait",
                "submit form",
            ]
        )
        task_ids = task_text_to_ids(task, cfg.task_max_len, cfg.task_vocab_size)

        thought = [rng.randrange(cfg.thought_vocab_size) for _ in range(cfg.thought_len)]
        obj_t = rng.randrange(cfg.num_object_types)
        bbox = [rng.randrange(cfg.bbox_bins) for _ in range(4)]
        a_type = rng.randrange(cfg.num_action_types)
        click = rng.randrange(g)
        key_id = rng.randrange(cfg.num_key_ids)

        return {
            "image": img,
            "task_token_ids": task_ids,
            "thought_ids": torch.tensor(thought, dtype=torch.long),
            "object_type": torch.tensor(obj_t, dtype=torch.long),
            "bbox_bins": torch.tensor(bbox, dtype=torch.long),
            "action_type": torch.tensor(a_type, dtype=torch.long),
            "click_cell": torch.tensor(click, dtype=torch.long),
            "key_id": torch.tensor(key_id, dtype=torch.long),
            "task_text": task,
        }


def collate_batch(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([s["image"] for s in samples], dim=0)
    task_ids = torch.stack([s["task_token_ids"] for s in samples], dim=0)
    thought = torch.stack([s["thought_ids"] for s in samples], dim=0)
    obj = torch.stack([s["object_type"] for s in samples], dim=0)
    bbox = torch.stack([s["bbox_bins"] for s in samples], dim=0)
    at = torch.stack([s["action_type"] for s in samples], dim=0)
    click = torch.stack([s["click_cell"] for s in samples], dim=0)
    kid = torch.stack([s["key_id"] for s in samples], dim=0)
    task_texts = [str(s["task_text"]) for s in samples]
    return {
        "image": images,
        "task_token_ids": task_ids,
        "thought_ids": thought,
        "object_type": obj,
        "bbox_bins": bbox,
        "action_type": at,
        "click_cell": click,
        "key_id": kid,
        "task_texts": task_texts,
    }


def build_symbol_dict_from_batch(
    logits: Dict[str, torch.Tensor],
    batch_idx: int,
    cfg: ModelConfig,
) -> Dict[str, Any]:
    """Argmax neural symbol heads to JSON-like dict (for planner / consistency loss)."""
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
