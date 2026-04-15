from __future__ import annotations

import random
from typing import Any, Dict, List

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


_TASK_POOL = (
    "click submit to search",
    "open menu cancel dialog",
    "scroll down and wait",
    "submit form",
)


def _u32(x: int) -> int:
    return x & 0xFFFFFFFF


def _split_mix(seed: int) -> int:
    """32-bit xorshift-style mix for deterministic pseudo-random ints from a global index."""
    s = _u32(seed ^ 0x9E3779B9)
    s = _u32(s * 1664525 + 1013904223)
    s ^= s >> 16
    s = _u32(s * 2246822519)
    s ^= s >> 13
    return s


def _rand_chain(gi: int, n: int) -> List[int]:
    out: List[int] = []
    s = _split_mix(gi)
    for _ in range(n):
        s = _u32(s * 2654435761 + 1)
        out.append(s)
    return out


def _embed_index_in_image(img: torch.Tensor, gi: int, n_bytes: int = 12) -> None:
    """Write ``gi`` into the first channels (little-endian) so CNN can read index from pixels."""
    flat = img.view(-1)
    v = gi
    for i in range(min(n_bytes, flat.numel())):
        flat[i] = (v & 0xFF) / 255.0
        v >>= 8


def structured_sample(
    cfg: ModelConfig,
    gi: int,
    *,
    easy: bool = False,
    encode_index: bool = False,
) -> Dict[str, Any]:
    """
    Build (image, task), then derive targets from ``gi`` (deterministic).
    If ``encode_index`` is True, ``gi`` is written into the first pixels and labels use
    simple modular arithmetic so the mapping is learnable from local image patches.
    """
    g = grid_cells(cfg)
    task = _TASK_POOL[gi % len(_TASK_POOL)]
    task_ids = task_text_to_ids(task, cfg.task_max_len, cfg.task_vocab_size)

    if easy:
        # Smooth, low-frequency images (learnable); labels stay deterministic in gi.
        t = torch.linspace(-1.0, 1.0, cfg.image_size)
        xx, yy = torch.meshgrid(t, t, indexing="ij")
        phase = (gi % 17) / 17.0 * 6.28318
        c1 = torch.cos(xx * 3.0 + phase) * 0.5 + 0.5
        c2 = torch.sin(yy * 3.0 + phase * 0.7) * 0.5 + 0.5
        c3 = torch.full_like(c1, (gi % 11) / 11.0)
        img = torch.stack([c1, c2, c3], dim=0)
    else:
        gen = torch.Generator()
        gen.manual_seed(_split_mix(gi))
        img = torch.rand(3, cfg.image_size, cfg.image_size, generator=gen)
        tid = (task_ids.float().mean().item() + gi % 7) * 0.01
        img[:, :8, :8] += tid * 0.12
        img = img.clamp(0.0, 1.0)

    if encode_index:
        _embed_index_in_image(img, gi)
        thought = [
            ((gi * (i + 31)) ^ (gi >> 2)) % cfg.thought_vocab_size
            for i in range(cfg.thought_len)
        ]
        obj_t = gi % cfg.num_object_types
        bbox = [((gi >> (2 * j)) ^ (gi << 1)) % cfg.bbox_bins for j in range(4)]
        a_type = gi % cfg.num_action_types
        click = (gi // max(1, cfg.num_action_types)) % g
        key_id = (gi // max(1, g * cfg.num_action_types)) % cfg.num_key_ids
    else:
        rs = _rand_chain(_split_mix(gi), cfg.thought_len + 10)
        thought = [rs[i] % cfg.thought_vocab_size for i in range(cfg.thought_len)]
        obj_t = rs[cfg.thought_len] % cfg.num_object_types
        bbox = [
            rs[cfg.thought_len + 1 + j] % cfg.bbox_bins for j in range(4)
        ]
        a_type = rs[cfg.thought_len + 5] % cfg.num_action_types
        click = rs[cfg.thought_len + 6] % g
        key_id = rs[cfg.thought_len + 7] % cfg.num_key_ids

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


class StructuredSyntheticGUIDataset(Dataset):
    """
    Indices ``index_start .. index_start+n-1`` — use disjoint ranges for train vs eval.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        n: int,
        index_start: int = 0,
        *,
        easy: bool = False,
        encode_index: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.n = n
        self.index_start = index_start
        self.easy = easy
        self.encode_index = encode_index

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        gi = self.index_start + idx
        return structured_sample(
            self.cfg, gi, easy=self.easy, encode_index=self.encode_index
        )


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
