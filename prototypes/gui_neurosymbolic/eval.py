#!/usr/bin/env python3
"""
Stub metrics: action accuracy on a tensor batch (extend with Selenium / replay harness).

Example::

    .venv/bin/python prototypes/gui_neurosymbolic/eval.py --checkpoint out/gui_ns.pt --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from torch.utils.data import DataLoader

from prototypes.gui_neurosymbolic.config import ModelConfig
from prototypes.gui_neurosymbolic.dataset import (
    StructuredSyntheticGUIDataset,
    SyntheticGUIDataset,
    collate_batch,
)
from prototypes.gui_neurosymbolic.model import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--data-mode",
        choices=("structured", "random"),
        default="structured",
    )
    p.add_argument(
        "--index-start",
        type=int,
        default=0,
        help="For structured mode: global index of first sample (match training eval offset).",
    )
    p.add_argument(
        "--encode-index",
        action="store_true",
        help="Must match training (embed gi in pixels + modular labels).",
    )
    p.add_argument(
        "--easy-structured",
        action="store_true",
        help="Must match training.",
    )
    p.add_argument(
        "--toy",
        action="store_true",
        help="Must match training (smaller heads).",
    )
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cfg = ModelConfig()
    if args.toy:
        cfg.thought_len = 8
        cfg.thought_vocab_size = 64
        cfg.grid_h = 8
        cfg.grid_w = 8
        cfg.num_object_types = 8
        cfg.num_action_types = 4
        cfg.bbox_bins = 8
        cfg.num_key_ids = 8
    model, _ = build_model(cfg)
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        model.load_state_dict(sd)
    model.to(device)
    model.eval()

    if args.data_mode == "structured":
        ds = StructuredSyntheticGUIDataset(
            cfg,
            args.samples,
            index_start=args.index_start,
            easy=args.easy_structured,
            encode_index=args.encode_index,
        )
    else:
        ds = SyntheticGUIDataset(cfg, args.samples, seed=0)
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_batch)

    correct_action = 0
    correct_click = 0
    correct_obj = 0
    correct_th = 0
    total = 0
    tl = cfg.thought_len
    for batch in loader:
        bsz = batch["image"].shape[0]
        for k in batch:
            if k == "task_texts":
                continue
            batch[k] = batch[k].to(device)
        logits = model(batch["image"], batch["task_token_ids"])
        pred_at = logits["action_type_logits"].argmax(dim=-1)
        pred_click = logits["click_logits"].argmax(dim=-1)
        pred_obj = logits["object_type_logits"].argmax(dim=-1)
        pred_th = logits["thought_logits"].argmax(dim=-1)
        correct_action += int((pred_at == batch["action_type"]).sum().item())
        correct_click += int((pred_click == batch["click_cell"]).sum().item())
        correct_obj += int((pred_obj == batch["object_type"]).sum().item())
        correct_th += int((pred_th == batch["thought_ids"]).sum().item())
        total += bsz

    print(f"action_type_acc={correct_action / max(1, total):.4f}")
    print(f"click_cell_acc={correct_click / max(1, total):.4f}")
    print(f"object_type_acc={correct_obj / max(1, total):.4f}")
    print(f"thought_token_acc={correct_th / max(1, total * tl):.4f}")


if __name__ == "__main__":
    main()
