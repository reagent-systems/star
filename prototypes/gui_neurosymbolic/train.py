#!/usr/bin/env python3
"""
Train GUI lite neuro-symbolic model on synthetic data (overfit sanity) or real datasets later.

Example::

    .venv/bin/python prototypes/gui_neurosymbolic/train.py --epochs 3 --device cpu --batch-size 8
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn
from torch import amp
from torch.utils.data import DataLoader

from prototypes.gui_neurosymbolic.config import ModelConfig
from prototypes.gui_neurosymbolic.dataset import (
    StructuredSyntheticGUIDataset,
    SyntheticGUIDataset,
    collate_batch,
)
from prototypes.gui_neurosymbolic.losses import multi_task_loss, planner_consistency_loss
from prototypes.gui_neurosymbolic.model import build_model
from prototypes.gui_neurosymbolic.symbolic_planner import SymbolicPlanner


def _symbols_from_batch(batch: Dict[str, torch.Tensor], cfg: ModelConfig) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    b = batch["object_type"].shape[0]
    for i in range(b):
        bb = [int(batch["bbox_bins"][i, j].item()) for j in range(4)]
        cx = (bb[0] + 0.5) / cfg.bbox_bins
        cy = (bb[1] + 0.5) / cfg.bbox_bins
        w = (bb[2] + 1) / cfg.bbox_bins
        h = (bb[3] + 1) / cfg.bbox_bins
        out.append(
            {
                "object_type_id": int(batch["object_type"][i].item()),
                "bbox_norm": [cx, cy, w, h],
                "bbox_bins": bb,
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--train-samples", type=int, default=256)
    p.add_argument("--eval-samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--save", type=str, default="", help="Path to save checkpoint .pt")
    p.add_argument("--rules", type=str, default="", help="Optional JSON rules for consistency loss")
    p.add_argument("--consistency-weight", type=float, default=0.0)
    p.add_argument("--early-stop-patience", type=int, default=0, help="0 = disabled")
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision even on CUDA")
    p.add_argument(
        "--data-mode",
        choices=("structured", "random"),
        default="structured",
        help="structured: deterministic labels from index (learnable); random: i.i.d. noise",
    )
    p.add_argument("--thought-weight", type=float, default=0.08)
    p.add_argument("--bbox-weight", type=float, default=0.35)
    p.add_argument("--action-weight", type=float, default=1.8)
    p.add_argument(
        "--eval-on-train",
        action="store_true",
        help="Measure eval loss/acc on the training set (overfit sanity; expect high acc when the stack fits).",
    )
    p.add_argument(
        "--toy",
        action="store_true",
        help="Smaller heads (8x8 click grid, short thought) so structured data can reach high acc in reasonable time.",
    )
    p.add_argument(
        "--actions-only",
        action="store_true",
        help="Optimize only action_type + click_cell CE (sanity / curriculum).",
    )
    p.add_argument(
        "--easy-structured",
        action="store_true",
        help="Use smooth cosine-grid images (stronger learnability than full PRNG noise).",
    )
    p.add_argument(
        "--encode-index",
        action="store_true",
        help="Embed global index in first pixels + modular targets (learnable sanity path).",
    )
    return p.parse_args()


@torch.no_grad()
def _eval_accuracy(
    model: nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    cfg: ModelConfig,
) -> Dict[str, float]:
    """Head-wise accuracy on eval set (structured data should approach 1.0 when fitted)."""
    model.eval()
    n_ex = 0
    sum_at = sum_cl = sum_obj = sum_th_tok = 0
    tl = cfg.thought_len
    for raw in eval_loader:
        batch = dict(raw)
        batch.pop("task_texts", None)
        for k in batch:
            batch[k] = batch[k].to(device)
        bsz = batch["image"].shape[0]
        logits = model(batch["image"], batch["task_token_ids"])
        pred_at = logits["action_type_logits"].argmax(dim=-1)
        pred_cl = logits["click_logits"].argmax(dim=-1)
        pred_obj = logits["object_type_logits"].argmax(dim=-1)
        pred_th = logits["thought_logits"].argmax(dim=-1)
        sum_at += int((pred_at == batch["action_type"]).sum().item())
        sum_cl += int((pred_cl == batch["click_cell"]).sum().item())
        sum_obj += int((pred_obj == batch["object_type"]).sum().item())
        sum_th_tok += int((pred_th == batch["thought_ids"]).sum().item())
        n_ex += bsz
    denom_th = max(1, n_ex * tl)
    return {
        "acc_action_type": sum_at / max(1, n_ex),
        "acc_click_cell": sum_cl / max(1, n_ex),
        "acc_object_type": sum_obj / max(1, n_ex),
        "acc_thought_token": sum_th_tok / denom_th,
    }


def train() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested (--device cuda) but this PyTorch build has no CUDA "
            "(torch.cuda.is_available() is False).\n"
            "On Google Colab: Runtime → Change runtime type → GPU (T4), then avoid "
            "`pip install torch` without the CUDA wheel index — it often replaces "
            "Colab's GPU build with a CPU-only wheel.\n"
            "Install extras with: pip install onnx\n"
            "Or train on CPU: --device cpu"
        )

    device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp

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
    model, n_params = build_model(cfg)
    if n_params > cfg.max_param_budget:
        raise RuntimeError(f"Model has {n_params} params, budget {cfg.max_param_budget}")
    model.to(device)

    if args.data_mode == "structured":
        train_ds = StructuredSyntheticGUIDataset(
            cfg,
            args.train_samples,
            index_start=0,
            easy=args.easy_structured,
            encode_index=args.encode_index,
        )
        eval_ds = (
            train_ds
            if args.eval_on_train
            else StructuredSyntheticGUIDataset(
                cfg,
                args.eval_samples,
                index_start=args.train_samples,
                easy=args.easy_structured,
                encode_index=args.encode_index,
            )
        )
    else:
        train_ds = SyntheticGUIDataset(cfg, args.train_samples, seed=args.seed)
        eval_ds = train_ds if args.eval_on_train else SyntheticGUIDataset(
            cfg, args.eval_samples, seed=args.seed + 1
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_batch,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_batch,
    )

    planner: SymbolicPlanner | None = None
    if args.rules:
        planner = SymbolicPlanner.from_json_file(args.rules)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = amp.GradScaler("cuda", enabled=use_amp)

    best_eval = math.inf
    patience_left = args.early_stop_patience

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        n_batches = 0
        opt.zero_grad(set_to_none=True)
        accum = 0
        for step, raw in enumerate(train_loader):
            batch = dict(raw)
            task_texts = batch.pop("task_texts")
            for k in batch:
                batch[k] = batch[k].to(device)

            with amp.autocast("cuda", enabled=use_amp):
                logits = model(batch["image"], batch["task_token_ids"])
                loss, _parts = multi_task_loss(
                    logits,
                    batch,
                    cfg,
                    thought_weight=args.thought_weight,
                    bbox_weight=args.bbox_weight,
                    action_weight=args.action_weight,
                    actions_only=args.actions_only,
                )
                if planner is not None and args.consistency_weight > 0:
                    syms = _symbols_from_batch(batch, cfg)
                    cl = planner_consistency_loss(
                        logits, batch, cfg, planner, task_texts, syms
                    )
                    loss = loss + args.consistency_weight * cl

            scaler.scale(loss / args.accum_steps).backward()
            running += float(loss.detach())
            n_batches += 1
            accum += 1

            if accum >= args.accum_steps:
                if args.grad_clip > 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                accum = 0

        if accum > 0:
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        avg_train = running / max(1, n_batches)

        model.eval()
        ev_sum = 0.0
        ev_n = 0
        with torch.no_grad():
            for batch in eval_loader:
                batch = dict(batch)
                batch.pop("task_texts", None)
                for k in batch:
                    batch[k] = batch[k].to(device)
                logits = model(batch["image"], batch["task_token_ids"])
                loss, _ = multi_task_loss(
                    logits,
                    batch,
                    cfg,
                    thought_weight=args.thought_weight,
                    bbox_weight=args.bbox_weight,
                    action_weight=args.action_weight,
                    actions_only=args.actions_only,
                )
                ev_sum += float(loss)
                ev_n += 1
        avg_eval = ev_sum / max(1, ev_n)

        accs = _eval_accuracy(model, eval_loader, device, cfg)
        print(
            f"epoch {epoch + 1}/{args.epochs} train_loss={avg_train:.4f} eval_loss={avg_eval:.4f} "
            f"params={n_params} | eval "
            f"acc_at={accs['acc_action_type']:.3f} acc_click={accs['acc_click_cell']:.3f} "
            f"acc_obj={accs['acc_object_type']:.3f} acc_th_tok={accs['acc_thought_token']:.3f}"
        )

        if args.early_stop_patience > 0:
            if avg_eval < best_eval:
                best_eval = avg_eval
                patience_left = args.early_stop_patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print("Early stopping.")
                    break

    if args.save:
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, path)
        print(f"Saved {path}")


if __name__ == "__main__":
    train()
