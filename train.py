"""
Synthetic arithmetic S-expressions + AdamW training; optional overfit sanity check.
"""

from __future__ import annotations

import argparse
import random
from typing import List, Tuple

import torch
import torch.nn as nn

from lisp_graph import eval_arithmetic, parse_sexpr, batch_trees
from model import NeuroSymbolicGraphTransformer


def random_expr(rng: random.Random, depth: int) -> str:
    """Build a random (+ / - / *) S-expression with integers in 0..9 (mostly shallow for learnability)."""
    if depth <= 0 or rng.random() < 0.45:
        return str(rng.randint(0, 9))
    op = rng.choice(["+", "-", "*"])
    if op == "-":
        if rng.random() < 0.45:
            return f"(- {random_expr(rng, depth - 1)})"
        return f"(- {random_expr(rng, depth - 1)} {random_expr(rng, depth - 1)})"
    n_args = 2 if rng.random() < 0.55 else rng.randint(2, 4)
    args = [random_expr(rng, depth - 1) for _ in range(n_args)]
    inner = " ".join(args)
    return f"({op} {inner})"


def make_dataset(n: int, seed: int) -> List[Tuple[str, float]]:
    rng = random.Random(seed)
    out: List[Tuple[str, float]] = []
    tries = 0
    while len(out) < n and tries < n * 50:
        tries += 1
        s = random_expr(rng, depth=3)
        try:
            t = parse_sexpr(s)
            y = eval_arithmetic(t)
        except (ValueError, RuntimeError):
            continue
        if not (-200.0 <= y <= 200.0):
            continue
        out.append((s, y))
    return out


def tensors_from_strings(exprs: List[str], device: torch.device) -> dict:
    trees = [parse_sexpr(s) for s in exprs]
    return batch_trees(trees, device=device)


def train_step(
    model: nn.Module,
    batch: dict,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    clip: float = 0.0,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    pred = model(batch)
    loss = criterion(pred, targets)
    loss.backward()
    if clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def eval_loss(model: nn.Module, batch: dict, targets: torch.Tensor, criterion: nn.Module) -> float:
    model.eval()
    pred = model(batch)
    return float(criterion(pred, targets).item())


def overfit_sanity(device: torch.device) -> None:
    """Train on a handful of examples until loss is tiny."""
    torch.manual_seed(0)
    examples = [
        "(+ 1 2)",
        "(* 2 3)",
        "(- 5 2)",
        "(* (+ 1 2) 3)",
    ]
    ys = [eval_arithmetic(parse_sexpr(s)) for s in examples]
    batch = tensors_from_strings(examples, device)
    targets = torch.tensor(ys, dtype=torch.float32, device=device)

    model = NeuroSymbolicGraphTransformer(d_model=64, n_heads=4, n_layers=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.0)
    crit = nn.MSELoss()

    for step in range(4000):
        loss = train_step(model, batch, targets, opt, crit, clip=1.0)
        if step % 1000 == 0 or step == 3999:
            print(f"  overfit step {step:4d} loss={loss:.6f}")

    with torch.no_grad():
        pred = model(batch)
    max_err = (pred - targets).abs().max().item()
    print(f"  overfit max abs error: {max_err:.6f}")
    if max_err > 0.15:
        raise RuntimeError("overfit sanity check failed: model should nearly memorize tiny batch")


def main() -> None:
    p = argparse.ArgumentParser(description="Train neurosymbolic graph transformer on arithmetic S-exprs")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-overfit-test", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)

    if not args.no_overfit_test:
        print("Running overfit sanity check...")
        overfit_sanity(device)

    torch.manual_seed(args.seed)
    train_data = make_dataset(4000, seed=args.seed)
    val_data = make_dataset(500, seed=args.seed + 1)
    print(f"train pairs: {len(train_data)}, val pairs: {len(val_data)}")

    compound = [s for s, _ in train_data if "(" in s]
    demo_exprs = (compound[:3] if len(compound) >= 3 else [train_data[i][0] for i in range(3)])

    model = NeuroSymbolicGraphTransformer(d_model=64, n_heads=4, n_layers=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.SmoothL1Loss(beta=0.5)

    for step in range(args.steps):
        batch_exprs = [random.choice(train_data)[0] for _ in range(args.batch_size)]
        ys = [eval_arithmetic(parse_sexpr(s)) for s in batch_exprs]
        batch = tensors_from_strings(batch_exprs, device)
        targets = torch.tensor(ys, dtype=torch.float32, device=device)
        loss = train_step(model, batch, targets, opt, crit)

        if step % 200 == 0 or step == args.steps - 1:
            vbatch = [val_data[i][0] for i in range(min(64, len(val_data)))]
            vys = [val_data[i][1] for i in range(len(vbatch))]
            vb = tensors_from_strings(vbatch, device)
            vt = torch.tensor(vys, dtype=torch.float32, device=device)
            vloss = eval_loss(model, vb, vt, crit)
            print(f"step {step:5d} train_loss={loss:.4f} val_loss={vloss:.4f}")

    # Sample inference (expressions drawn from training set so they are in-distribution)
    with torch.no_grad():
        b = tensors_from_strings(demo_exprs, device)
        pred = model(b)
    print("\nSample predictions (from training set):")
    for s, p in zip(demo_exprs, pred.tolist()):
        gt = eval_arithmetic(parse_sexpr(s))
        print(f"  {s:30s}  pred={p:8.3f}  target={gt:.1f}")


if __name__ == "__main__":
    main()
