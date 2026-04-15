#!/usr/bin/env python3
"""
Run inference: screenshot + task -> neural logits -> optional symbolic planner -> action JSON.

Example::

    .venv/bin/python prototypes/gui_neurosymbolic/infer.py --checkpoint out/gui_ns.pt \\
        --image path/to.png --task "click submit"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from torchvision.io import read_image
from torchvision.transforms.functional import resize

from prototypes.gui_neurosymbolic.config import ModelConfig
from prototypes.gui_neurosymbolic.dataset import task_text_to_ids
from prototypes.gui_neurosymbolic.model import GUILiteNeuroSymbolicModel, build_model
from prototypes.gui_neurosymbolic.symbolic_planner import (
    SymbolicPlanner,
    neural_logits_to_action_dict,
)


def load_image(path: str, size: int) -> torch.Tensor:
    t = read_image(path).float() / 255.0
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    t = resize(t, [size, size], antialias=True)
    return t


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, default="", help="Optional .pt from train.py")
    p.add_argument("--rules", type=str, default="", help="Optional symbolic rules JSON")
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--execute", action="store_true", help="If set, attempt pyautogui (requires display)")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cfg = ModelConfig()
    model, _ = build_model(cfg)
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        model.load_state_dict(sd)
    model.to(device)
    model.eval()

    img = load_image(args.image, cfg.image_size).unsqueeze(0).to(device)
    task_ids = task_text_to_ids(args.task, cfg.task_max_len, cfg.task_vocab_size).unsqueeze(0).to(device)

    logits = model(img, task_ids)
    sym = model.predict_symbol_json(logits, 0)
    raw_action = neural_logits_to_action_dict(
        logits["action_type_logits"],
        logits["click_logits"],
        logits["key_logits"],
        cfg.grid_h,
        cfg.grid_w,
        batch_idx=0,
    )

    out: Dict[str, Any] = {
        "task": args.task,
        "symbols": sym,
        "neural_action": raw_action,
    }

    if args.rules:
        planner = SymbolicPlanner.from_json_file(args.rules)
        planned = planner.plan(args.task, sym, neural_action=raw_action)
        out["symbolic"] = planned
        out["final_action"] = planned.get("action", raw_action)
    else:
        out["final_action"] = raw_action

    print(json.dumps(out, indent=2))

    if args.execute:
        import pyautogui

        fa = out.get("final_action") or {}
        t = fa.get("type")
        if t == "click":
            w, h = pyautogui.size()
            pyautogui.click(int(float(fa["x"]) * w), int(float(fa["y"]) * h))
        elif t == "keypress":
            pyautogui.press(str(fa.get("key", "enter")))


if __name__ == "__main__":
    main()
