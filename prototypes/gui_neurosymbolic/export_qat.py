#!/usr/bin/env python3
"""
Post-training static quantization hooks + ONNX export and file size report.

Example::

    .venv/bin/python prototypes/gui_neurosymbolic/export_qat.py --checkpoint out/gui_ns.pt \\
        --onnx-out out/gui_ns.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn

from prototypes.gui_neurosymbolic.config import ModelConfig
from prototypes.gui_neurosymbolic.model import GUILiteNeuroSymbolicModel, build_model, param_count


class _OnnxWrapper(nn.Module):
    """ONNX export expects tensor outputs, not a dict."""

    def __init__(self, inner: GUILiteNeuroSymbolicModel) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, image: torch.Tensor, task_token_ids: torch.Tensor) -> tuple:
        o = self.inner(image, task_token_ids)
        return (
            o["thought_logits"],
            o["object_type_logits"],
            o["bbox_logits"],
            o["action_type_logits"],
            o["click_logits"],
            o["key_logits"],
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, default="", help="Weights from train.py")
    p.add_argument("--onnx-out", type=str, default="out/gui_lite.onnx")
    p.add_argument("--size-budget-mb", type=float, default=16.0)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def _size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


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

    example_image = torch.randn(1, 3, cfg.image_size, cfg.image_size, device=device)
    example_task = torch.zeros(1, cfg.task_max_len, dtype=torch.long, device=device)

    wrapped = _OnnxWrapper(model)
    p_before = param_count(model)
    print(f"float32 params: {p_before}")

    out_path = Path(args.onnx_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapped,
        (example_image, example_task),
        str(out_path),
        input_names=["image", "task_token_ids"],
        output_names=[
            "thought_logits",
            "object_type_logits",
            "bbox_logits",
            "action_type_logits",
            "click_logits",
            "key_logits",
        ],
        dynamic_axes={
            "image": {0: "batch"},
            "task_token_ids": {0: "batch"},
            "thought_logits": {0: "batch"},
            "object_type_logits": {0: "batch"},
            "bbox_logits": {0: "batch"},
            "action_type_logits": {0: "batch"},
            "click_logits": {0: "batch"},
            "key_logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    mb = _size_mb(out_path)
    print(f"ONNX written: {out_path} ({mb:.3f} MB)")
    if mb > args.size_budget_mb:
        print(f"Warning: size exceeds budget {args.size_budget_mb} MB (quantize/prune for deployment).")

    # Full-model static QAT needs replacing Linear+MultiheadAttention stacks with quantized
    # stubs; use torch.ao.quantization.quantize_dynamic on linear layers for a quick int8 baseline.
    try:
        qmodel = GUILiteNeuroSymbolicModel(cfg)
        if args.checkpoint:
            ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
            qmodel.load_state_dict(sd)
        qmodel.eval()
        qdyn = torch.ao.quantization.quantize_dynamic(
            qmodel, {torch.nn.Linear}, dtype=torch.qint8, inplace=False
        )
        dyn_path = out_path.with_suffix(".int8_dyn.pt")
        torch.save(qdyn, dyn_path)
        print(f"Dynamic quantized model saved: {dyn_path} ({_size_mb(dyn_path):.3f} MB)")
    except Exception as e:
        print(
            "Dynamic quantize skipped (expected on some CPU builds without FBGEMM/QNNPACK):",
            e,
        )


if __name__ == "__main__":
    main()
