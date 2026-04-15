from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Union

import torch


RuleDict = Dict[str, Any]


def _load_json(path: Union[str, Path]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _task_matches(cond: Mapping[str, Any], task_lower: str) -> bool:
    if "task_contains" in cond:
        if str(cond["task_contains"]).lower() not in task_lower:
            return False
    if "task_contains_not" in cond:
        if str(cond["task_contains_not"]).lower() in task_lower:
            return False
    return True


def _object_matches(cond: Mapping[str, Any], neural_symbols: Mapping[str, Any]) -> bool:
    if "object_type_id" in cond:
        want = cond["object_type_id"]
        oid = neural_symbols.get("object_type_id")
        if oid is None:
            return False
        if isinstance(want, list):
            if int(oid) not in [int(x) for x in want]:
                return False
        else:
            if int(oid) != int(want):
                return False
    return True


@dataclass
class SymbolicPlanner:
    """
    Editable forward-ordered rule list (JSON). First matching rule can refine the
    proposed neural action. Rules never execute OS-level input; they only emit structured hints.
    """

    rules: List[RuleDict] = field(default_factory=list)

    @classmethod
    def from_json_file(cls, path: Union[str, Path]) -> "SymbolicPlanner":
        data = _load_json(path)
        if not isinstance(data, list):
            raise ValueError("Rules file must be a JSON array of rule objects")
        return cls(rules=list(data))

    def rule_applies(self, rule: Mapping[str, Any], task_text: str, neural_symbols: Mapping[str, Any]) -> bool:
        cond = rule.get("if") or {}
        if not isinstance(cond, dict):
            return False
        tl = task_text.lower()
        return _task_matches(cond, tl) and _object_matches(cond, neural_symbols)

    def plan(
        self,
        task_text: str,
        neural_symbols: Mapping[str, Any],
        state: Optional[MutableMapping[str, Any]] = None,
        neural_action: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Merge neural action with first matching symbolic rule.
        Returns a dict suitable for downstream execution (coords normalized 0..1).
        """
        state = dict(state or {})
        out: Dict[str, Any] = {
            "task_text": task_text,
            "symbols": dict(neural_symbols),
            "state": state,
            "rule_name": None,
            "action": dict(neural_action) if neural_action else {},
        }
        for rule in self.rules:
            if not self.rule_applies(rule, task_text, neural_symbols):
                continue
            then = rule.get("then") or {}
            out["rule_name"] = rule.get("name")
            action = dict(out["action"])
            if isinstance(then, dict):
                if "action_type" in then:
                    at = str(then["action_type"]).lower()
                    if at == "click":
                        action["type"] = "click"
                        if then.get("prefer") == "center_of_bbox" and "bbox_norm" in neural_symbols:
                            cx, cy, w, h = neural_symbols["bbox_norm"]
                            action["x"] = float(cx)
                            action["y"] = float(cy)
                    elif at == "key":
                        action["type"] = "keypress"
                        keyn = then.get("key_name", "enter")
                        action["key"] = keyn
            out["action"] = action
            break
        return out


_ACTION_TYPE_MAP = (
    "noop",
    "click",
    "key",
    "scroll_up",
    "scroll_down",
    "wait",
    "drag",
    "double_click",
)


def neural_logits_to_action_dict(
    action_type_logits: torch.Tensor,
    click_logits: torch.Tensor,
    key_logits: torch.Tensor,
    grid_h: int,
    grid_w: int,
    batch_idx: int = 0,
) -> Dict[str, Any]:
    """Convert model action heads to a serializable action (training/inference helper)."""
    at = int(action_type_logits[batch_idx].argmax().item())
    name = _ACTION_TYPE_MAP[at] if at < len(_ACTION_TYPE_MAP) else "noop"
    cell = int(click_logits[batch_idx].argmax().item())
    gh = max(1, grid_h)
    gw = max(1, grid_w)
    row, col = cell // gw, cell % gw
    x = (col + 0.5) / gw
    y = (row + 0.5) / gh
    kid = int(key_logits[batch_idx].argmax().item())
    if name == "click":
        return {"type": "click", "x": x, "y": y, "grid_cell": cell, "action_type_id": at}
    if name == "key":
        return {"type": "keypress", "key_id": kid, "action_type_id": at}
    return {"type": name, "action_type_id": at}
