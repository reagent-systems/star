"""
Schema stubs for adapting public GUI datasets (GUI Odyssey, Show-UI, ScreenSpot, RICO, GUI Act).

Each loader should map rows to the canonical training dict used by ``dataset.collate_batch``:

- ``image``: CHW float tensor or uint8 (handled in Dataset)
- ``task_token_ids``: long tensor ``[task_max_len]``
- ``thought_ids``, ``object_type``, ``bbox_bins``, ``action_type``, ``click_cell``, ``key_id``
- ``task_text``: str

Example (pseudo)::

    def row_to_sample(row: dict, cfg: ModelConfig) -> dict:
        ...
"""

from __future__ import annotations

from typing import Any, Dict, TypedDict


class CanonicalSample(TypedDict, total=False):
    image: Any
    task_token_ids: Any
    thought_ids: Any
    object_type: Any
    bbox_bins: Any
    action_type: Any
    click_cell: Any
    key_id: Any
    task_text: str


DATASET_IDS = (
    "gui_odyssey",
    "show_ui",
    "screen_spot",
    "rico",
    "gui_act",
)


def describe_schema() -> Dict[str, Any]:
    """Return a machine-readable description for future dataset adapters."""
    return {
        "datasets": list(DATASET_IDS),
        "notes": "Map bbox in pixels to bbox_bins using image size and cfg.bbox_bins; "
        "map click (x,y) to grid cell row*W+col.",
    }
