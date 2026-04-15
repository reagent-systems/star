#!/usr/bin/env python3
"""
Record screen frames + mouse/keyboard events for imitation learning (run locally with a display).

Requires: pip install pyautogui mss

Example::

    .venv/bin/python prototypes/gui_neurosymbolic/record_demo.py --out-dir demos/run1 --seconds 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--fps", type=float, default=2.0, help="Target capture rate (Hz)")
    p.add_argument("--monitor", type=int, default=0)
    return p.parse_args()


def main() -> None:
    try:
        import mss
        import mss.tools
    except ImportError as e:
        raise SystemExit("Install mss: pip install mss") from e
    try:
        import pyautogui
    except ImportError as e:
        raise SystemExit("Install pyautogui: pip install pyautogui") from e

    args = parse_args()
    out = Path(args.out_dir)
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    events: List[Dict[str, Any]] = []
    t0 = time.time()
    interval = 1.0 / max(0.25, args.fps)
    frame_idx = 0

    pyautogui.FAILSAFE = True

    with mss.mss() as sct:
        mon = sct.monitors[args.monitor + 1]
        while time.time() - t0 < args.seconds:
            shot = sct.grab(mon)
            png_path = frames_dir / f"frame_{frame_idx:06d}.png"
            mss.tools.to_png(shot.rgb, shot.size, output=str(png_path))
            pos = pyautogui.position()
            events.append(
                {
                    "t": time.time() - t0,
                    "frame": frame_idx,
                    "mouse": {"x": pos.x, "y": pos.y},
                }
            )
            frame_idx += 1
            time.sleep(interval)

    meta = {
        "seconds": args.seconds,
        "fps": args.fps,
        "monitor": args.monitor,
        "n_frames": frame_idx,
    }
    with open(out / "actions.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")
        for ev in events:
            f.write(json.dumps(ev) + "\n")

    print(f"Wrote {frame_idx} frames to {frames_dir} and log {out / 'actions.jsonl'}")


if __name__ == "__main__":
    main()
