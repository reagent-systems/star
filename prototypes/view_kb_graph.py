#!/usr/bin/env python3
"""
Visualize a knowledge graph from ``--export-json`` output of
``neuro_symbolic_vector_graph_prototype.py``.

Uses **NetworkX + Matplotlib** (pure ``pip``; no system Graphviz required).

Example::

    python prototypes/neuro_symbolic_vector_graph_prototype.py \\
        --load-checkpoint checkpoints/yours.pt \\
        --export-json out/kb.json --max-grad-sentences 300

    python prototypes/view_kb_graph.py out/kb.json --out out/preview.png --show

    # Or from a .dot file produced by --export-dot (requires: pip install pydot && brew install graphviz)
    python prototypes/view_kb_graph.py --dot out/kb.dot --out out/from_dot.png
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from typing import Any, Dict, Iterator, List, Tuple

# -----------------------------------------------------------------------------
# Load triples from JSON export
# -----------------------------------------------------------------------------


def _sym_name(x: Any) -> str | None:
    if isinstance(x, dict) and "sym" in x:
        return str(x["sym"])
    return None


def iter_triples_from_export(data: Dict[str, Any]) -> Iterator[Tuple[str, str, str]]:
    for e in data.get("graph_edges") or []:
        if isinstance(e, dict) and all(k in e for k in ("subj", "relation", "obj")):
            yield str(e["subj"]), str(e["relation"]), str(e["obj"])

    for fact in data.get("facts") or []:
        if not isinstance(fact, list) or len(fact) != 3:
            continue
        p, s, o = _sym_name(fact[0]), _sym_name(fact[1]), _sym_name(fact[2])
        if p and s and o:
            yield s, p, o  # subj --predicate--> obj


def subsample_edges(edges: List[Tuple[str, str, str]], max_edges: int, seed: int) -> List[Tuple[str, str, str]]:
    if len(edges) <= max_edges:
        return edges
    rng = random.Random(seed)
    return rng.sample(edges, max_edges)


# -----------------------------------------------------------------------------
# Draw with NetworkX + Matplotlib
# -----------------------------------------------------------------------------


def render_matplotlib(
    edges: List[Tuple[str, str, str]],
    out_path: str,
    show: bool,
    figsize: Tuple[float, float],
    label_max: int,
) -> None:
    import matplotlib.pyplot as plt
    import networkx as nx

    g = nx.DiGraph()
    for subj, pred, obj in edges:
        g.add_edge(subj, obj, label=_short_label(pred, label_max))

    if g.number_of_nodes() == 0:
        print("No nodes to draw (empty graph).", file=sys.stderr)
        sys.exit(1)

    pos = nx.spring_layout(g, seed=42, k=0.9 / max(1, g.number_of_nodes() ** 0.5))
    plt.figure(figsize=figsize, dpi=120)
    ax = plt.gca()
    nx.draw_networkx_nodes(g, pos, node_color="#4a90d9", node_size=800, ax=ax)
    nx.draw_networkx_labels(g, pos, font_size=7, ax=ax)
    nx.draw_networkx_edges(g, pos, edge_color="#333333", arrows=True, arrowsize=14, ax=ax, connectionstyle="arc3,rad=0.08")
    edge_labels = nx.get_edge_attributes(g, "label")
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=5, ax=ax)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Wrote {out_path}")
    if show:
        plt.show()
    else:
        plt.close()


def _short_label(s: str, maxlen: int) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) <= maxlen:
        return s
    return s[: maxlen - 1] + "…"


# -----------------------------------------------------------------------------
# Optional: render .dot via pydot (needs system Graphviz ``dot``)
# -----------------------------------------------------------------------------


def render_dot_file(dot_path: str, out_path: str) -> None:
    try:
        import pydot
    except ImportError as e:
        raise SystemExit(
            "Reading .dot requires: pip install pydot\n"
            "and Graphviz installed so `dot` is on PATH (e.g. brew install graphviz)."
        ) from e
    graphs = pydot.graph_from_dot_file(dot_path)
    if not graphs:
        raise SystemExit(f"Could not parse DOT file: {dot_path}")
    g = graphs[0]
    ext = out_path.rsplit(".", 1)[-1].lower()
    fmt = {"png": "png", "svg": "svg", "pdf": "pdf"}.get(ext, "png")
    g.write(out_path, format=fmt)
    print(f"Wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("json_path", nargs="?", help="path to kb.json from --export-json")
    src.add_argument("--dot", metavar="PATH", help="path to .dot from --export-dot (uses pydot + system dot)")
    p.add_argument("--out", default="kb_graph.png", help="output image path (.png / .svg / .pdf for --dot)")
    p.add_argument("--max-edges", type=int, default=400, help="subsample if graph is huge")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--show", action="store_true", help="open interactive window after save (needs GUI backend)")
    p.add_argument("--figscale", type=float, default=12.0, help="matplotlib figure width in inches")
    p.add_argument("--label-max", type=int, default=28, help="max chars on edge labels")
    args = p.parse_args()

    if args.dot:
        render_dot_file(args.dot, args.out)
        return

    assert args.json_path
    with open(args.json_path, encoding="utf-8") as f:
        data = json.load(f)

    edges = list(iter_triples_from_export(data))
    if not edges:
        print("No edges found in JSON (need graph_edges or symbol triple facts).", file=sys.stderr)
        sys.exit(1)

    edges = subsample_edges(edges, args.max_edges, args.seed)
    print(f"Drawing {len(edges)} edges ({data.get('counts', {})})")

    import matplotlib

    matplotlib.use("TkAgg" if args.show else "Agg")

    h = args.figscale * 0.75
    render_matplotlib(edges, args.out, args.show, (args.figscale, h), args.label_max)


if __name__ == "__main__":
    main()
