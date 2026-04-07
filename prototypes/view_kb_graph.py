#!/usr/bin/env python3
"""
Visualize a knowledge graph from ``--export-json`` output of
``neuro_symbolic_vector_graph_prototype.py``.

Uses **NetworkX + Matplotlib** (pure ``pip``; no system Graphviz required).

Example::

    python prototypes/neuro_symbolic_vector_graph_prototype.py \\
        --load-checkpoint checkpoints/yours.pt \\
        --export-json out/kb.json --max-grad-sentences 300

    python prototypes/view_kb_graph.py out/kb.json --out out/preview.png

    # Or from a .dot file (requires: pip install pydot && brew install graphviz)
    python prototypes/view_kb_graph.py --dot out/kb.dot --out out/from_dot.png
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
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
            yield s, p, o


def subsample_edges(edges: List[Tuple[str, str, str]], max_edges: int, seed: int) -> List[Tuple[str, str, str]]:
    if len(edges) <= max_edges:
        return edges
    rng = random.Random(seed)
    return rng.sample(edges, max_edges)


def _short_label(s: str, maxlen: int) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) <= maxlen:
        return s
    return s[: maxlen - 1] + "…"


def collapse_parallel_edges(
    edges: List[Tuple[str, str, str]],
    pred_preview: int,
    pred_chars: int,
) -> List[Tuple[str, str, str]]:
    """One arc per (subject, object); labels merge predicates (reduces clutter)."""
    groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for s, p, o in edges:
        groups[(s, o)].append(p)
    out: List[Tuple[str, str, str]] = []
    for (s, o) in sorted(groups.keys()):
        preds = groups[(s, o)]
        parts = [_short_label(p, pred_chars) for p in preds[:pred_preview]]
        lab = " · ".join(parts)
        rest = len(preds) - pred_preview
        if rest > 0:
            lab += f" (+{rest})"
        out.append((s, lab, o))
    return out


# -----------------------------------------------------------------------------
# Layout + draw
# -----------------------------------------------------------------------------


def _compute_pos(g: Any, layout: str, seed: int) -> Dict[Any, Tuple[float, ...]]:
    import networkx as nx

    n = g.number_of_nodes()
    if n == 0:
        return {}
    u = g.to_undirected()

    if layout == "circular":
        return nx.circular_layout(g, scale=2.5)

    if layout == "kamada":
        try:
            return nx.kamada_kawai_layout(u, scale=3.0)
        except Exception:
            layout = "spring"

    if layout == "spring":
        # Stronger repulsion than defaults — fewer overlapping nodes in dense graphs.
        k = max(1.2, 5.0 / (n ** 0.45))
        return nx.spring_layout(u, seed=seed, k=k, iterations=250, scale=3.0)

    # auto
    if n <= 90:
        try:
            return nx.kamada_kawai_layout(u, scale=3.0)
        except Exception:
            pass
    k = max(1.2, 5.0 / (n ** 0.45))
    return nx.spring_layout(u, seed=seed, k=k, iterations=250, scale=3.0)


def render_matplotlib(
    edges: List[Tuple[str, str, str]],
    out_path: str,
    show: bool,
    figsize: Tuple[float, float],
    label_max: int,
    layout: str,
    seed: int,
    collapse: bool,
    pred_preview: int,
    pred_chars: int,
    show_edge_labels: str,
    node_label_max: int,
) -> None:
    import matplotlib.pyplot as plt
    import networkx as nx

    raw = list(edges)
    if collapse:
        edges = collapse_parallel_edges(raw, pred_preview=pred_preview, pred_chars=pred_chars)
    else:
        edges = raw

    g = nx.DiGraph()
    for subj, pred, obj in edges:
        g.add_edge(subj, obj, label=_short_label(pred, label_max))

    if g.number_of_nodes() == 0:
        print("No nodes to draw (empty graph).", file=sys.stderr)
        sys.exit(1)

    pos = _compute_pos(g, layout=layout, seed=seed)
    n, m = g.number_of_nodes(), g.number_of_edges()

    node_size = int(max(280, min(2200, 6000 / math.sqrt(max(1, n)))))
    font_nodes = max(5, min(9, 13 - int(math.sqrt(max(1, n)) / 2)))
    edge_width = max(0.4, min(1.2, 3.5 / math.log2(m + 2)))
    edge_alpha = max(0.25, min(0.75, 1.2 - 0.25 * math.log10(m + 1)))

    plt.figure(figsize=figsize, dpi=140)
    ax = plt.gca()

    nx.draw_networkx_nodes(
        g, pos, node_color="#2c5282", node_size=node_size, edgecolors="#1a365d",
        linewidths=0.6, ax=ax,
    )
    labels = {nd: _short_label(str(nd), node_label_max) for nd in g.nodes}
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=font_nodes, font_color="white", ax=ax)

    nx.draw_networkx_edges(
        g, pos, edge_color="#4a5568", arrows=True, arrowsize=12, width=edge_width,
        alpha=edge_alpha, ax=ax, connectionstyle="arc3,rad=0.12", min_source_margin=18, min_target_margin=18,
    )

    auto_labels = (
        show_edge_labels == "auto" and m <= 55 and (collapse or len(raw) <= 70)
    ) or show_edge_labels == "on"
    if auto_labels:
        edge_labels = nx.get_edge_attributes(g, "label")
        efs = max(4, min(7, 9 - int(math.sqrt(max(1, m)) / 3)))
        nx.draw_networkx_edge_labels(
            g, pos, edge_labels=edge_labels, font_size=efs, font_color="#2d3748",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.88),
            ax=ax,
        )

    ax.set_axis_off()
    mode = "collapsed" if collapse else "full"
    ax.set_title(f"KB graph  |  nodes={n}  edges_drawn={m}  triples={len(raw)}  ({mode})", fontsize=11, pad=12)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out_path}")
    if show:
        plt.show()
    else:
        plt.close()


# -----------------------------------------------------------------------------
# Optional: render .dot via pydot
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
    p.add_argument("--max-edges", type=int, default=400, help="subsample raw triples if huge")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--show", action="store_true", help="open interactive window after save")
    p.add_argument(
        "--figscale",
        type=float,
        default=14.0,
        help="figure width in inches (height is about 0.72 of width)",
    )
    p.add_argument("--label-max", type=int, default=36, help="max chars stored on each drawn edge label")
    p.add_argument(
        "--layout",
        choices=("auto", "spring", "kamada", "circular"),
        default="auto",
        help="node placement: auto tries kamada for small graphs, spring with strong repulsion otherwise",
    )
    p.add_argument(
        "--no-collapse",
        action="store_true",
        help="draw one arrow per triple (often messy); default merges same (subj,obj) into one edge",
    )
    p.add_argument("--pred-preview", type=int, default=2, help="when collapsing, max relation names shown before (+N)")
    p.add_argument("--pred-chars", type=int, default=20, help="max chars per relation snippet when collapsing")
    p.add_argument(
        "--edge-labels",
        choices=("auto", "on", "off"),
        default="auto",
        help="auto hides edge text when the graph is crowded",
    )
    p.add_argument("--node-label-max", type=int, default=18, help="max chars per node name")
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
    print(f"Drawing from {len(edges)} triples ({data.get('counts', {})})")

    import matplotlib

    matplotlib.use("TkAgg" if args.show else "Agg")

    n_est = len({s for s, _, _ in edges} | {o for _, _, o in edges})
    w = max(args.figscale, 10.0 + math.sqrt(max(1, n_est)) * 0.65)
    h = w * 0.72

    render_matplotlib(
        edges,
        args.out,
        args.show,
        (w, h),
        args.label_max,
        layout=args.layout,
        seed=args.seed,
        collapse=not args.no_collapse,
        pred_preview=args.pred_preview,
        pred_chars=args.pred_chars,
        show_edge_labels=args.edge_labels,
        node_label_max=args.node_label_max,
    )


if __name__ == "__main__":
    main()
