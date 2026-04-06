"""
S-expressions as batched binary cons-trees (LISP-style graph) + adjacency for Graph Transformer.

Each tree is a list of nodes per row: CONS nodes have left/right child indices; atoms carry atom_id
into nn.Embedding. Padding uses mask; adjacency A[b,i,j] includes self-loops and undirected tree edges.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

# --- Vocabulary indices for atom embedding (must match model) ---
PAD_ATOM = 0
NIL_ATOM = 1
PLUS = 2
MINUS = 3
TIMES = 4
# digits 0..9 map to 5..14
DIGIT0 = 5


def _digit_id(d: int) -> int:
    return DIGIT0 + int(d)


@dataclass
class ConsTree:
    """Flat cons-tree for one expression: n nodes, root at index root_idx."""

    left: torch.Tensor  # [n] int64, -1 for atoms
    right: torch.Tensor  # [n] int64
    is_atom: torch.Tensor  # [n] bool
    atom_id: torch.Tensor  # [n] int64; meaningful where is_atom
    root_idx: int
    adjacency: torch.Tensor  # [n, n] float32, undirected + self-loops


def _add_edge(adj: torch.Tensor, i: int, j: int) -> None:
    adj[i, j] = 1.0
    adj[j, i] = 1.0


def _build_adjacency(n: int, left: List[int], right: List[int], is_atom: List[bool]) -> torch.Tensor:
    adj = torch.zeros(n, n, dtype=torch.float32)
    for i in range(n):
        adj[i, i] = 1.0
    for i in range(n):
        if is_atom[i]:
            continue
        li, ri = left[i], right[i]
        if li >= 0:
            _add_edge(adj, i, li)
        if ri >= 0:
            _add_edge(adj, i, ri)
    return adj


class _Node:
    __slots__ = ()


class _Atom(_Node):
    def __init__(self, atom_id: int):
        self.atom_id = atom_id


class _Cons(_Node):
    def __init__(self, left: _Node, right: _Node):
        self.left = left
        self.right = right


def _flatten(node: _Node) -> ConsTree:
    left: List[int] = []
    right: List[int] = []
    is_atom: List[bool] = []
    atom_id: List[int] = []

    def visit(n: _Node) -> int:
        idx = len(left)
        left.append(-1)
        right.append(-1)
        is_atom.append(False)
        atom_id.append(PAD_ATOM)
        if isinstance(n, _Atom):
            is_atom[idx] = True
            atom_id[idx] = n.atom_id
            left[idx] = -1
            right[idx] = -1
            return idx
        assert isinstance(n, _Cons)
        li = visit(n.left)
        ri = visit(n.right)
        left[idx] = li
        right[idx] = ri
        is_atom[idx] = False
        atom_id[idx] = PAD_ATOM
        return idx

    root = visit(node)
    n = len(left)
    adj = _build_adjacency(n, left, right, is_atom)
    return ConsTree(
        left=torch.tensor(left, dtype=torch.long),
        right=torch.tensor(right, dtype=torch.long),
        is_atom=torch.tensor(is_atom, dtype=torch.bool),
        atom_id=torch.tensor(atom_id, dtype=torch.long),
        root_idx=root,
        adjacency=adj,
    )


def _chain_list(elems: List[_Node]) -> _Node:
    """(a b c ...) as a . (b . (c . nil))."""
    if not elems:
        return _Atom(NIL_ATOM)
    if len(elems) == 1:
        return elems[0]
    return _Cons(elems[0], _chain_list(elems[1:]))


def _parse_atom(tok: str) -> _Atom:
    t = tok.strip()
    if t == "nil" or t == "()":
        return _Atom(NIL_ATOM)
    if t == "+":
        return _Atom(PLUS)
    if t == "-":
        return _Atom(MINUS)
    if t == "*":
        return _Atom(TIMES)
    if re.fullmatch(r"-?\d+", t):
        v = int(t)
        if not 0 <= v <= 9:
            raise ValueError(f"digit out of 0..9 range: {v}")
        return _Atom(_digit_id(v))
    raise ValueError(f"unknown atom: {tok}")


def _tokenize(s: str) -> List[str]:
    s = s.strip()
    out: List[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c in "()":
            out.append(c)
            i += 1
            continue
        j = i
        while j < len(s) and s[j] not in "() \t\n\r":
            j += 1
        out.append(s[i:j])
        i = j
    return out


def _parse_tokens(toks: List[str], pos: int) -> Tuple[_Node, int]:
    if pos >= len(toks):
        raise ValueError("unexpected end")
    t = toks[pos]
    if t == "(":
        pos += 1
        elems: List[_Node] = []
        while pos < len(toks) and toks[pos] != ")":
            n, pos = _parse_tokens(toks, pos)
            elems.append(n)
        if pos >= len(toks) or toks[pos] != ")":
            raise ValueError("missing )")
        pos += 1
        return _chain_list(elems), pos
    if t == ")":
        raise ValueError("unexpected )")
    return _parse_atom(t), pos + 1


def parse_sexpr(s: str) -> ConsTree:
    toks = _tokenize(s)
    node, pos = _parse_tokens(toks, 0)
    if pos != len(toks):
        raise ValueError(f"trailing tokens: {toks[pos:]}")
    return _flatten(node)


def eval_arithmetic(tree: ConsTree) -> float:
    """Evaluate (+ ...), (- a b), (* ...) on nested cons-trees; digits 0..9 only."""

    def ev(i: int) -> float:
        if tree.is_atom[i].item():
            aid = int(tree.atom_id[i].item())
            if aid == NIL_ATOM:
                raise ValueError("nil in eval")
            if DIGIT0 <= aid <= DIGIT0 + 9:
                return float(aid - DIGIT0)
            raise ValueError("non-numeric atom in eval")
        li = int(tree.left[i].item())
        ri = int(tree.right[i].item())
        if not tree.is_atom[li].item():
            raise ValueError("expected flat list form (op . rest)")
        op = int(tree.atom_id[li].item())
        if op not in (PLUS, MINUS, TIMES):
            raise ValueError("bad operator")

        def collect_rest(j: int) -> List[float]:
            if tree.is_atom[j].item():
                if int(tree.atom_id[j].item()) == NIL_ATOM:
                    return []
                return [ev(j)]
            lj = int(tree.left[j].item())
            rj = int(tree.right[j].item())
            return [ev(lj)] + collect_rest(rj)

        args = collect_rest(ri)
        if op == PLUS:
            return sum(args) if args else 0.0
        if op == TIMES:
            p = 1.0
            for a in args:
                p *= a
            return p
        if op == MINUS:
            if len(args) == 1:
                return -args[0]
            if len(args) == 2:
                return args[0] - args[1]
            raise ValueError("- takes 1 or 2 args")
        raise RuntimeError("unreachable")

    return ev(tree.root_idx)


def batch_trees(trees: Sequence[ConsTree], device: Optional[torch.device] = None) -> dict:
    """Pad a list of ConsTree into batched tensors."""
    if not trees:
        raise ValueError("empty batch")
    dev = device or torch.device("cpu")
    max_n = max(t.left.shape[0] for t in trees)
    bsz = len(trees)
    left = torch.full((bsz, max_n), -1, dtype=torch.long, device=dev)
    right = torch.full((bsz, max_n), -1, dtype=torch.long, device=dev)
    is_atom = torch.zeros((bsz, max_n), dtype=torch.bool, device=dev)
    atom_id = torch.full((bsz, max_n), PAD_ATOM, dtype=torch.long, device=dev)
    root_idx = torch.zeros(bsz, dtype=torch.long, device=dev)
    adjacency = torch.zeros(bsz, max_n, max_n, dtype=torch.float32, device=dev)
    mask = torch.zeros(bsz, max_n, dtype=torch.bool, device=dev)

    for bi, t in enumerate(trees):
        n = t.left.shape[0]
        left[bi, :n] = t.left.to(dev)
        right[bi, :n] = t.right.to(dev)
        is_atom[bi, :n] = t.is_atom.to(dev)
        atom_id[bi, :n] = t.atom_id.to(dev)
        root_idx[bi] = t.root_idx
        adjacency[bi, :n, :n] = t.adjacency.to(dev)
        mask[bi, :n] = True

    return {
        "left": left,
        "right": right,
        "is_atom": is_atom,
        "atom_id": atom_id,
        "root_idx": root_idx,
        "adjacency": adjacency,
        "mask": mask,
    }


__all__ = [
    "PAD_ATOM",
    "NIL_ATOM",
    "PLUS",
    "MINUS",
    "TIMES",
    "DIGIT0",
    "ConsTree",
    "parse_sexpr",
    "eval_arithmetic",
    "batch_trees",
]
