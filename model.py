"""
Graph Transformer over cons-tree nodes: multi-head attention with structural bias from adjacency.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from lisp_graph import DIGIT0, PAD_ATOM

# Atom embedding vocab: PAD, NIL, +, -, *, digits 0-9
NUM_ATOM_TYPES = DIGIT0 + 10  # exclusive upper bound


class GraphTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self._struct_bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: [B, N, D]
        adjacency: [B, N, N] float (0/1 tree + self-loops)
        attn_mask: [B, N, N] float, -inf where masked
        """
        b, n, d = x.shape
        residual = x
        x = self.norm1(x)

        qkv = self.qkv(x).reshape(b, n, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        struct = self._struct_bias * adjacency.unsqueeze(1)
        scores = scores + struct + attn_mask.unsqueeze(1)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, d)
        out = self.out_proj(out)
        out = self.dropout(out)
        x = residual + out

        residual2 = x
        x = self.norm2(x)
        x = residual2 + self.ff(x)
        return x


def _build_attn_mask(mask: torch.Tensor) -> torch.Tensor:
    """mask: [B, N] True = valid node. Invalid query rows attend only to self (avoids all -inf softmax)."""
    b, n = mask.shape
    device = mask.device
    dq = mask.unsqueeze(2)
    dk = mask.unsqueeze(1)
    pair_ok = dq & dk
    eye = torch.eye(n, dtype=torch.bool, device=device).unsqueeze(0)
    pad_query_self = (~mask).unsqueeze(2) & eye
    ok = pair_ok | pad_query_self
    return torch.where(
        ok,
        torch.zeros(b, n, n, device=device, dtype=torch.float32),
        torch.full((b, n, n), float("-inf"), device=device, dtype=torch.float32),
    )


class NeuroSymbolicGraphTransformer(nn.Module):
    """
    Embeds cons-tree nodes (atoms + CONS), runs Graph Transformer layers, pools at root, predicts scalar.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.atom_embed = nn.Embedding(NUM_ATOM_TYPES, d_model, padding_idx=PAD_ATOM)
        self.cons_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cons_embed, std=0.02)

        self.in_proj = nn.Linear(d_model, d_model)
        self.layers = nn.ModuleList(
            [GraphTransformerLayer(d_model, n_heads, dropout=dropout) for _ in range(n_layers)]
        )
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        is_atom = batch["is_atom"]
        atom_id = batch["atom_id"]
        adjacency = batch["adjacency"]
        mask = batch["mask"]
        root_idx = batch["root_idx"]

        b, n = mask.shape
        device = mask.device

        ae = self.atom_embed(atom_id.clamp(min=0, max=NUM_ATOM_TYPES - 1))
        ce = self.cons_embed.expand(b, n, self.d_model)
        h = torch.where(is_atom.unsqueeze(-1), ae, ce)
        h = self.in_proj(h)

        attn_base = _build_attn_mask(mask)

        for layer in self.layers:
            h = layer(h, adjacency, attn_base)

        idx = root_idx.view(b, 1, 1).expand(b, 1, self.d_model)
        pooled = h.gather(1, idx).squeeze(1)
        out = self.head(pooled).squeeze(-1)
        return out


__all__ = ["NUM_ATOM_TYPES", "GraphTransformerLayer", "NeuroSymbolicGraphTransformer"]
