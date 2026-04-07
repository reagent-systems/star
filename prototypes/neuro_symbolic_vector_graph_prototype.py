#!/usr/bin/env python3
"""
Neuro-symbolic vector–graph prototype (single file).

- **Symbolic:** explicit facts and rules, unification, forward chaining.
- **Neural:** Hugging Face ``datasets`` + ``transformers`` encoder trained with real
  gradients (relation classification). Confidences come from trained softmax, not simulations.
- **Hybrid:** promoted triples in a knowledge base plus per-entity CLS-derived
  embeddings in a vector–graph store (cosine retrieval).

**Default corpus:** ``SemEvalWorkshop/sem_eval_2010_task_8`` — 19-way relation
classification, sentences with ``<e1>...</e1>`` and ``<e2>...</e2>`` markup.

**Training policy (phase A):** train the classifier for K epochs, then run symbolic
graduation on held-out text using model probabilities.

Dependencies: ``torch``, ``numpy``, ``datasets``, ``transformers``.

Example::

    python prototypes/neuro_symbolic_vector_graph_prototype.py \\
        --epochs 2 --max-train-samples 800 --max-eval-samples 200 --device cpu
"""

from __future__ import annotations

import argparse
import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, BatchEncoding, get_linear_schedule_with_warmup

# =============================================================================
# Symbolic core
# =============================================================================


@dataclass(frozen=True, order=True)
class Symbol:
    name: str

    def __repr__(self) -> str:
        return f"Symbol({self.name!r})"


@dataclass(frozen=True, order=True)
class Variable:
    name: str

    def __repr__(self) -> str:
        return f"Variable({self.name!r})"


Atom = Symbol | Variable
Expression = Tuple[Any, ...]


def _is_var(x: Any) -> bool:
    return isinstance(x, Variable)


def _is_sym(x: Any) -> bool:
    return isinstance(x, Symbol)


def walk_atom(x: Atom, subst: Dict[Variable, Atom]) -> Atom:
    if _is_var(x):
        v: Any = x
        while _is_var(v) and v in subst:
            v = subst[v]
        if _is_var(v):
            return subst.get(v, v)
        return v
    return x


def apply_subst(expr: Expression, subst: Dict[Variable, Atom]) -> Expression:
    out: List[Any] = []
    for e in expr:
        if isinstance(e, tuple):
            out.append(apply_subst(e, subst))
        else:
            out.append(walk_atom(e, subst))
    return tuple(out)


def unify(
    a: Expression | Atom,
    b: Expression | Atom,
    subst: Optional[Dict[Variable, Atom]] = None,
) -> Optional[Dict[Variable, Atom]]:
    if subst is None:
        subst = {}

    if isinstance(a, tuple) and isinstance(b, tuple):
        if len(a) != len(b):
            return None
        s = dict(subst)
        for x, y in zip(a, b):
            res = unify(x, y, s)
            if res is None:
                return None
            s = res
        return s

    ua = walk_atom(a, subst) if isinstance(a, (Variable, Symbol)) else a
    ub = walk_atom(b, subst) if isinstance(b, (Variable, Symbol)) else b

    if _is_var(ua):
        if ua == ub:
            return subst
        if occurs_check(ua, ub, subst):
            return None
        out = dict(subst)
        out[ua] = ub  # type: ignore[assignment]
        return out
    if _is_var(ub):
        return unify(ub, ua, subst)
    if _is_sym(ua) and _is_sym(ub):
        return subst if ua.name == ub.name else None
    return None


def occurs_check(v: Variable, term: Atom, subst: Dict[Variable, Atom]) -> bool:
    """True if binding v -> term would create a cycle (occurs check for atoms)."""
    t = walk_atom(term, subst)
    if _is_var(t):
        return v == t
    if _is_sym(t):
        return False
    return False


@dataclass
class Rule:
    head: Expression
    body: Tuple[Expression, ...]


@dataclass
class KnowledgeBase:
    facts: Set[Expression] = field(default_factory=set)
    rules: List[Rule] = field(default_factory=list)

    def add_fact(self, fact: Expression) -> bool:
        if fact in self.facts:
            return False
        self.facts.add(fact)
        return True

    def add_rule(self, rule: Rule) -> None:
        self.rules.append(rule)

    def predicates_same_pair_conflict(self, fact: Expression) -> bool:
        if len(fact) != 3:
            return False
        p_new, s_new, o_new = fact
        if not (_is_sym(p_new) and _is_sym(s_new) and _is_sym(o_new)):
            return False
        for ex in self.facts:
            if len(ex) != 3:
                continue
            p, s, o = ex
            if not (_is_sym(p) and _is_sym(s) and _is_sym(o)):
                continue
            if s == s_new and o == o_new and p != p_new:
                return True
        return False

    def is_consistent_candidate(self, fact: Expression) -> bool:
        return not self.predicates_same_pair_conflict(fact)


class InferenceEngine:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def forward_chain(self, max_rounds: int = 32) -> int:
        added_total = 0
        for _ in range(max_rounds):
            batch: List[Expression] = []
            for rule in self.kb.rules:
                batch.extend(_derive_from_rule(self.kb.facts, rule))
            if not batch:
                break
            for f in batch:
                if self.kb.add_fact(f):
                    added_total += 1
        return added_total


def _derive_from_rule(facts: Set[Expression], rule: Rule) -> List[Expression]:
    out: List[Expression] = []
    if len(rule.body) == 0:
        if rule.head not in facts:
            out.append(rule.head)
        return out
    if len(rule.body) == 1:
        lit = rule.body[0]
        for fact in facts:
            subst = unify(lit, fact, {})
            if subst is None:
                continue
            head_i = apply_subst(rule.head, subst)
            if head_i not in facts:
                out.append(head_i)
        return out
    if len(rule.body) == 2:
        l1, l2 = rule.body
        for f1 in facts:
            s1 = unify(l1, f1, {})
            if s1 is None:
                continue
            for f2 in facts:
                s2 = unify(l2, f2, s1)
                if s2 is None:
                    continue
                head_i = apply_subst(rule.head, s2)
                if head_i not in facts:
                    out.append(head_i)
        return out
    return out


# =============================================================================
# Vector–graph hybrid store
# =============================================================================


@dataclass
class HybridNode:
    node_id: str
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorGraphKB:
    def __init__(self, embedding_dim: int) -> None:
        self.embedding_dim = embedding_dim
        self.nodes: Dict[str, HybridNode] = {}
        self.edges: List[Tuple[str, str, str, str]] = []

    def upsert_node(self, node_id: str, vec: np.ndarray, **meta: Any) -> None:
        v = np.asarray(vec, dtype=np.float64).reshape(-1)
        if v.shape[0] != self.embedding_dim:
            raise ValueError(f"expected dim {self.embedding_dim}, got {v.shape[0]}")
        if node_id in self.nodes:
            old = self.nodes[node_id]
            merged = (old.embedding + v) / 2.0
            self.nodes[node_id] = HybridNode(node_id, merged, {**old.metadata, **meta})
        else:
            self.nodes[node_id] = HybridNode(node_id, v.copy(), dict(meta))

    def add_edge(self, subj_id: str, rel_name: str, obj_id: str, provenance: str = "") -> None:
        self.edges.append((subj_id, rel_name, obj_id, provenance))

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))

    def neural_retrieve(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        q = np.asarray(query_vec, dtype=np.float64).reshape(-1)
        scored = [(nid, self._cosine(n.embedding, q)) for nid, n in self.nodes.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# =============================================================================
# Text utilities + metrics
# =============================================================================

E1_RE = re.compile(r"<e1>\s*(.*?)\s*</e1>", re.IGNORECASE | re.DOTALL)
E2_RE = re.compile(r"<e2>\s*(.*?)\s*</e2>", re.IGNORECASE | re.DOTALL)


def extract_e1_e2(sentence: str) -> Tuple[str, str]:
    m1 = E1_RE.search(sentence)
    m2 = E2_RE.search(sentence)
    if not m1 or not m2:
        raise ValueError("sentence missing <e1>/<e2> markup")
    e1 = re.sub(r"\s+", " ", m1.group(1)).strip()[:200]
    e2 = re.sub(r"\s+", " ", m2.group(1)).strip()[:200]
    if not e1 or not e2:
        raise ValueError("empty entity span")
    return e1, e2


def sanitize_entity_id(text: str) -> str:
    t = re.sub(r"\s+", "_", text.strip())[:120]
    t = re.sub(r"[^a-zA-Z0-9_\-]", "", t)
    return t or "entity"


def subset_examples(
    sentences: List[str],
    labels: List[int],
    max_n: Optional[int],
) -> Tuple[List[str], List[int]]:
    if max_n is None or max_n <= 0 or max_n >= len(sentences):
        return sentences, labels
    return sentences[:max_n], labels[:max_n]


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int], num_labels: int) -> float:
    f1s: List[float] = []
    for c in range(num_labels):
        tp = fp = fn = 0
        for t, p in zip(y_true, y_pred):
            if p == c and t == c:
                tp += 1
            elif p == c and t != c:
                fp += 1
            elif t == c and p != c:
                fn += 1
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(sum(f1s) / max(1, len(f1s)))


def accuracy_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if not y_true:
        return 0.0
    return float(sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true))


# =============================================================================
# Neural model
# =============================================================================


class RelationTransformer(nn.Module):
    def __init__(self, model_name: str, num_labels: int) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(cls))
        return logits, cls


class NeuralPerceiver:
    def __init__(
        self,
        model: RelationTransformer,
        tokenizer: AutoTokenizer,
        id2label: Dict[int, str],
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.device = device

    @torch.no_grad()
    def perceive_sentence(
        self,
        sentence: str,
        top_k: int = 1,
        min_prob: float = 0.0,
    ) -> List[Tuple[Expression, float, np.ndarray]]:
        self.model.eval()
        enc: BatchEncoding = self.tokenizer(
            sentence,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in enc.items()}
        logits, cls = self.model(batch["input_ids"], batch["attention_mask"])
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        cls_vec = cls.squeeze(0).detach().float().cpu().numpy()
        e1, e2 = extract_e1_e2(sentence)
        sid, oid = sanitize_entity_id(e1), sanitize_entity_id(e2)
        ranked = sorted(enumerate(probs.tolist()), key=lambda x: x[1], reverse=True)[: max(1, top_k)]
        out: List[Tuple[Expression, float, np.ndarray]] = []
        for idx, p in ranked:
            if p < min_prob:
                continue
            pred = Symbol(self.id2label[idx])
            expr: Expression = (pred, Symbol(sid), Symbol(oid))
            out.append((expr, float(p), cls_vec.copy()))
        return out


def collate_for_model(
    tokenizer: AutoTokenizer,
    sentences: List[str],
    labels: Optional[List[int]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    enc = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    batch: Dict[str, torch.Tensor] = {k: v.to(device) for k, v in enc.items()}
    if labels is not None:
        batch["labels"] = torch.tensor(labels, dtype=torch.long, device=device)
    return batch


def train_epochs(
    model: RelationTransformer,
    tokenizer: AutoTokenizer,
    train_sentences: List[str],
    train_labels: List[int],
    eval_sentences: List[str],
    eval_labels: List[int],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    num_labels: int,
    seed: int,
) -> Tuple[List[float], List[float], List[float]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train_sentences) / batch_size))
    total_steps = steps_per_epoch * epochs
    sched = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )
    loss_fn = nn.CrossEntropyLoss()

    train_losses: List[float] = []
    val_accs: List[float] = []
    val_f1s: List[float] = []

    for _ in range(epochs):
        model.train()
        order = list(range(len(train_sentences)))
        random.shuffle(order)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(order), batch_size):
            idxs = order[start : start + batch_size]
            sents = [train_sentences[i] for i in idxs]
            labs = [train_labels[i] for i in idxs]
            batch = collate_for_model(tokenizer, sents, labs, device)
            labels_bt = batch.pop("labels")
            opt.zero_grad(set_to_none=True)
            logits, _ = model(batch["input_ids"], batch["attention_mask"])
            loss = loss_fn(logits, labels_bt)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            sched.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        train_losses.append(epoch_loss / max(1, n_batches))

        metrics = evaluate_classifier(
            model,
            tokenizer,
            eval_sentences,
            eval_labels,
            device,
            batch_size,
            num_labels,
        )
        val_accs.append(metrics["accuracy"])
        val_f1s.append(metrics["macro_f1"])

    return train_losses, val_accs, val_f1s


@torch.no_grad()
def evaluate_classifier(
    model: RelationTransformer,
    tokenizer: AutoTokenizer,
    sentences: List[str],
    labels: List[int],
    device: torch.device,
    batch_size: int,
    num_labels: int,
) -> Dict[str, float]:
    model.eval()
    y_t: List[int] = []
    y_p: List[int] = []
    for start in range(0, len(sentences), batch_size):
        sents = sentences[start : start + batch_size]
        labs = labels[start : start + batch_size]
        batch = collate_for_model(tokenizer, sents, labs, device)
        labels_bt = batch.pop("labels")
        logits, _ = model(batch["input_ids"], batch["attention_mask"])
        pred = logits.argmax(dim=-1)
        y_t.extend(labels_bt.tolist())
        y_p.extend(pred.tolist())
    return {
        "accuracy": accuracy_score(y_t, y_p),
        "macro_f1": macro_f1(y_t, y_p, num_labels),
    }


# =============================================================================
# Neuro-symbolic trainer + dynamic heuristics
# =============================================================================


class NeuroSymbolicTrainer:
    """
    After the encoder is trained (phase A), promotes high-confidence predictions
    into the symbolic KB and runs forward chaining. Adjusts graduation threshold
    from validation accuracy gap and fact-growth heuristics.
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        vkb: VectorGraphKB,
        perceiver: NeuralPerceiver,
        inference: InferenceEngine,
        init_threshold: float = 0.65,
        threshold_min: float = 0.55,
        threshold_max: float = 0.92,
        overfitting_window: int = 3,
        embedding_dim: int = 768,
    ) -> None:
        self.kb = kb
        self.vkb = vkb
        self.perceiver = perceiver
        self.inference = inference
        self.confidence_threshold = init_threshold
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.overfitting_window = max(2, overfitting_window)
        self.embedding_dim = embedding_dim
        self.history: List[Tuple[float, float, int, int]] = []

    def promote_from_sentences(self, sentences: Iterable[str], provenance: str = "graduated") -> int:
        promoted = 0
        for sent in sentences:
            try:
                cands = self.perceiver.perceive_sentence(
                    sent,
                    top_k=1,
                    min_prob=self.confidence_threshold,
                )
            except (ValueError, RuntimeError):
                continue
            for expr, prob, emb in cands:
                if prob < self.confidence_threshold:
                    continue
                if not self.kb.is_consistent_candidate(expr):
                    continue
                if not self.kb.add_fact(expr):
                    continue
                pred, subj, obj = expr
                if _is_sym(pred) and _is_sym(subj) and _is_sym(obj):
                    subj_id, obj_id = subj.name, obj.name
                    half = emb[: self.embedding_dim] if emb.size >= self.embedding_dim else emb
                    pad = np.zeros(self.embedding_dim, dtype=np.float64)
                    pad[: min(len(half), self.embedding_dim)] = half[:(min(len(half), self.embedding_dim))]
                    self.vkb.upsert_node(subj_id, pad, prob=float(prob))
                    self.vkb.upsert_node(obj_id, pad, prob=float(prob))
                    self.vkb.add_edge(subj_id, pred.name, obj_id, provenance=provenance)
                promoted += 1
        return promoted

    def monitor_overfitting(self) -> None:
        """
        Use recent train vs validation accuracy gap and fact growth to adapt the
        graduation threshold (no simulated metrics).
        """
        if len(self.history) < 2:
            return
        window = self.history[-self.overfitting_window :]
        gaps = [h[0] - h[1] for h in window]
        avg_gap = sum(gaps) / len(gaps)
        growth = window[-1][3] - window[0][3]
        if avg_gap > 0.12 or growth > 400:
            self.confidence_threshold = min(self.threshold_max, self.confidence_threshold + 0.03)
        elif avg_gap < 0.04 and growth < 150:
            self.confidence_threshold = max(self.threshold_min, self.confidence_threshold - 0.02)

    def record_epoch(self, train_acc: float, val_acc: float, n_facts: int, n_edges: int) -> None:
        self.history.append((train_acc, val_acc, n_facts, n_edges))
        self.monitor_overfitting()


def abstraction_rules(related: Symbol, id2label: Dict[int, str]) -> List[Rule]:
    """Map each concrete SemEval relation to a generic related_to(X,Y) for chaining demos."""
    x = Variable("X")
    y = Variable("Y")
    rules: List[Rule] = []
    seen: Set[str] = set()
    for name in id2label.values():
        if name in seen:
            continue
        seen.add(name)
        pred = Symbol(name)
        rules.append(Rule((related, x, y), ((pred, x, y),)))
    return rules


def optional_family_seed(kb: KnowledgeBase) -> None:
    """Small disjoint demo: family rules + facts (symbol namespaced with fam_)."""
    A, B, C, D = Symbol("fam_Alice"), Symbol("fam_Bob"), Symbol("fam_Carol"), Symbol("fam_Dave")
    parent = Symbol("parent_of")
    male = Symbol("male")
    female = Symbol("female")
    brother = Symbol("brother_of")
    uncle = Symbol("uncle_of")
    x, y, z = Variable("FX"), Variable("FY"), Variable("FZ")
    kb.add_fact((parent, A, B))
    kb.add_fact((parent, B, C))
    kb.add_fact((male, B, B))
    kb.add_fact((female, A, A))
    kb.add_fact((male, D, D))
    kb.add_fact((parent, D, B))
    kb.add_rule(Rule((brother, x, y), ((parent, z, x), (parent, z, y), (male, x, x))))
    kb.add_rule(Rule((uncle, x, y), ((brother, x, z), (parent, z, y))))


def _self_test() -> None:
    x, y = Variable("X"), Variable("Y")
    a, b = Symbol("a"), Symbol("b")
    s0: Dict[Variable, Atom] = {}
    s1 = unify((Symbol("r"), x, b), (Symbol("r"), a, b), s0)
    assert s1 is not None and s1[x] == a
    expr = (Symbol("p"), a, b)
    kb = KnowledgeBase()
    kb.add_fact(expr)
    kb.add_rule(Rule((Symbol("q"), x, y), ((Symbol("p"), x, y),)))
    eng = InferenceEngine(kb)
    n = eng.forward_chain(max_rounds=4)
    assert n >= 1
    assert (Symbol("q"), a, b) in kb.facts


def build_id2label_from_features(train_ds: Any, label_field: str = "relation") -> Dict[int, str]:
    feat = train_ds.features[label_field]
    n = len(feat.names)
    return {i: feat.int2str(i) for i in range(n)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="SemEvalWorkshop/sem_eval_2010_task_8")
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default="test", help="SemEval release: use test as val (no official val split).")
    p.add_argument("--text-field", default="sentence")
    p.add_argument("--label-field", default="relation")
    p.add_argument("--model-name", default="distilbert-base-uncased")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-train-samples", type=int, default=0, help="0 = all")
    p.add_argument("--max-eval-samples", type=int, default=0, help="0 = all")
    p.add_argument("--max-grad-sentences", type=int, default=500, help="cap sentences scanned for graduation")
    p.add_argument("--init-threshold", type=float, default=0.55)
    p.add_argument("--family-demo", action="store_true", help="add toy family KB for extra forward-chaining demo")
    p.add_argument("--skip-train", action="store_true", help="load model weights only if you add checkpoint support later (not implemented)")
    args = p.parse_args()
    if args.skip_train:
        raise SystemExit("--skip-train is not supported in this prototype (train real weights first).")

    if __debug__:
        _self_test()

    train_ds = load_dataset(args.dataset, split=args.train_split)
    eval_ds = load_dataset(args.dataset, split=args.eval_split)
    text_f, lab_f = args.text_field, args.label_field
    train_text = [str(row[text_f]) for row in train_ds]
    train_labels = [int(row[lab_f]) for row in train_ds]
    eval_text = [str(row[text_f]) for row in eval_ds]
    eval_labels = [int(row[lab_f]) for row in eval_ds]

    m_tr = args.max_train_samples or None
    m_ev = args.max_eval_samples or None
    train_text, train_labels = subset_examples(train_text, train_labels, m_tr)
    eval_text, eval_labels = subset_examples(eval_text, eval_labels, m_ev)

    id2label = build_id2label_from_features(train_ds, lab_f)
    num_labels = len(id2label)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "Requested --device cuda, but this PyTorch build has no CUDA (CPU-only wheel).\n"
            "  Fix now: rerun with --device cpu\n"
            "  Fix GPU: install a CUDA-enabled torch from https://pytorch.org/get-started/locally/ "
            "into this venv, then use --device cuda."
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = RelationTransformer(args.model_name, num_labels)

    print(
        f"Training on {args.dataset} | train={len(train_text)} eval={len(eval_text)} "
        f"| labels={num_labels} | device={device}"
    )
    t_losses, v_accs, v_f1s = train_epochs(
        model,
        tokenizer,
        train_text,
        train_labels,
        eval_text,
        eval_labels,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        num_labels=num_labels,
        seed=args.seed,
    )
    for i, (tl, va, vf) in enumerate(zip(t_losses, v_accs, v_f1s)):
        print(f"  epoch {i + 1}: train_loss={tl:.4f} val_acc={va:.4f} val_macro_f1={vf:.4f}")

    hidden = model.encoder.config.hidden_size
    kb = KnowledgeBase()
    related = Symbol("related_to")
    for r in abstraction_rules(related, id2label):
        kb.add_rule(r)
    if args.family_demo:
        optional_family_seed(kb)

    vkb = VectorGraphKB(embedding_dim=hidden)
    perceiver = NeuralPerceiver(model, tokenizer, id2label, device)
    inference = InferenceEngine(kb)
    trainer = NeuroSymbolicTrainer(
        kb,
        vkb,
        perceiver,
        inference,
        init_threshold=args.init_threshold,
        embedding_dim=hidden,
    )

    grad_cap = args.max_grad_sentences if args.max_grad_sentences > 0 else len(eval_text)
    grad_sents = eval_text[:grad_cap]

    n_before = len(kb.facts)
    promoted = trainer.promote_from_sentences(grad_sents, provenance="eval_graduation")
    derived = inference.forward_chain(max_rounds=24)
    n_facts = len(kb.facts)

    train_metrics_end = evaluate_classifier(
        model, tokenizer, train_text[: min(len(train_text), 2000)], train_labels[: min(len(train_text), 2000)],
        device, args.batch_size, num_labels,
    )
    val_metrics_end = evaluate_classifier(
        model, tokenizer, eval_text, eval_labels, device, args.batch_size, num_labels
    )
    trainer.record_epoch(train_metrics_end["accuracy"], val_metrics_end["accuracy"], n_facts, len(vkb.edges))
    print(
        f"Graduation: promoted={promoted} derived_by_rules={derived} "
        f"kb_facts={n_facts} (was {n_before}) threshold={trainer.confidence_threshold:.3f}"
    )

    demo_queries = list(kb.facts)[:5]
    print("Sample symbolic facts (first 5):")
    for f in demo_queries:
        print(f"  {f}")

    if vkb.nodes:
        any_id = next(iter(vkb.nodes))
        q = vkb.nodes[any_id].embedding
        neigh = vkb.neural_retrieve(q, top_k=3)
        print("Vector retrieval (cosine) from one stored entity embedding:")
        for nid, sc in neigh:
            print(f"  {nid}: {sc:.4f}")
    else:
        print("Vector-graph store has no node embeddings yet (raise graduation threshold or train longer).")


if __name__ == "__main__":
    main()
