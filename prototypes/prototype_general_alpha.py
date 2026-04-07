#!/usr/bin/env python3
"""
Generalized Neuro-Symbolic Knowledge Graph Builder (prototype_general_alpha)

A dataset-agnostic neuro-symbolic system that discovers relationships in ANY text
dataset using zero-shot classification — NO training required.

Architecture:
  1. Load any HuggingFace dataset (streaming supported)
  2. Extract concepts/entities from text via lightweight NLP (no spaCy needed)
  3. Classify relationships between entity pairs using zero-shot NLI models
  4. Graduate high-confidence predictions into a symbolic Knowledge Base
  5. Run forward-chaining inference to derive new facts
  6. Export as JSON/Graphviz, query interactively

Example usage::

    python prototypes/prototype_general_alpha.py \\
        --dataset grandsmile/Generative_Coding_Dataset \\
        --text-field question \\
        --label-field tags \\
        --streaming \\
        --max-samples 200 \\
        --device cuda \\
        --min-confidence 0.3 \\
        --export-json out/kb_coding.json \\
        --export-dot out/kb_coding.dot \\
        --trace-inference

Dependencies: torch, numpy, datasets, transformers, tqdm
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, pipeline

# =============================================================================
# Symbolic Core  (reused from neuro_symbolic_vector_graph_prototype.py)
# =============================================================================


@dataclass(frozen=True, order=True)
class Symbol:
    """A ground (constant) symbol in the symbolic KB."""
    name: str

    def __repr__(self) -> str:
        return f"Symbol({self.name!r})"


@dataclass(frozen=True, order=True)
class Variable:
    """A logic variable, bound during unification."""
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
    """Chase variable bindings until we reach a symbol or unbound var."""
    if _is_var(x):
        v: Any = x
        while _is_var(v) and v in subst:
            v = subst[v]
        if _is_var(v):
            return subst.get(v, v)
        return v
    return x


def apply_subst(expr: Expression, subst: Dict[Variable, Atom]) -> Expression:
    """Apply a substitution to every atom in a nested expression."""
    out: List[Any] = []
    for e in expr:
        if isinstance(e, tuple):
            out.append(apply_subst(e, subst))
        else:
            out.append(walk_atom(e, subst))
    return tuple(out)


def occurs_check(v: Variable, term: Atom, subst: Dict[Variable, Atom]) -> bool:
    """True if binding v -> term would create a cycle."""
    t = walk_atom(term, subst)
    if _is_var(t):
        return v == t
    return False


def unify(
    a: Expression | Atom,
    b: Expression | Atom,
    subst: Optional[Dict[Variable, Atom]] = None,
) -> Optional[Dict[Variable, Atom]]:
    """Unify two expressions/atoms, returning a substitution or None."""
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
        out[ua] = ub
        return out
    if _is_var(ub):
        return unify(ub, ua, subst)
    if _is_sym(ua) and _is_sym(ub):
        return subst if ua.name == ub.name else None
    return None


@dataclass
class Rule:
    """A Horn clause: head :- body[0], body[1], ..."""
    head: Expression
    body: Tuple[Expression, ...]


@dataclass
class KnowledgeBase:
    """Symbolic fact store with consistency checking."""
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
        """Check if another predicate already exists for same (subj, obj) pair."""
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
    """Forward-chaining inference over a KnowledgeBase."""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def forward_chain(self, max_rounds: int = 32, trace: Optional[List[str]] = None) -> int:
        """
        Forward chaining. If ``trace`` is a list, append a human-readable line for each
        newly derived fact (rule head + which rule body matched).
        """
        added_total = 0
        for rnum in range(max_rounds):
            staged: List[Tuple[Expression, str]] = []
            for rule in self.kb.rules:
                derived = _derive_from_rule(self.kb.facts, rule)
                for head_i in derived:
                    staged.append((head_i, str(rule)))
            if not staged:
                break
            seen: Set[Expression] = set()
            for fact, rule_str in staged:
                if fact in seen:
                    continue
                seen.add(fact)
                if self.kb.add_fact(fact):
                    added_total += 1
                    if trace is not None:
                        trace.append(f"round {rnum + 1}: {fact}\n  <- rule {rule_str}")
        return added_total


def _derive_from_rule(facts: Set[Expression], rule: Rule) -> List[Expression]:
    """Derive new facts from a rule given current facts."""
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
# Vector–Graph Hybrid Store  (reused from original)
# =============================================================================


@dataclass
class HybridNode:
    """A node in the vector-graph store with an embedding and metadata."""
    node_id: str
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorGraphKB:
    """Hybrid vector + graph store: cosine retrieval over entity embeddings."""

    def __init__(self, embedding_dim: int) -> None:
        self.embedding_dim = embedding_dim
        self.nodes: Dict[str, HybridNode] = {}
        self.edges: List[Tuple[str, str, str, str]] = []  # (subj, rel, obj, provenance)

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
# Text Utilities
# =============================================================================

# Stopwords for concept extraction (common English words to skip)
STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "shall should may might can could of in to for on with at by from as into "
    "through during before after above below between under over about up out off "
    "then than that this these those it its he she they we you i me him her us "
    "them my your his their our who what which where when how why all each every "
    "both few more most other some such no nor not only own same so too very much "
    "just also already still even now here there again further once and but or if "
    "while because until although though since unless whether either neither yet "
    "however therefore moreover nevertheless furthermore meanwhile instead "
    "get got getting make makes made let go goes went come came take takes took "
    "give given write writes wrote read reads return returns use uses used find "
    "finds found print prints need needs try tries keep keeps".split()
)

# Regex patterns for noun phrase extraction
# Match: "adjective? noun noun?" patterns — 2-3 word technical phrases
NP_PATTERN = re.compile(
    r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+"    # capitalized multi-word (e.g. "Binary Search")
    r"|[a-z]+(?:[-_][a-z]+)+"                   # hyphenated/underscored terms
    r"|[a-z]{3,}\s+[a-z]{3,}(?:\s+[a-z]{3,})?)" # 2-3 word lowercase phrases
    r"\b",
    re.UNICODE,
)

# Single important words (programming/tech terms often lowercase)
WORD_PATTERN = re.compile(r"\b[a-zA-Z][a-zA-Z0-9]{2,}\b")


def sanitize_entity_id(text: str) -> str:
    """Normalize text to a safe entity identifier."""
    t = re.sub(r"\s+", "_", text.strip().lower())[:120]
    t = re.sub(r"[^a-zA-Z0-9_\-]", "", t)
    return t or "entity"


def extract_concepts(text: str, max_concepts: int = 8) -> List[str]:
    """
    Extract key concepts from text using lightweight NLP.

    Strategy:
    1. Extract multi-word noun phrases (capitalized, hyphenated, etc.)
    2. Extract frequent meaningful single words
    3. Rank by TF (term frequency in document) and length
    4. Deduplicate and return top N

    Args:
        text: Input text string
        max_concepts: Maximum number of concepts to return

    Returns:
        List of concept strings, ordered by relevance score
    """
    if not text or len(text.strip()) < 5:
        return []

    # Clean text: remove code blocks, URLs, special chars for NLP
    clean = re.sub(r"```[\s\S]*?```", " ", text)
    clean = re.sub(r"`[^`]+`", lambda m: m.group(0).strip("`"), clean)
    clean = re.sub(r"https?://\S+", "", clean)
    clean = re.sub(r"[<>{}()\[\]=;:,\"\'/\\|&^%$#@!~*+]", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()

    if len(clean) < 5:
        return []

    # Count all meaningful words for TF scoring
    word_freq: Dict[str, int] = collections.Counter()
    for w in WORD_PATTERN.findall(clean):
        wl = w.lower()
        if wl not in STOPWORDS and len(wl) >= 3:
            word_freq[wl] += 1

    # Extract multi-word phrases
    phrases: Dict[str, int] = collections.Counter()
    for m in NP_PATTERN.finditer(clean):
        phrase = m.group(0).strip()
        pl = phrase.lower()
        words = pl.split()
        # Skip if too many stopwords (content words must be majority)
        content_words = [w for w in words if w not in STOPWORDS and len(w) >= 3]
        if len(content_words) < max(1, math.ceil(len(words) * 0.6)):
            continue
        # Trim leading/trailing stopwords
        while words and words[0] in STOPWORDS:
            words.pop(0)
        while words and words[-1] in STOPWORDS:
            words.pop()
        if not words:
            continue
        trimmed = " ".join(words)
        if len(trimmed) >= 3:
            phrases[trimmed] += 1

    # Score concepts: phrase length bonus + frequency
    scored: Dict[str, float] = {}

    # Multi-word phrases get priority
    for phrase, count in phrases.items():
        n_words = len(phrase.split())
        score = count * (1.0 + 0.5 * n_words)  # bonus for multi-word
        scored[phrase] = score

    # Single important words (high frequency or technical-looking)
    for word, count in word_freq.most_common(max_concepts * 3):
        if word not in scored:
            # Check it's not a substring of an existing phrase
            is_part_of_phrase = any(word in p.split() for p in scored)
            if not is_part_of_phrase:
                scored[word] = count * 0.8  # slight penalty vs phrases

    # Sort by score, take top N
    ranked = sorted(scored.items(), key=lambda x: (-x[1], x[0]))
    concepts = [c for c, _ in ranked[:max_concepts]]

    return concepts


def generate_entity_pairs(
    concepts: List[str], max_pairs: int = 10
) -> List[Tuple[str, str]]:
    """
    Generate entity pairs from extracted concepts for relation classification.

    Pairs the top concepts with each other, limiting total pairs.
    """
    if len(concepts) < 2:
        return []
    pairs = list(itertools.combinations(concepts[:max_pairs], 2))
    return pairs[:max_pairs]


# =============================================================================
# Zero-Shot Relation Discovery
# =============================================================================

# Default domain-agnostic relation types
DEFAULT_RELATIONS = [
    "is a type of",
    "is related to",
    "is used for",
    "is part of",
    "requires",
    "is similar to",
    "causes",
    "depends on",
    "is opposite of",
]


class ZeroShotRelationDiscoverer:
    """
    Discovers relationships between entity pairs using a pre-trained
    zero-shot NLI classifier. NO TRAINING required.

    Uses the HuggingFace zero-shot-classification pipeline with models like
    facebook/bart-large-mnli or cross-encoder/nli-deberta-v3-base.
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        candidate_relations: Optional[List[str]] = None,
        device: str = "cpu",
        min_confidence: float = 0.3,
    ):
        """
        Args:
            model_name: HuggingFace model for zero-shot classification
            candidate_relations: List of candidate relation type strings
            device: torch device string
            min_confidence: Minimum confidence to accept a relation
        """
        self.model_name = model_name
        self.candidate_relations = candidate_relations or DEFAULT_RELATIONS
        self.min_confidence = min_confidence
        self.device_str = device

        print(f"Loading zero-shot classifier: {model_name} (device={device})")
        # Map device: pipeline expects int for GPU or -1 for CPU
        dev_arg = -1
        if device.startswith("cuda"):
            if ":" in device:
                dev_arg = int(device.split(":")[1])
            else:
                dev_arg = 0
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=dev_arg,
        )
        print(f"  Loaded. Candidate relations: {self.candidate_relations}")

    def classify_relation(
        self,
        entity1: str,
        entity2: str,
        context: str = "",
    ) -> Tuple[str, float]:
        """
        Classify the relationship between two entities.

        Args:
            entity1: First entity/concept string
            entity2: Second entity/concept string
            context: Optional surrounding text for context

        Returns:
            (relation_label, confidence) tuple
        """
        # Construct a premise that presents the relationship
        if context:
            premise = f"{context} The relationship between '{entity1}' and '{entity2}'."
        else:
            premise = f"'{entity1}' and '{entity2}'."

        # Construct hypothesis templates
        hypotheses = [
            f"'{entity1}' {rel} '{entity2}'" for rel in self.candidate_relations
        ]

        result = self.classifier(
            premise,
            candidate_labels=self.candidate_relations,
            hypothesis_template="{}.",
            multi_label=False,
        )

        best_label = result["labels"][0]
        best_score = result["scores"][0]

        return best_label, float(best_score)

    def batch_classify(
        self,
        entity_pairs: List[Tuple[str, str]],
        contexts: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> List[Tuple[str, str, str, float]]:
        """
        Classify relationships for a batch of entity pairs.

        Args:
            entity_pairs: List of (entity1, entity2) tuples
            contexts: Optional list of context strings (one per pair)
            show_progress: Whether to show tqdm progress bar

        Returns:
            List of (entity1, entity2, relation, confidence) tuples
            (only those above min_confidence)
        """
        results: List[Tuple[str, str, str, float]] = []
        if not entity_pairs:
            return results

        bar = tqdm(
            entity_pairs,
            desc="  Classifying relations",
            unit="pair",
            disable=not show_progress,
            dynamic_ncols=True,
        )

        for i, (e1, e2) in enumerate(bar):
            ctx = contexts[i] if contexts else ""
            try:
                rel, conf = self.classify_relation(e1, e2, ctx)
                if conf >= self.min_confidence:
                    results.append((e1, e2, rel, conf))
                if show_progress:
                    bar.set_postfix(
                        found=len(results),
                        last_rel=rel[:20] if rel else "",
                        conf=f"{conf:.2f}",
                    )
            except Exception as exc:
                # Skip problematic pairs silently
                continue

        return results


# =============================================================================
# Embedding Provider  (for VectorGraphKB nodes)
# =============================================================================


class EmbeddingProvider:
    """
    Provides entity embeddings using a pre-trained transformer encoder.
    Uses mean pooling of the last hidden state.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased", device: str = "cpu"):
        self.device = torch.device(device)
        print(f"Loading embedding model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.embedding_dim = self.model.config.hidden_size
        print(f"  Embedding dim: {self.embedding_dim}")

    @torch.no_grad()
    def embed(self, text: str) -> np.ndarray:
        """Get embedding vector for a text string."""
        enc = self.tokenizer(
            text, truncation=True, max_length=128, return_tensors="pt"
        )
        batch = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**batch)
        # Mean pooling
        mask = batch["attention_mask"].unsqueeze(-1).float()
        summed = (out.last_hidden_state * mask).sum(dim=1)
        counted = mask.sum(dim=1).clamp(min=1e-8)
        vec = (summed / counted).squeeze(0).cpu().numpy().astype(np.float64)
        return vec

    @torch.no_grad()
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for a batch of texts."""
        if not texts:
            return []
        enc = self.tokenizer(
            texts, truncation=True, max_length=128, padding=True, return_tensors="pt"
        )
        batch = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**batch)
        mask = batch["attention_mask"].unsqueeze(-1).float()
        summed = (out.last_hidden_state * mask).sum(dim=1)
        counted = mask.sum(dim=1).clamp(min=1e-8)
        vecs = (summed / counted).cpu().numpy().astype(np.float64)
        return [vecs[i] for i in range(len(texts))]


# =============================================================================
# Knowledge Graph Builder
# =============================================================================


class KnowledgeGraphBuilder:
    """
    Orchestrates concept extraction, relation discovery, KB graduation,
    and inference for building a knowledge graph from any text dataset.
    """

    def __init__(
        self,
        discoverer: ZeroShotRelationDiscoverer,
        embedder: EmbeddingProvider,
        min_confidence: float = 0.3,
        max_concepts_per_doc: int = 8,
        max_pairs_per_doc: int = 10,
    ):
        self.discoverer = discoverer
        self.embedder = embedder
        self.min_confidence = min_confidence
        self.max_concepts = max_concepts_per_doc
        self.max_pairs = max_pairs_per_doc

        # Symbolic stores
        self.kb = KnowledgeBase()
        self.vkb = VectorGraphKB(embedding_dim=embedder.embedding_dim)
        self.inference = InferenceEngine(self.kb)

        # Stats
        self.total_docs = 0
        self.total_concepts = 0
        self.total_pairs_classified = 0
        self.total_relations_found = 0
        self.total_promoted = 0

        # Staging area: (e1, e2, relation, confidence, context)
        self.staged_relations: List[Tuple[str, str, str, float, str]] = []

        # Track all discovered concepts
        self.concept_frequency: Dict[str, int] = collections.Counter()

        # Install default abstraction rules
        self._install_abstraction_rules(discoverer.candidate_relations)

    def _install_abstraction_rules(self, relation_names: List[str]) -> None:
        """Create abstraction rules: each specific relation implies related_to(X, Y)."""
        related = Symbol("related_to")
        x, y = Variable("X"), Variable("Y")
        for rel_name in relation_names:
            pred = Symbol(sanitize_entity_id(rel_name))
            self.kb.add_rule(Rule((related, x, y), ((pred, x, y),)))

        # Transitivity for 'is_a_type_of': if A is-type-of B, B is-type-of C => A is-type-of C
        isa = Symbol(sanitize_entity_id("is a type of"))
        z = Variable("Z")
        self.kb.add_rule(Rule((isa, x, z), ((isa, x, y), (isa, y, z))))

        # Transitivity for 'depends_on'
        dep = Symbol(sanitize_entity_id("depends on"))
        self.kb.add_rule(Rule((dep, x, z), ((dep, x, y), (dep, y, z))))

        # Transitivity for 'is_part_of'
        part = Symbol(sanitize_entity_id("is part of"))
        self.kb.add_rule(Rule((part, x, z), ((part, x, y), (part, y, z))))

    def process_document(
        self,
        text: str,
        doc_id: str = "",
        label: Optional[str] = None,
    ) -> int:
        """
        Process a single document: extract concepts, generate pairs, classify relations.

        Args:
            text: Document text
            doc_id: Optional document identifier
            label: Optional category/tag label

        Returns:
            Number of relations discovered in this document
        """
        self.total_docs += 1

        # Extract concepts
        concepts = extract_concepts(text, max_concepts=self.max_concepts)
        if not concepts:
            return 0

        self.total_concepts += len(concepts)
        for c in concepts:
            self.concept_frequency[c] += 1

        # If we have a label, add it as a concept too
        if label:
            label_concepts = [l.strip() for l in str(label).split(",") if l.strip()]
            for lc in label_concepts[:3]:
                lc_clean = lc.strip().lower()
                if lc_clean and lc_clean not in STOPWORDS and len(lc_clean) >= 2:
                    concepts.append(lc_clean)
                    self.concept_frequency[lc_clean] += 1

        # Generate entity pairs
        pairs = generate_entity_pairs(concepts, max_pairs=self.max_pairs)
        if not pairs:
            return 0

        self.total_pairs_classified += len(pairs)

        # Truncate context for zero-shot premise
        context = text[:300] if len(text) > 300 else text

        # Classify relations
        found = 0
        for e1, e2 in pairs:
            try:
                rel, conf = self.discoverer.classify_relation(e1, e2, context)
                if conf >= self.min_confidence:
                    self.staged_relations.append((e1, e2, rel, conf, context[:150]))
                    found += 1
            except Exception:
                continue

        self.total_relations_found += found
        return found

    def process_batch(
        self,
        texts: List[str],
        labels: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> None:
        """
        Process a batch of documents.

        Args:
            texts: List of text strings
            labels: Optional list of label strings
            show_progress: Show progress bar
        """
        bar = tqdm(
            enumerate(texts),
            total=len(texts),
            desc="Processing documents",
            unit="doc",
            disable=not show_progress,
            dynamic_ncols=True,
        )
        for i, text in bar:
            label = labels[i] if labels else None
            n_rels = self.process_document(text, doc_id=str(i), label=label)
            if show_progress:
                bar.set_postfix(
                    concepts=self.total_concepts,
                    relations=self.total_relations_found,
                    staged=len(self.staged_relations),
                )

    def graduate_to_kb(self, show_progress: bool = True) -> int:
        """
        Graduate staged relations into the symbolic Knowledge Base.

        Creates symbolic facts (predicate, subject, object) and upserts
        entity embeddings into the VectorGraphKB.

        Returns:
            Number of facts promoted
        """
        promoted = 0
        entities_to_embed: Set[str] = set()

        bar = tqdm(
            self.staged_relations,
            desc="Graduating to KB",
            unit="rel",
            disable=not show_progress,
            dynamic_ncols=True,
        )

        for e1, e2, rel, conf, ctx in bar:
            # Create symbolic expression
            pred_id = sanitize_entity_id(rel)
            subj_id = sanitize_entity_id(e1)
            obj_id = sanitize_entity_id(e2)

            pred = Symbol(pred_id)
            subj = Symbol(subj_id)
            obj = Symbol(obj_id)

            fact: Expression = (pred, subj, obj)

            # Check consistency and add
            if self.kb.is_consistent_candidate(fact):
                if self.kb.add_fact(fact):
                    self.vkb.add_edge(subj_id, pred_id, obj_id, provenance=f"zero-shot:{conf:.3f}")
                    entities_to_embed.add(e1)
                    entities_to_embed.add(e2)
                    promoted += 1

            if show_progress:
                bar.set_postfix(promoted=promoted)

        # Embed entities for vector retrieval
        if entities_to_embed:
            entity_list = list(entities_to_embed)
            print(f"  Embedding {len(entity_list)} unique entities...")
            # Batch embed
            batch_size = 32
            for i in range(0, len(entity_list), batch_size):
                batch = entity_list[i : i + batch_size]
                vecs = self.embedder.embed_batch(batch)
                for ent, vec in zip(batch, vecs):
                    eid = sanitize_entity_id(ent)
                    self.vkb.upsert_node(eid, vec, label=ent)

        self.total_promoted = promoted
        return promoted

    def run_inference(self, trace: Optional[List[str]] = None) -> int:
        """Run forward-chaining inference on the KB."""
        return self.inference.forward_chain(max_rounds=24, trace=trace)

    def query_entity(self, needle: str, max_results: int = 40) -> List[Expression]:
        """Find all facts mentioning an entity (case-insensitive substring)."""
        needle_l = needle.lower()
        hits = [f for f in self.kb.facts if needle_l in str(f).lower()]
        return hits[:max_results]

    def summary_stats(self) -> Dict[str, Any]:
        """Return summary statistics."""
        return {
            "documents_processed": self.total_docs,
            "concepts_extracted": self.total_concepts,
            "unique_concepts": len(self.concept_frequency),
            "entity_pairs_classified": self.total_pairs_classified,
            "relations_discovered": self.total_relations_found,
            "facts_promoted": self.total_promoted,
            "total_kb_facts": len(self.kb.facts),
            "total_kb_rules": len(self.kb.rules),
            "graph_nodes": len(self.vkb.nodes),
            "graph_edges": len(self.vkb.edges),
            "top_concepts": self.concept_frequency.most_common(20),
        }


# =============================================================================
# Export Utilities
# =============================================================================


def _expr_to_serializable(expr: Any) -> Any:
    """Convert symbolic expressions to JSON-serializable form."""
    if isinstance(expr, Symbol):
        return {"sym": expr.name}
    if isinstance(expr, Variable):
        return {"var": expr.name}
    if isinstance(expr, tuple):
        return [_expr_to_serializable(x) for x in expr]
    return str(expr)


def export_kb_json(
    path: str,
    kb: KnowledgeBase,
    vkb: VectorGraphKB,
    stats: Dict[str, Any],
    inference_trace: Optional[List[str]] = None,
) -> None:
    """Export KB to JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    facts = [_expr_to_serializable(f) for f in sorted(kb.facts, key=lambda x: str(x))]
    rules = [
        {"head": _expr_to_serializable(r.head), "body": [_expr_to_serializable(b) for b in r.body]}
        for r in kb.rules
    ]
    edges = [
        {"subj": s, "relation": rel, "obj": o, "provenance": prov}
        for s, rel, o, prov in vkb.edges
    ]
    nodes_meta = {
        nid: {"metadata": n.metadata, "embedding_dim": int(n.embedding.shape[0])}
        for nid, n in vkb.nodes.items()
    }

    bundle: Dict[str, Any] = {
        "facts": facts,
        "rules": rules,
        "graph_edges": edges,
        "graph_nodes": nodes_meta,
        "stats": stats,
    }
    if inference_trace:
        bundle["inference_trace"] = inference_trace

    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, default=str)
    print(f"  Wrote JSON export: {path}")


def stable_graph_id(label: str) -> str:
    """Deterministic short node ID for Graphviz."""
    return "n_" + hashlib.sha256(label.encode("utf-8")).hexdigest()[:16]


def export_kb_dot(path: str, kb: KnowledgeBase, vkb: VectorGraphKB) -> None:
    """Export KB as Graphviz DOT file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    lines = [
        "digraph knowledge_graph {",
        "  rankdir=LR;",
        '  node [shape=box, fontname=Helvetica, fontsize=10, style=filled, fillcolor="#E8F4FD"];',
        '  edge [fontname=Helvetica, fontsize=8, color="#555555"];',
    ]
    declared: Set[str] = set()
    for fact in kb.facts:
        if len(fact) != 3:
            continue
        pred, subj, obj = fact
        if not (_is_sym(pred) and _is_sym(subj) and _is_sym(obj)):
            continue
        sid, oid = stable_graph_id(subj.name), stable_graph_id(obj.name)
        for gid, glabel in ((sid, subj.name), (oid, obj.name)):
            if gid not in declared:
                declared.add(gid)
                lab = glabel.replace("\\", "\\\\").replace('"', '\\"')[:80]
                lines.append(f'  {gid} [label="{lab}"];')
        pl = pred.name.replace("\\", "\\\\").replace('"', '\\"')[:80]
        lines.append(f'  {sid} -> {oid} [label="{pl}"];')
    lines.append("}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote Graphviz DOT: {path}  (render: dot -Tsvg {path} -o graph.svg)")


# =============================================================================
# Dataset Loading
# =============================================================================


def load_hf_dataset(
    dataset_name: str,
    text_field: str,
    label_field: Optional[str] = None,
    split: str = "train",
    streaming: bool = False,
    max_samples: int = 0,
) -> Tuple[List[str], Optional[List[str]], List[str]]:
    """
    Load a HuggingFace dataset generically.

    Args:
        dataset_name: HuggingFace dataset identifier
        text_field: Name of the text column
        label_field: Optional name of the label/category column
        split: Dataset split to load
        streaming: Whether to use streaming mode
        max_samples: Maximum samples to load (0 = all, requires non-streaming for 'all')

    Returns:
        (texts, labels_or_None, available_fields)
    """
    print(f"\nLoading dataset: {dataset_name}")
    print(f"  Split: {split} | Streaming: {streaming}")

    ds = load_dataset(dataset_name, split=split, streaming=streaming)

    # Detect available fields
    if streaming:
        # For streaming, peek at first item
        peek = next(iter(ds))
        available_fields = list(peek.keys())
        print(f"  Available fields: {available_fields}")

        if text_field not in available_fields:
            raise ValueError(
                f"Text field '{text_field}' not found. Available: {available_fields}"
            )
        if label_field and label_field not in available_fields:
            print(f"  Warning: label field '{label_field}' not found, ignoring.")
            label_field = None

        # Collect samples from stream
        texts: List[str] = []
        labels: Optional[List[str]] = [] if label_field else None
        limit = max_samples if max_samples > 0 else 10000  # safety limit for streaming

        print(f"  Streaming up to {limit} samples...")
        bar = tqdm(ds, total=limit, desc="  Loading", unit="row", dynamic_ncols=True)
        for row in bar:
            txt = str(row.get(text_field, "")).strip()
            if not txt or len(txt) < 10:
                continue
            texts.append(txt)
            if labels is not None and label_field:
                labels.append(str(row.get(label_field, "")))
            if len(texts) >= limit:
                break
    else:
        available_fields = list(ds.features.keys()) if hasattr(ds, "features") else list(ds[0].keys())
        print(f"  Available fields: {available_fields}")

        if text_field not in available_fields:
            raise ValueError(
                f"Text field '{text_field}' not found. Available: {available_fields}"
            )
        if label_field and label_field not in available_fields:
            print(f"  Warning: label field '{label_field}' not found, ignoring.")
            label_field = None

        limit = max_samples if max_samples > 0 else len(ds)
        texts = []
        labels = [] if label_field else None
        for i in range(min(limit, len(ds))):
            txt = str(ds[i][text_field]).strip()
            if not txt or len(txt) < 10:
                continue
            texts.append(txt)
            if labels is not None and label_field:
                labels.append(str(ds[i][label_field]))

    print(f"  Loaded {len(texts)} documents")
    return texts, labels, available_fields


# =============================================================================
# Pretty Printing
# =============================================================================


def print_summary(
    stats: Dict[str, Any],
    inference_trace: Optional[List[str]] = None,
    elapsed: float = 0.0,
) -> None:
    """Print a nicely formatted summary of what was discovered."""
    width = 60
    print("\n" + "=" * width)
    print("  NEURO-SYMBOLIC KNOWLEDGE GRAPH — DISCOVERY SUMMARY")
    print("=" * width)

    print(f"\n  📄 Documents processed:      {stats['documents_processed']:>6}")
    print(f"  🔍 Concepts extracted:       {stats['concepts_extracted']:>6}")
    print(f"  🧩 Unique concepts:          {stats['unique_concepts']:>6}")
    print(f"  🔗 Entity pairs classified:  {stats['entity_pairs_classified']:>6}")
    print(f"  ✅ Relations discovered:      {stats['relations_discovered']:>6}")
    print(f"  📥 Facts promoted to KB:     {stats['facts_promoted']:>6}")
    print(f"  📊 Total KB facts:           {stats['total_kb_facts']:>6}")
    print(f"  📐 KB rules:                 {stats['total_kb_rules']:>6}")
    print(f"  🌐 Graph nodes:              {stats['graph_nodes']:>6}")
    print(f"  ➡️  Graph edges:              {stats['graph_edges']:>6}")

    if elapsed > 0:
        print(f"\n  ⏱️  Total time:              {elapsed:>6.1f}s")

    top = stats.get("top_concepts", [])
    if top:
        print(f"\n  Top concepts (by frequency):")
        for concept, freq in top[:15]:
            bar_len = min(20, freq)
            bar_str = "█" * bar_len
            print(f"    {concept:<30s} {freq:>3} {bar_str}")

    if inference_trace:
        n_derived = len(inference_trace)
        cap = 20
        print(f"\n  Inference trace ({n_derived} facts derived by forward chaining):")
        for line in inference_trace[:cap]:
            print(f"    {line}")
        if n_derived > cap:
            print(f"    ... ({n_derived - cap} more, use --export-json for full trace)")

    print("\n" + "=" * width)


# =============================================================================
# Self-test
# =============================================================================


def _self_test() -> None:
    """Quick sanity check of symbolic engine + concept extraction."""
    # Symbolic
    x, y = Variable("X"), Variable("Y")
    a, b = Symbol("a"), Symbol("b")
    s1 = unify((Symbol("r"), x, b), (Symbol("r"), a, b), {})
    assert s1 is not None and s1[x] == a
    kb = KnowledgeBase()
    kb.add_fact((Symbol("p"), a, b))
    kb.add_rule(Rule((Symbol("q"), x, y), ((Symbol("p"), x, y),)))
    eng = InferenceEngine(kb)
    n = eng.forward_chain(max_rounds=4)
    assert n >= 1
    assert (Symbol("q"), a, b) in kb.facts

    # Concept extraction
    concepts = extract_concepts(
        "Machine learning algorithms require large datasets for training. "
        "Neural networks are a type of machine learning model."
    )
    assert len(concepts) > 0, "concept extraction returned empty"

    # Entity pairs
    pairs = generate_entity_pairs(concepts)
    assert len(pairs) >= 0  # may be 0 if only 1 concept

    print("Self-test passed ✓")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generalized Neuro-Symbolic Knowledge Graph Builder — "
        "discovers relationships in ANY text dataset using zero-shot classification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Coding dataset (streaming, large)
  python prototypes/prototype_general_alpha.py \\
      --dataset grandsmile/Generative_Coding_Dataset \\
      --text-field question --label-field tags --streaming --max-samples 100

  # Wikipedia sentences
  python prototypes/prototype_general_alpha.py \\
      --dataset sentence-transformers/wikipedia-en-sentences \\
      --text-field sentence --max-samples 200

  # Custom relations for a medical dataset
  python prototypes/prototype_general_alpha.py \\
      --dataset medical_questions_pairs \\
      --text-field question_1 --max-samples 100 \\
      --relations "treats,causes,prevents,diagnoses,is symptom of"
""",
    )

    # Dataset args
    p.add_argument("--dataset", required=True, help="HuggingFace dataset name/path")
    p.add_argument("--text-field", required=True, help="Column name containing text")
    p.add_argument("--label-field", default="", help="Optional column with categories/tags")
    p.add_argument("--split", default="train", help="Dataset split (default: train)")
    p.add_argument("--streaming", action="store_true", help="Stream dataset (for large datasets)")
    p.add_argument("--max-samples", type=int, default=100, help="Max documents to process (default: 100)")

    # Model args
    p.add_argument(
        "--model-name",
        default="facebook/bart-large-mnli",
        help="Zero-shot classifier model (default: facebook/bart-large-mnli)",
    )
    p.add_argument(
        "--embed-model",
        default="distilbert-base-uncased",
        help="Embedding model for vector store (default: distilbert-base-uncased)",
    )
    p.add_argument("--device", default="cpu", help="Device: cpu, cuda, cuda:0, etc.")

    # Relation discovery args
    p.add_argument(
        "--relations",
        default="",
        help="Comma-separated custom relation types to ADD to defaults",
    )
    p.add_argument("--min-confidence", type=float, default=0.3, help="Min confidence for relations (default: 0.3)")
    p.add_argument("--max-concepts", type=int, default=8, help="Max concepts per document (default: 8)")
    p.add_argument("--max-pairs", type=int, default=10, help="Max entity pairs per document (default: 10)")

    # Export args
    p.add_argument("--export-json", default="", metavar="PATH", help="Export KB as JSON")
    p.add_argument("--export-dot", default="", metavar="PATH", help="Export KB as Graphviz DOT")
    p.add_argument("--trace-inference", action="store_true", help="Show forward-chaining trace")
    p.add_argument("--query-entity", default="", metavar="SUBSTRING", help="Query facts by entity substring")
    p.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    args = p.parse_args()

    # Validate device
    device_str = args.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device_str = "cpu"

    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    show_progress = not args.no_progress

    # Self-test
    _self_test()

    start_time = time.time()

    # --- 1. Load Dataset ---
    label_field = args.label_field if args.label_field else None
    texts, labels, fields = load_hf_dataset(
        dataset_name=args.dataset,
        text_field=args.text_field,
        label_field=label_field,
        split=args.split,
        streaming=args.streaming,
        max_samples=args.max_samples,
    )

    if not texts:
        raise SystemExit("No texts loaded. Check --dataset, --text-field, and --split.")

    # --- 2. Set up candidate relations ---
    candidate_relations = list(DEFAULT_RELATIONS)
    if args.relations:
        extra = [r.strip() for r in args.relations.split(",") if r.strip()]
        candidate_relations.extend(extra)
        print(f"Added custom relations: {extra}")

    # --- 3. Initialize components ---
    discoverer = ZeroShotRelationDiscoverer(
        model_name=args.model_name,
        candidate_relations=candidate_relations,
        device=device_str,
        min_confidence=args.min_confidence,
    )

    embedder = EmbeddingProvider(
        model_name=args.embed_model,
        device=device_str,
    )

    builder = KnowledgeGraphBuilder(
        discoverer=discoverer,
        embedder=embedder,
        min_confidence=args.min_confidence,
        max_concepts_per_doc=args.max_concepts,
        max_pairs_per_doc=args.max_pairs,
    )

    # --- 4. Process documents ---
    print(f"\nProcessing {len(texts)} documents...")
    builder.process_batch(texts, labels=labels, show_progress=show_progress)

    # --- 5. Graduate to KB ---
    print(f"\nGraduating {len(builder.staged_relations)} staged relations to KB...")
    promoted = builder.graduate_to_kb(show_progress=show_progress)
    print(f"  Promoted {promoted} facts")

    # --- 6. Run inference ---
    inf_trace: Optional[List[str]] = [] if args.trace_inference else None
    derived = builder.run_inference(trace=inf_trace)
    print(f"  Forward chaining derived {derived} new facts")

    # --- 7. Summary ---
    stats = builder.summary_stats()
    elapsed = time.time() - start_time
    print_summary(stats, inference_trace=inf_trace, elapsed=elapsed)

    # --- 8. Sample facts ---
    sample_facts = list(builder.kb.facts)[:10]
    if sample_facts:
        print("\nSample KB facts (up to 10):")
        for f in sample_facts:
            print(f"  {f}")

    # --- 9. Vector retrieval demo ---
    if builder.vkb.nodes:
        any_id = next(iter(builder.vkb.nodes))
        q = builder.vkb.nodes[any_id].embedding
        neigh = builder.vkb.neural_retrieve(q, top_k=5)
        print(f"\nVector retrieval (cosine neighbors of '{any_id}'):")
        for nid, sc in neigh:
            meta = builder.vkb.nodes[nid].metadata
            label_str = meta.get("label", nid)
            print(f"  {label_str:<30s} cosine={sc:.4f}")

    # --- 10. Exports ---
    if args.export_json:
        export_kb_json(args.export_json, builder.kb, builder.vkb, stats, inf_trace)
    if args.export_dot:
        export_kb_dot(args.export_dot, builder.kb, builder.vkb)

    # --- 11. Query ---
    if args.query_entity:
        hits = builder.query_entity(args.query_entity)
        print(f"\nFacts matching '{args.query_entity}' ({len(hits)} found):")
        for f in hits[:40]:
            print(f"  {f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
