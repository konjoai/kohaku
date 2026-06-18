"""Memory conflict detection — heuristic contradiction signals between
memories that share a topic.

The detector is intentionally *heuristic*, not a verdict. Three independent
signals contribute to a ``contradiction_score`` in [0, 1]:

1. **High HDC cosine similarity** — both memories are about the same thing
   (above ``similarity_threshold``, default 0.40).
2. **Subject anchor overlap with predicate divergence** — when the first
   ≥2 tokens overlap (same subject, e.g. "the user prefers …") but the
   trailing content tokens diverge with no overlap, that's a classic
   "same subject, different claim" signal.
3. **Polarity flip** — exactly one of the two memories contains a
   negation marker (``not``, ``no``, ``never``, ``n't``).
4. **Numeric divergence** — both contain numbers and the numbers differ.

Each signal contributes a weighted partial score; a pair is *flagged*
only when the total clears ``contradiction_threshold`` (default 0.50).

The score is meant to power a triage UI, not to auto-delete anything.
:func:`detect_conflicts` returns the ranked list; the caller decides what
to do — confirm, dismiss, or merge via the existing consolidation hook.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from kohaku._index import index_over
from kohaku.enriched import EnrichedMemoryStore

logger = logging.getLogger(__name__)


# Tokens that count as polarity-flipping when one memory has them and the
# other does not. Lowered, whitespace-tokenised; ``n't`` is also matched
# against the raw label via a regex so "doesn't" / "isn't" / "won't" all
# count.
NEGATION_TOKENS: frozenset[str] = frozenset({
    "not", "no", "never", "none", "nothing", "neither", "nor", "without",
})
NEGATION_RE = re.compile(r"\bn[’']t\b|\bn't\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
# Filler words we strip from anchor / predicate token sets — they carry no
# topical content and would falsely inflate overlap.
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "to",
    "in", "on", "of", "for", "and", "or", "but", "by", "with", "at",
    "this", "that", "these", "those", "it", "its",
})

# Default tuning. Each ``_w_*`` is the partial weight contributed by that
# signal when present; they sum to 1.0 so the final score is in [0, 1].
DEFAULT_SIMILARITY_THRESHOLD: float = 0.40
DEFAULT_CONTRADICTION_THRESHOLD: float = 0.45
_W_SIMILARITY: float = 0.40   # gated on similarity ≥ threshold
_W_NEGATION: float = 0.30
_W_NUMERIC: float = 0.15
_W_PREDICATE_DIVERGE: float = 0.15


@dataclass(frozen=True)
class ConflictPair:
    """One detected contradiction signal between two memories."""
    a_id: int
    b_id: int
    label_a: str
    label_b: str
    similarity: float
    contradiction_score: float
    reasons: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "a_id": int(self.a_id),
            "b_id": int(self.b_id),
            "label_a": self.label_a,
            "label_b": self.label_b,
            "similarity": round(float(self.similarity), 4),
            "contradiction_score": round(float(self.contradiction_score), 4),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class ConflictResolution:
    """Outcome of resolving a conflict — what stayed, what got removed."""
    kept_id: Optional[int]
    removed_ids: Tuple[int, ...]
    action: str   # "keep_a" | "keep_b" | "keep_both" | "dismiss"

    def to_dict(self) -> dict:
        return {
            "kept_id": int(self.kept_id) if self.kept_id is not None else None,
            "removed_ids": [int(i) for i in self.removed_ids],
            "action": self.action,
        }


# ──────────────────────────── tokenisation ─────────────────────────────────

def _tokenise(text: str) -> List[str]:
    """Lower-cased word tokenisation with stopwords stripped."""
    out: List[str] = []
    for raw in re.findall(r"[a-zA-Z']+", text.lower()):
        if raw and raw not in _STOPWORDS:
            out.append(raw)
    return out


def _subject_anchor(tokens: Sequence[str], n: int = 2) -> Tuple[str, ...]:
    """First ``n`` content tokens. Anchors approximate the subject phrase."""
    return tuple(tokens[:n])


def _predicate_set(tokens: Sequence[str], anchor_n: int = 2) -> Set[str]:
    """Content tokens *after* the anchor — the predicate slot."""
    return set(tokens[anchor_n:])


def _has_negation(text: str, tokens: Sequence[str]) -> bool:
    if any(t in NEGATION_TOKENS for t in tokens):
        return True
    return bool(NEGATION_RE.search(text))


def _numbers_in(text: str) -> Set[str]:
    return set(_NUMBER_RE.findall(text))


# ──────────────────────────── scoring ──────────────────────────────────────

def _score_pair(
    label_a: str,
    label_b: str,
    similarity: float,
    *,
    similarity_threshold: float,
) -> Tuple[float, Tuple[str, ...]]:
    """Compute the contradiction score and the human-readable reasons."""
    if similarity < similarity_threshold:
        return 0.0, ()
    reasons: List[str] = []
    score = _W_SIMILARITY  # we earned this by clearing the gate
    reasons.append(f"shared topic (cosine {similarity:.2f})")

    ta = _tokenise(label_a)
    tb = _tokenise(label_b)
    anchor_a = _subject_anchor(ta)
    anchor_b = _subject_anchor(tb)
    pred_a = _predicate_set(ta)
    pred_b = _predicate_set(tb)

    if anchor_a and anchor_b and anchor_a == anchor_b and pred_a and pred_b:
        # Same subject framing — predicate divergence becomes meaningful.
        # Use Jaccard distance so partial overlap (e.g. shared "morning")
        # still earns some divergence credit when the rest diverges.
        inter = len(pred_a & pred_b)
        union = len(pred_a | pred_b)
        jaccard = inter / union if union else 1.0
        if jaccard < 0.5:
            # Scale the partial weight by how disjoint the predicates are.
            partial = _W_PREDICATE_DIVERGE * (1.0 - 2.0 * jaccard)
            score += partial
            reasons.append(
                f"same subject, predicate Jaccard {jaccard:.2f}"
            )

    neg_a = _has_negation(label_a, ta)
    neg_b = _has_negation(label_b, tb)
    if neg_a != neg_b:
        score += _W_NEGATION
        reasons.append("one side negated")

    nums_a = _numbers_in(label_a)
    nums_b = _numbers_in(label_b)
    if nums_a and nums_b and nums_a != nums_b:
        score += _W_NUMERIC
        reasons.append(f"numeric divergence {sorted(nums_a)} vs {sorted(nums_b)}")

    return min(1.0, score), tuple(reasons)


# ──────────────────────────── public API ───────────────────────────────────

def detect_conflicts(
    store: EnrichedMemoryStore,
    *,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    contradiction_threshold: float = DEFAULT_CONTRADICTION_THRESHOLD,
    max_pairs: int = 100,
) -> List[ConflictPair]:
    """Scan an :class:`EnrichedMemoryStore` for contradiction signals.

    O(n²) — fine at the kohaku scale (≤ a few thousand entries). Results
    are sorted by ``contradiction_score`` descending; ties broken by
    raw similarity.
    """
    if not 0.0 < similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in (0, 1]")
    if not 0.0 < contradiction_threshold <= 1.0:
        raise ValueError("contradiction_threshold must be in (0, 1]")
    if max_pairs < 0:
        raise ValueError("max_pairs must be >= 0")

    entries = store.episodic.entries()
    out: List[ConflictPair] = []
    # Batched cosine: score each pivot row against all others in one pass; the
    # heuristic text scoring still runs per candidate pair.
    idx = index_over(entries)
    for i in range(len(entries)):
        ei = entries[i]
        sims = idx.all_scores(ei.key.data)
        for j in range(i + 1, len(entries)):
            ej = entries[j]
            sim = float(sims[j])
            score, reasons = _score_pair(
                ei.label, ej.label, sim,
                similarity_threshold=similarity_threshold,
            )
            if score >= contradiction_threshold:
                out.append(ConflictPair(
                    a_id=ei.id, b_id=ej.id,
                    label_a=ei.label, label_b=ej.label,
                    similarity=sim,
                    contradiction_score=score,
                    reasons=reasons,
                ))
    out.sort(key=lambda p: (p.contradiction_score, p.similarity), reverse=True)
    return out[:max_pairs]


def resolve_conflict(
    store: EnrichedMemoryStore,
    *,
    a_id: int,
    b_id: int,
    keep: str,
) -> ConflictResolution:
    """Apply a resolution decision to two conflicting memories.

    ``keep`` ∈ {"a", "b", "both", "dismiss"}:

    * ``"a"`` — drop b, keep a.
    * ``"b"`` — drop a, keep b.
    * ``"both"`` — leave both in place (caller decided they're not contradictions).
    * ``"dismiss"`` — alias for ``"both"`` — close the conflict without action.
    """
    if keep not in ("a", "b", "both", "dismiss"):
        raise ValueError("keep must be one of 'a', 'b', 'both', 'dismiss'")
    if a_id == b_id:
        raise ValueError("a_id and b_id must differ")

    live_ids = {e.id for e in store.episodic.entries()}
    if a_id not in live_ids:
        raise ValueError(f"unknown memory_id {a_id}")
    if b_id not in live_ids:
        raise ValueError(f"unknown memory_id {b_id}")

    if keep == "a":
        _remove_entries(store, [b_id])
        return ConflictResolution(kept_id=a_id, removed_ids=(b_id,), action="keep_a")
    if keep == "b":
        _remove_entries(store, [a_id])
        return ConflictResolution(kept_id=b_id, removed_ids=(a_id,), action="keep_b")
    # both / dismiss
    return ConflictResolution(
        kept_id=None, removed_ids=(),
        action="keep_both" if keep == "both" else "dismiss",
    )


def _remove_entries(store: EnrichedMemoryStore, ids: Iterable[int]) -> int:
    """Drop entries from the underlying EpisodicMemory + metadata table.

    Mirrors :meth:`MemoryHealthAnalyzer.delete_stale` — pokes at the
    ``_entries`` list directly because :class:`EpisodicMemory` has no public
    removal API by design (it would break the FIFO/timestamp contract).
    """
    target = set(int(i) for i in ids)
    if not target:
        return 0
    kept = [e for e in store.episodic._entries if e.id not in target]
    removed = len(store.episodic._entries) - len(kept)
    store.episodic._entries = kept
    if removed:
        store.episodic._mark_mutated()  # invalidate the retrieval-index cache
    for eid in target:
        store._meta.pop(eid, None)
        if store.provenance is not None:
            store.provenance.delete(eid)
    return removed
