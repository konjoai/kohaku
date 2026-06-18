#!/usr/bin/env python3
"""Heuristic triple-extraction precision/recall benchmark (Track D).

The extractor (:mod:`kohaku.extraction`) trades recall for precision on purpose:
it must not invent facts. This measures both, honestly, against a hand-labelled
corpus of memory-style sentences — including *negatives* (prose that should yield
nothing). A wrong triple on a negative is a false positive; a missed triple on a
positive is a false negative.

Precision is the number that matters for a memory store (a fabricated fact
poisons reasoning), so we report it prominently alongside recall and F1. No
latency claims here; this is a correctness curve, written under
benchmarks/results/ and never overwritten.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import kohaku  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from kohaku.extraction import extract_triples  # noqa: E402

# (sentence, expected (subject, attribute, value) or None for a true negative).
Gold = Tuple[str, Optional[Tuple[str, str, str]]]

CORPUS: List[Gold] = [
    # of-form
    ("The capital of France is Paris", ("france", "capital", "paris")),
    ("The capital of Japan is Tokyo", ("japan", "capital", "tokyo")),
    ("The currency of Germany is the euro", ("germany", "currency", "euro")),
    ("The population of India is huge", ("india", "population", "huge")),
    # reversed of-form
    ("Tokyo is the capital of Japan", ("japan", "capital", "tokyo")),
    ("Lisbon is the capital of Portugal", ("portugal", "capital", "lisbon")),
    # possessive
    ("France's currency is the euro", ("france", "currency", "euro")),
    ("USA's capital is Washington", ("usa", "capital", "washington")),
    ("Mexico's currency: peso", ("mexico", "currency", "peso")),
    ("Alice's role is engineer", ("alice", "role", "engineer")),
    # is-a / type
    ("Fido is a dog", ("fido", "type", "dog")),
    ("Python is a programming language", ("python", "type", "programming language")),
    # preference verbs
    ("The user prefers aisle seats", ("user", "prefers", "aisle seats")),
    ("Bob likes jazz", ("bob", "likes", "jazz")),
    ("The customer uses Linux", ("customer", "uses", "linux")),
    # ── out-of-scope facts (real facts in unsupported phrasings) ──
    # Included deliberately so recall reflects the true envelope: these ARE
    # facts, but the heuristics don't cover their phrasing, so they are honest
    # false negatives — not false positives. Recall < 1.0 is the point.
    ("Paris, the capital of France, is lovely", ("france", "capital", "paris")),
    ("She lives in Berlin", ("she", "lives_in", "berlin")),
    ("John works as a teacher", ("john", "occupation", "teacher")),
    ("Water boils at 100 degrees", ("water", "boiling_point", "100 degrees")),
    # ── true negatives (must yield nothing) ──
    ("I went hiking yesterday", None),
    ("What time is the meeting?", None),
    ("Please remember to buy milk", None),
    ("It was a great day", None),
    ("Let's discuss this tomorrow", None),
    ("The weather looks nice today", None),
    ("Thanks for your help", None),
]


def _predict(text: str) -> Optional[Tuple[str, str, str]]:
    triples = extract_triples(text)
    if not triples:
        return None
    t = triples[0]
    return (t.subject, t.attribute, t.value)


def run() -> dict:
    tp = fp = fn = tn = 0
    misses: List[str] = []
    for sentence, gold in CORPUS:
        pred = _predict(sentence)
        if gold is None:
            if pred is None:
                tn += 1
            else:
                fp += 1
                misses.append(f"FALSE POSITIVE: {sentence!r} -> {pred}")
        else:
            if pred == gold:
                tp += 1
            else:
                fn += 1
                misses.append(f"MISS: {sentence!r} -> {pred} (want {gold})")

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n_sentences": len(CORPUS),
        "n_positives": tp + fn,
        "n_negatives": tn + fp,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "misses": misses,
    }


def main() -> None:
    result = run()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(__file__).resolve().parent / "results" / f"{stamp}_extraction"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": stamp,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "result": result,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    print("heuristic triple extraction — precision/recall")
    print("-" * 46)
    print(
        f"sentences   : {result['n_sentences']}  "
        f"(+{result['n_positives']} / -{result['n_negatives']})"
    )
    print(f"precision   : {result['precision']}  (fp={result['false_positives']})")
    print(f"recall      : {result['recall']}  (fn={result['false_negatives']})")
    print(f"f1          : {result['f1']}")
    for line in result["misses"]:
        print(f"  {line}")
    print(f"\nwrote {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
