#!/usr/bin/env python3
"""From prose to relational reasoning — learning facts an agent *read*.

Run:  PYTHONPATH=python python3 examples/extraction_demo.py

Analogical memory reasons over structured records, but agents accumulate prose.
``Memory.learn`` closes the gap: it stores the sentence verbatim (episodic recall)
AND extracts ``(subject, attribute, value)`` triples into the analogical store —
so the "dollar of Mexico" trick runs over what the agent was *told*, not records
you hand-built. The extractor is high-precision: prose it can't parse adds nothing
(no fabricated facts), which keeps reasoning honest.
"""

from __future__ import annotations

from kohaku import Memory, extract_triples


def main() -> None:
    mem = Memory(dims=10_000)

    print("Feeding the agent plain sentences (learn = store + extract):")
    notes = [
        "The capital of the USA is Washington",
        "USA's currency is the dollar",
        "The capital of Mexico is Mexico City",
        "Mexico's currency is the peso",
        "I had a great lunch today",  # not a fact — yields no triple
    ]
    for text in notes:
        _eid, triples = mem.learn(text)
        learned = ", ".join(f"{t.subject}.{t.attribute}={t.value}" for t in triples)
        print(
            f"  {text!r}\n      -> {learned or '(no structured fact — stored as prose only)'}"
        )

    print("\nEpisodic recall still works on the raw prose:")
    for hit in mem.query("what is the currency", top_k=2):
        print(f"  {hit.score:.3f}  {hit.text}")

    print("\nRelational reasoning now runs on the EXTRACTED facts:")
    cur = mem.attribute("usa", "currency")
    print(f"  attribute('usa','currency') -> {cur.value} (confidence {cur.confidence})")
    ana = mem.analogy("usa", "mexico", "dollar")
    print(
        f"  'dollar of mexico' analogy   -> {ana.value} (confidence {ana.confidence})"
    )

    print("\nStandalone extraction (no memory needed):")
    for t in extract_triples("Fido is a dog and the user prefers aisle seats"):
        print(f"  {t.subject}.{t.attribute} = {t.value}  (conf {t.confidence})")


if __name__ == "__main__":
    main()
