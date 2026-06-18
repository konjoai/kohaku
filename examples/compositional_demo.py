#!/usr/bin/env python3
"""Compositional recall — retrieve by several cues at once.

Run:  PYTHONPATH=python python3 examples/compositional_demo.py

Real recall is multi-constraint: you remember a few fragments and want the
memory that fits them all. kohaku bundles the cue vectors into one composite
query (a soft conjunction) and ranks by closeness to the *combination* — in a
single pass, no model call. Hopfield cleanup is available for noisy cues
(opt-in; at 10k-D plain cosine is already very noise-robust).
"""
from __future__ import annotations

from kohaku import Memory


def main() -> None:
    mem = Memory()
    mem.store("User loves Italian red wine from Tuscany")
    mem.store("User enjoys French cheese and fresh baguettes")
    mem.store("User prefers Japanese green tea in the morning")
    mem.store("User dislikes Italian coffee but likes espresso")
    mem.store("User went hiking in the Italian Alps last summer")

    print("Single cue 'Italian' — ambiguous, several memories match:")
    for h in mem.query("Italian", top_k=3, reinforce=False):
        print(f"  {h.similarity:.3f}  {h.text}")

    print("\nMulti-cue ['Italian', 'wine', 'Tuscany'] — the combination wins:")
    for h in mem.recall_composite(["Italian", "wine", "Tuscany"], top_k=3, reinforce=False):
        print(f"  {h.similarity:.3f}  {h.text}")

    print("\nMulti-cue ['Italian', 'mountains', 'summer'] — different combination:")
    for h in mem.recall_composite(["Italian", "mountains", "summer"], top_k=3, reinforce=False):
        print(f"  {h.similarity:.3f}  {h.text}")

    print("\nNoisy/partial cue with Hopfield cleanup (pattern completion):")
    for h in mem.recall_composite(["Japanese", "tea"], top_k=2, cleanup=True, reinforce=False):
        print(f"  {h.similarity:.3f}  {h.text}")


if __name__ == "__main__":
    main()
