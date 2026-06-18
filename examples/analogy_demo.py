#!/usr/bin/env python3
"""Analogical memory — the thing a vector database cannot do.

Run:  PYTHONPATH=python python3 examples/analogy_demo.py

kohaku stores each record as a superposition of bound (attribute, value) pairs,
so it can do *algebra* over memory — recover an attribute, or transfer a value
across records by analogy — with no model call. This is "What is the dollar of
Mexico?" (Kanerva 2010), running over an agent's own memory.
"""
from __future__ import annotations

from kohaku import AnalogicalMemory


def main() -> None:
    mem = AnalogicalMemory()
    mem.add_record("USA", {"currency": "dollar", "capital": "washington", "language": "english"})
    mem.add_record("Mexico", {"currency": "peso", "capital": "mexico_city", "language": "spanish"})
    mem.add_record("France", {"currency": "euro", "capital": "paris", "language": "french"})
    mem.add_record("Japan", {"currency": "yen", "capital": "tokyo", "language": "japanese"})

    print("Attribute queries (unbind + cleanup):")
    for name, attr in [("USA", "currency"), ("France", "capital"), ("Japan", "language")]:
        r = mem.get(name, attr)
        print(f"  {attr} of {name:<7} -> {r.value:<12} (confidence {r.confidence})")

    print("\nAnalogical transfer (the dollar-of-Mexico trick):")
    for src, tgt, val in [
        ("USA", "Mexico", "dollar"),
        ("USA", "France", "washington"),
        ("Japan", "USA", "yen"),
        ("France", "Japan", "french"),
    ]:
        r = mem.analogy(src, tgt, val)
        print(f"  {val} is to {src} as ? is to {tgt:<7} -> {r.value:<12} "
              f"(confidence {r.confidence}, margin {r.margin})")

    print("\nAgent use case — transfer a learned preference across domains:")
    mem.add_record("flight_pref", {"seat": "aisle", "meal": "vegetarian", "time": "morning"})
    mem.add_record("train_pref", {"seat": "window", "meal": "snack", "time": "evening"})
    r = mem.analogy("flight_pref", "train_pref", "aisle")
    print(f"  user picks 'aisle' on flights; the train analog is -> {r.value} "
          f"(confidence {r.confidence})")


if __name__ == "__main__":
    main()
