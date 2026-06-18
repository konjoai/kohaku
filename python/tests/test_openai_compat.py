"""Tests for kohaku.openai_compat — the OpenAI-compatible memory middleware.

Covers the two responsibilities: context augmentation (retrieve + inject) and
learning from exchanges — both the existing episodic storage of assistant replies
and the new structured-fact mining into an optional ``AnalogicalMemory`` (the
extraction bridge wired into the conversation flow).
"""

from __future__ import annotations

import pytest

from kohaku import AnalogicalMemory
from kohaku.context import ContextConfig, ContextMemoryManager
from kohaku.openai_compat import MemoryMiddleware


def _manager() -> ContextMemoryManager:
    return ContextMemoryManager(ContextConfig(max_tokens=1000, tokens_per_entry=50))


# ── construction ──────────────────────────────────────────────────────────────


def test_invalid_learn_facts_from_rejected():
    with pytest.raises(ValueError, match="learn_facts_from"):
        MemoryMiddleware(_manager(), learn_facts_from="nonsense")


# ── fact mining (the new behaviour) ───────────────────────────────────────────


def _exchange() -> list[dict]:
    return [
        {"role": "user", "content": "USA's currency is the dollar"},
        {"role": "assistant", "content": "Noted — the US dollar."},
        {"role": "user", "content": "The capital of Mexico is Mexico City"},
        {"role": "assistant", "content": "Got it."},
    ]


def test_no_analogical_store_means_no_facts_learned():
    mw = MemoryMiddleware(_manager())
    learned = mw.learn_from_exchange(_exchange())
    assert learned == []


def test_facts_mined_from_user_messages_into_analogical():
    ana = AnalogicalMemory()
    mw = MemoryMiddleware(_manager(), analogical=ana)
    learned = mw.learn_from_exchange(_exchange())

    facts = {(t.subject, t.attribute, t.value) for t in learned}
    assert ("usa", "currency", "dollar") in facts
    assert ("mexico", "capital", "mexico city") in facts
    # and they are queryable in the analogical store
    assert ana.get("usa", "currency").value == "dollar"


def test_default_does_not_learn_from_assistant():
    ana = AnalogicalMemory()
    mw = MemoryMiddleware(_manager(), analogical=ana)
    mw.learn_from_exchange(
        [
            {"role": "user", "content": "What is France's currency?"},
            {"role": "assistant", "content": "France's currency is the euro"},
        ]
    )
    # the fact lives only in the assistant turn → not learned by default
    assert "france" not in ana


def test_learn_facts_from_both_includes_assistant():
    ana = AnalogicalMemory()
    mw = MemoryMiddleware(_manager(), analogical=ana, learn_facts_from="both")
    mw.learn_from_exchange(
        [
            {"role": "user", "content": "Tell me about France"},
            {"role": "assistant", "content": "France's currency is the euro"},
        ]
    )
    assert ana.get("france", "currency").value == "euro"


def test_unparseable_chatter_mines_nothing():
    ana = AnalogicalMemory()
    mw = MemoryMiddleware(_manager(), analogical=ana)
    learned = mw.learn_from_exchange(
        [
            {"role": "user", "content": "hey, how's it going?"},
            {"role": "assistant", "content": "All good!"},
        ]
    )
    assert learned == []
    assert len(ana) == 0


# ── episodic storage still works alongside fact mining ────────────────────────


def test_assistant_replies_still_stored_episodically():
    mgr = _manager()
    ana = AnalogicalMemory()
    mw = MemoryMiddleware(mgr, analogical=ana)
    mw.learn_from_exchange(_exchange())
    # the assistant responses became retrievable context
    block = mgr.build_context_block("currency")
    assert "dollar" in block.lower()


def test_malformed_messages_are_skipped():
    ana = AnalogicalMemory()
    mw = MemoryMiddleware(_manager(), analogical=ana)
    learned = mw.learn_from_exchange(
        ["not a dict", {"role": "user", "content": "Fido is a dog"}]  # type: ignore[list-item]
    )
    assert ("fido", "type", "dog") in {
        (t.subject, t.attribute, t.value) for t in learned
    }
