"""OpenAI-compatible memory middleware for Kohaku.

Intercepts OpenAI-style ``messages`` lists, retrieves relevant memories from a
``ContextMemoryManager``, and injects them as a system message prefix.  The module
is self-contained and requires no external dependencies beyond ``kohaku`` itself.

When an :class:`~kohaku.AnalogicalMemory` is supplied, the middleware also mines
**structured facts** from the conversation as it flows — turning passing prose
("my flight seat preference is aisle") into ``(subject, attribute, value)``
records the agent can later *reason* over, not just recall. Facts are extracted
from **user** messages only by default: the user is the trustworthy source of
their own preferences and world, whereas assistant text may be model-generated.
Extraction is high-precision (see :mod:`kohaku.extraction`) — prose it can't
confidently parse contributes nothing, so no fabricated facts leak in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from kohaku.analogy import AnalogicalMemory
    from kohaku.context import ContextMemoryManager
    from kohaku.extraction import Triple


class MemoryMiddleware:
    """Intercepts OpenAI-style message lists, retrieves relevant memories, and
    injects them as a system message prefix.

    Usage::

        middleware = MemoryMiddleware(manager)
        augmented_messages = middleware.augment(messages)
        # messages is list[{"role": str, "content": str}]

    The middleware is stateless between calls except through the underlying
    ``ContextMemoryManager`` — it is safe to share a single instance across requests.
    """

    def __init__(
        self,
        manager: ContextMemoryManager,
        inject_as_role: str = "system",
        *,
        analogical: Optional[AnalogicalMemory] = None,
        learn_facts_from: str = "user",
    ) -> None:
        if learn_facts_from not in ("user", "assistant", "both"):
            raise ValueError("learn_facts_from must be 'user', 'assistant', or 'both'")
        self.manager = manager
        self.inject_as_role = inject_as_role
        self.analogical = analogical
        self.learn_facts_from = learn_facts_from

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def augment(self, messages: list[dict]) -> list[dict]:
        """Find the last user message, retrieve relevant memories, and prepend a
        system message with the context block when memories are found.

        Parameters
        ----------
        messages:
            OpenAI-style list of ``{"role": ..., "content": ...}`` dicts.

        Returns
        -------
        list[dict]
            A new list with an optional injected system message prepended.  The
            original list is never mutated.
        """
        if not messages:
            return list(messages)

        # Find the last user message for the retrieval query
        last_user_content: str | None = None
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                last_user_content = str(msg.get("content", ""))
                break

        if last_user_content is None:
            # No user message found — return unmodified copy
            return list(messages)

        context_block = self.manager.build_context_block(last_user_content)
        if not context_block:
            return list(messages)

        injected: dict = {"role": self.inject_as_role, "content": context_block}
        return [injected] + list(messages)

    def learn_from_exchange(self, messages: list[dict]) -> List[Triple]:
        """Store assistant responses episodically and mine structured facts.

        Scans the message list for consecutive user→assistant pairs and stores each
        assistant response in the ``ContextMemoryManager`` so future queries can
        retrieve it as a relevant memory. When an ``AnalogicalMemory`` was supplied,
        also extracts ``(subject, attribute, value)`` triples from the configured
        message roles into it, so later turns can *reason* over what was said.

        Parameters
        ----------
        messages:
            OpenAI-style list of ``{"role": ..., "content": ...}`` dicts.

        Returns
        -------
        list[Triple]
            The structured facts learned this call (empty when no analogical store
            is attached or nothing parsed). Never raises on unparseable prose.
        """
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "assistant":
                continue
            # Find the nearest preceding user message
            user_content: str | None = None
            for j in range(i - 1, -1, -1):
                prev = messages[j]
                if isinstance(prev, dict) and prev.get("role") == "user":
                    user_content = str(prev.get("content", ""))
                    break
            if user_content is None:
                continue
            assistant_content = str(msg.get("content", ""))
            if assistant_content:
                self.manager.store(
                    key=user_content,
                    value=assistant_content,
                    label="assistant_response",
                )
        return self._learn_facts(messages)

    def _learn_facts(self, messages: list[dict]) -> List[Triple]:
        """Extract triples from the configured roles into the analogical store."""
        if self.analogical is None:
            return []
        if self.learn_facts_from == "both":
            roles = {"user", "assistant"}
        else:
            roles = {self.learn_facts_from}
        learned: List[Triple] = []
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") not in roles:
                continue
            learned.extend(self.analogical.learn(str(msg.get("content", ""))))
        return learned
