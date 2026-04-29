"""OpenAI-compatible memory middleware for Kohaku.

Intercepts OpenAI-style ``messages`` lists, retrieves relevant memories from a
``ContextMemoryManager``, and injects them as a system message prefix.  The module
is self-contained and requires no external dependencies beyond ``kohaku`` itself.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kohaku.context import ContextMemoryManager


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
        manager: "ContextMemoryManager",
        inject_as_role: str = "system",
    ) -> None:
        self.manager = manager
        self.inject_as_role = inject_as_role

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

    def learn_from_exchange(self, messages: list[dict]) -> None:
        """Store assistant responses as memories keyed by the preceding user message.

        Scans the message list for consecutive user→assistant pairs and stores each
        assistant response in the ``ContextMemoryManager`` so future queries can
        retrieve it as a relevant memory.

        Parameters
        ----------
        messages:
            OpenAI-style list of ``{"role": ..., "content": ...}`` dicts.
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
