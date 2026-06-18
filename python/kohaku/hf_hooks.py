"""HuggingFace Transformers integration hooks for Kohaku.

This module is **always importable** even when ``transformers`` is not installed.
Instantiating ``KohakuMemoryCallback`` without transformers raises ``ImportError``
at construction time, not at import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kohaku.context import ContextMemoryManager


# ---------------------------------------------------------------------------
# Stub — always importable
# ---------------------------------------------------------------------------


class KohakuMemoryCallbackStub:
    """Importable stub when transformers is not installed.

    Raises ``ImportError`` on instantiation so callers get a clear, actionable
    error message instead of a confusing ``AttributeError``.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise ImportError(
            "transformers is required for KohakuMemoryCallback. "
            "Install it with: pip install transformers"
        )


# ---------------------------------------------------------------------------
# Real implementation — only defined when transformers is available
# ---------------------------------------------------------------------------

try:
    from transformers import TrainerCallback  # type: ignore[import]

    class KohakuMemoryCallback(TrainerCallback):  # type: ignore[misc]
        """HuggingFace ``TrainerCallback`` that stores attention activations as hypervectors.

        Attach to a ``Trainer`` to automatically record training metrics and (optionally)
        mean attention activations into a ``ContextMemoryManager`` for later retrieval.

        Usage::

            from kohaku.hf_hooks import KohakuMemoryCallback
            callback = KohakuMemoryCallback(memory_manager)
            trainer = Trainer(..., callbacks=[callback])
        """

        def __init__(
            self,
            manager: "ContextMemoryManager",
            store_every_n_steps: int = 100,
        ) -> None:
            super().__init__()
            self.manager = manager
            self.store_every_n_steps = store_every_n_steps
            self._step: int = 0

        def on_step_end(
            self,
            args: object,
            state: object,
            control: object,
            **kwargs: object,
        ) -> None:
            """Called by HF Trainer after every optimizer step.

            If ``model`` is present in kwargs and has an ``encoder`` with
            ``attentions``, stores the mean attention of the last layer.  Otherwise
            records a step-counter entry so the callback is still useful without a
            specific model architecture.
            """
            self._step += 1
            if self._step % self.store_every_n_steps != 0:
                return

            label = f"step_{self._step}"

            # Attempt to extract mean attention from the last forward pass outputs.
            # HF models expose ``attentions`` on their output objects when
            # ``output_attentions=True`` is set; we record the mean if available.
            try:
                import numpy as np  # type: ignore[import]

                outputs = kwargs.get("outputs")
                if (
                    outputs is not None
                    and hasattr(outputs, "attentions")
                    and outputs.attentions
                ):
                    last_layer = outputs.attentions[-1]
                    # last_layer shape: (batch, heads, seq, seq)
                    mean_attn = float(
                        np.mean(last_layer.detach().cpu().float().numpy())
                    )
                    self.manager.store(
                        key=label,
                        value=f"mean_attention={mean_attn:.6f}",
                        label=label,
                    )
                    return
            except Exception:
                pass

            # Fallback: record step existence
            self.manager.store(
                key=label,
                value=f"training_step={self._step}",
                label=label,
            )

        def on_log(
            self,
            args: object,
            state: object,
            control: object,
            logs: dict | None = None,
            **kwargs: object,
        ) -> None:
            """Called by HF Trainer when metrics are logged.

            Stores the log dict as a stringified memory entry keyed by the current
            global step so metrics can be retrieved later for analysis.
            """
            if not logs:
                return
            step_key = f"log_step_{getattr(state, 'global_step', self._step)}"
            # Serialize most important metrics into a compact string
            parts = [
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in logs.items()
            ]
            value = " | ".join(parts)
            self.manager.store(key=step_key, value=value, label=step_key)

except ImportError:
    # transformers not installed — fall back to stub so the name is always bound
    KohakuMemoryCallback = KohakuMemoryCallbackStub  # type: ignore[misc,assignment]
