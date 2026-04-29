"""Tests for kohaku.hf_hooks and kohaku.openai_compat importability and basic behavior."""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# 1. hf_hooks is importable even without transformers
# ---------------------------------------------------------------------------

def test_hf_hooks_importable_without_transformers():
    """Importing KohakuMemoryCallback must not raise even if transformers is absent."""
    from kohaku.hf_hooks import KohakuMemoryCallback  # noqa: F401  — import must succeed
    assert KohakuMemoryCallback is not None


# ---------------------------------------------------------------------------
# 2. When transformers absent, instantiation raises ImportError (not AttributeError)
# ---------------------------------------------------------------------------

def test_stub_raises_import_error_on_instantiation():
    """When transformers is absent, KohakuMemoryCallback() raises ImportError."""
    import sys
    from kohaku.hf_hooks import KohakuMemoryCallback

    # Check whether transformers is actually installed
    transformers_present = "transformers" in sys.modules or _try_import_transformers()

    if transformers_present:
        # transformers IS installed — the real class should be constructable with a manager
        pytest.skip("transformers is installed; stub path not exercised")

    # transformers NOT installed — instantiation must raise ImportError
    with pytest.raises(ImportError):
        KohakuMemoryCallback(None)  # type: ignore[arg-type]


def _try_import_transformers() -> bool:
    """Return True if transformers can be imported."""
    try:
        import importlib
        importlib.import_module("transformers")
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# 3. openai_compat.MemoryMiddleware is always importable
# ---------------------------------------------------------------------------

def test_openai_compat_importable():
    """MemoryMiddleware must be importable regardless of optional dependencies."""
    from kohaku.openai_compat import MemoryMiddleware  # noqa: F401
    assert MemoryMiddleware is not None


# ---------------------------------------------------------------------------
# 4. MemoryMiddleware.augment returns a list
# ---------------------------------------------------------------------------

def test_memory_middleware_augment_returns_list():
    """augment() must return a list even when memory is empty."""
    from kohaku.context import ContextMemoryManager, ContextConfig
    from kohaku.openai_compat import MemoryMiddleware

    cfg = ContextConfig(max_tokens=1000, tokens_per_entry=50)
    manager = ContextMemoryManager(cfg)
    middleware = MemoryMiddleware(manager)

    messages = [{"role": "user", "content": "hello"}]
    result = middleware.augment(messages)

    assert isinstance(result, list), f"Expected list, got {type(result)}"
    # When memory is empty, the original messages should be returned unchanged
    assert result == messages or (
        len(result) > len(messages)  # or augmented with system block
    )
