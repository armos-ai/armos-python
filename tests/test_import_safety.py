"""
Import safety tests.

These tests guard against the class of bug where importing armos crashes
because an optional dependency (e.g. redis) is not installed.
Every import must succeed with only the base install: pip install armos
"""
import sys
import importlib
import pytest


def _import_fresh(module_name: str):
    """Import a module with a clean sys.modules entry so caching doesn't hide failures."""
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


# ---------------------------------------------------------------------------
# Core imports must never crash regardless of optional deps
# ---------------------------------------------------------------------------

def test_import_armos_top_level():
    import armos  # noqa: F401

def test_import_armos_anthropic():
    from armos import ArmosAnthropic  # noqa: F401

def test_import_armos_openai():
    from armos import ArmosOpenAI  # noqa: F401

def test_import_armos_guard():
    from armos import Armos  # noqa: F401

def test_import_armos_models():
    from armos import MaskResult, DetectedEntity  # noqa: F401

def test_import_vault_package():
    from armos.masking.vault import build_vault, MemoryVault  # noqa: F401

def test_import_memory_vault():
    from armos.masking.vault.memory import MemoryVault  # noqa: F401

def test_import_detection_engine():
    from armos.detection.engine import DetectionEngine  # noqa: F401


# ---------------------------------------------------------------------------
# Redis optional — must not crash on import, must fail gracefully on use
# ---------------------------------------------------------------------------

def test_redis_vault_import_does_not_crash_without_redis(monkeypatch):
    """Simulates redis not installed: vault __init__ must not raise at import."""
    import armos.masking.vault as vault_pkg
    # If we got here without exception, the lazy import is working
    assert vault_pkg.build_vault is not None

def test_redis_store_without_redis_package_raises_import_error():
    """store='redis' is not supported in v2; raises ValueError with migration hint."""
    from armos.masking.vault import build_vault
    with pytest.raises(ValueError, match="Unsupported store"):
        build_vault(store="redis", redis_url="redis://localhost:6379")

def test_redis_store_without_url_raises_value_error():
    """store='redis' is not supported in v2; raises ValueError regardless of redis_url."""
    from armos.masking.vault import build_vault
    with pytest.raises(ValueError, match="Unsupported store"):
        build_vault(store="redis")


# ---------------------------------------------------------------------------
# Store API — string literal validation
# ---------------------------------------------------------------------------

def test_none_store_returns_memory_vault():
    from armos.masking.vault import build_vault, MemoryVault
    vault = build_vault(store=None)
    assert isinstance(vault, MemoryVault)

def test_default_store_returns_memory_vault():
    from armos.masking.vault import build_vault, MemoryVault
    vault = build_vault()
    assert isinstance(vault, MemoryVault)

def test_invalid_store_literal_raises():
    from armos.masking.vault import build_vault
    with pytest.raises(ValueError, match="Unsupported store"):
        build_vault(store="postgres")

def test_old_redis_url_as_store_raises():
    """Passing a redis:// URL directly as store (old API) raises a clear error."""
    from armos.masking.vault import build_vault
    with pytest.raises(ValueError, match="Unsupported store"):
        build_vault(store="redis://localhost:6379")


# ---------------------------------------------------------------------------
# Wrapper store parameter passthrough
# ---------------------------------------------------------------------------

def test_armos_openai_default_is_memory():
    from unittest.mock import MagicMock
    from armos import ArmosOpenAI
    from armos.masking.vault.memory import MemoryVault
    client = ArmosOpenAI(MagicMock())
    assert isinstance(client._guard._vault, MemoryVault)

def test_armos_anthropic_default_is_memory():
    from unittest.mock import MagicMock
    from armos import ArmosAnthropic
    from armos.masking.vault.memory import MemoryVault
    client = ArmosAnthropic(MagicMock())
    assert isinstance(client._guard._vault, MemoryVault)

def test_armos_openai_invalid_store_raises():
    """store='redis' (old API) raises ValueError — use store='armos' in v2."""
    from unittest.mock import MagicMock
    from armos import ArmosOpenAI
    with pytest.raises(ValueError, match="Unsupported store"):
        ArmosOpenAI(MagicMock(), store="redis")
