# SPDX-License-Identifier: Apache-2.0
from .base import BaseVault
from .memory import MemoryVault
from .armos_vault import ArmosVault

__all__ = ["BaseVault", "MemoryVault", "ArmosVault"]


def build_vault(
    store: str | None = None,
    api_key: str | None = None,
    base_url: str = "https://proxy.armos.dev",
    **_kwargs,
) -> BaseVault:
    """
    Factory function. Builds the appropriate vault from the *store* argument.

    Args:
        store:    ``None`` for in-memory (default), or ``"armos"`` for the
                  client-side encrypted Armos cloud vault.
        api_key:  Armos API key (required when ``store="armos"``).
        base_url: Armos proxy base URL (optional, used when ``store="armos"``).
    """
    if store is None:
        return MemoryVault()

    if store == "armos":
        if not api_key:
            raise ValueError(
                "api_key is required when store='armos'. "
                "Example: Armos(store='armos', armos_api_key='your-key')"
            )
        return ArmosVault(api_key=api_key, base_url=base_url)

    raise ValueError(
        f"Unsupported store: '{store}'. Valid options: None (in-memory) or 'armos'."
    )
