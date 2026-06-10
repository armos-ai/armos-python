# SPDX-License-Identifier: Apache-2.0
import asyncio
from typing import Literal
from .models import MaskResult
from .detection.engine import DetectionEngine
from .masking.tokenizer import Tokenizer
from .masking.vault import build_vault
from .masking.vault.base import BaseVault

_DEFAULT_VAULT_TTL = 86400


class Armos:
    """
    Core privacy engine.

    Exposes mask() and demask() as standalone methods.
    Used internally by ArmosOpenAI and ArmosAnthropic wrappers.
    Can also be used directly for custom LLM integrations.
    """

    def __init__(
        self,
        store: Literal["armos"] | None = None,
        armos_api_key: str | None = None,
        armos_base_url: str = "https://proxy.armos.dev",
    ):
        self._vault: BaseVault = build_vault(
            store=store,
            api_key=armos_api_key,
            base_url=armos_base_url,
        )
        self._engine = DetectionEngine()
        self._tokenizer = Tokenizer(self._vault)

    def mask(self, text: str) -> MaskResult:
        """Detect and mask all PII in text."""
        if not text:
            return MaskResult(text=text or "", entities=[], uncertain=[])

        certain, uncertain = self._engine.detect_all(text)
        result = self._tokenizer.mask(text, certain)
        result.uncertain = uncertain
        return result

    def demask(self, text: str) -> str:
        """Restore all tokens in text to their real values."""
        return self._tokenizer.demask(text)

    async def amask(self, text: str) -> MaskResult:
        """Async version of mask(). Runs spaCy/Presidio detection in a thread pool."""
        return await asyncio.to_thread(self.mask, text)

    async def ademask(self, text: str) -> str:
        """Async version of demask(). Runs in a thread pool."""
        return await asyncio.to_thread(self.demask, text)

    def clear_vault(self) -> None:
        """Clear all stored token mappings."""
        self._vault.clear()
