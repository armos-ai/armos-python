# SPDX-License-Identifier: Apache-2.0
"""
Armos — Automatic PII masking for OpenAI and Anthropic SDKs.

One line change. PII never reaches your LLM provider.

Quick start:
    from openai import OpenAI
    from armos import ArmosOpenAI

    client = ArmosOpenAI(OpenAI())
    # Use exactly as you would the normal OpenAI client.
    # PII is masked automatically before every request.
    # Real values are restored automatically in every response.
"""

from .guard import Armos
from .wrappers.openai import ArmosOpenAI, ArmosAsyncOpenAI
from .wrappers.anthropic import ArmosAnthropic, ArmosAsyncAnthropic
from .models import MaskResult, DetectedEntity
from .masking.vault import ArmosVault

__version__ = "2.0.0"
__all__ = [
    "Armos",
    "ArmosOpenAI",
    "ArmosAsyncOpenAI",
    "ArmosAnthropic",
    "ArmosAsyncAnthropic",
    "MaskResult",
    "DetectedEntity",
    "ArmosVault",
]
