# SPDX-License-Identifier: MIT
from presidio_analyzer import Pattern, PatternRecognizer


class APIKeyRecogniser(PatternRecognizer):
    """Detects API keys and secrets developers accidentally include in prompts."""

    PATTERNS = [
        Pattern("openai_key",      r"\bsk-(?:proj-)?[A-Za-z0-9]{20,}\b", score=0.99),
        Pattern("anthropic_key",   r"\bsk-ant-[A-Za-z0-9\-_]{20,}\b",   score=0.99),
        Pattern("aws_access_key",  r"\bAKIA[A-Z0-9]{16}\b",              score=0.99),
        Pattern("aws_secret_key",
            r"(?:aws[_\-\s]*secret[_\-\s]*(?:access[_\-\s]*)?key|AWS_SECRET_ACCESS_KEY)\s*[=:\s]+['\"]?([A-Za-z0-9/+=]{40})['\"]?",
            score=0.95,
        ),
        Pattern("github_token",    r"\bghp_[A-Za-z0-9]{36,}\b",          score=0.99),
        Pattern("bearer_token",    r"\bBearer\s+[A-Za-z0-9\-_\.]{20,}\b", score=0.85),
        Pattern("generic_api_key",
            r"\b(?:api[_-]?key|apikey|api[_-]?secret)\s*[=:]\s*['\"]?([A-Za-z0-9\-_]{16,})['\"]?",
            score=0.85
        ),
    ]

    CONTEXT = [
        "api key", "api_key", "secret", "token",
        "bearer", "authorization", "auth", "credential"
    ]

    def __init__(self):
        super().__init__(
            supported_entity="API_KEY",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="en",
        )
