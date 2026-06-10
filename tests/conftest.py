"""
Shared test configuration for armos test suite.

Provides a module-level flag `using_fallback` that is True when the armos-ner
model could not be loaded from HuggingFace (e.g. due to rate limiting in CI)
and the fallback spaCy model is in use instead.

This is computed once at import time so it does not add extra DetectionEngine
instantiations on top of the ones the test modules already perform.
"""
from armos.detection.engine import DetectionEngine, _FALLBACK_MODEL

_engine = DetectionEngine()
using_fallback: bool = _engine._model_path == _FALLBACK_MODEL
