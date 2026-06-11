# SPDX-License-Identifier: MIT
import re
import warnings
from pathlib import Path
from typing import List
import spacy
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from presidio_analyzer.nlp_engine.ner_model_configuration import NerModelConfiguration

try:
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover
    snapshot_download = None  # type: ignore[assignment]

from ..models import DetectedEntity
from .recognisers.aadhaar import AadhaarRecogniser
from .recognisers.pan import PANRecogniser
from .recognisers.standard import APIKeyRecogniser
from .recognisers.address import PhysicalAddressRecogniser

_URL_CREDS_RE = re.compile(r"://[^@\s]*$")

_HF_MODEL_ID = "armos-ai/en_armos_ner"
_FALLBACK_MODEL = "en_core_web_lg"
_CACHE_DIR = Path.home() / ".cache" / "armos" / "models"


def _is_cached() -> bool:
    """Return True if the model is already cached locally."""
    marker = _CACHE_DIR / f"models--{_HF_MODEL_ID.replace('/', '--')}"
    return marker.exists()


def _load_model_path() -> str:
    """
    Download armos-ai/en_armos_ner from HuggingFace (public, no token required)
    and cache it at ~/.cache/armos/models/.

    Returns the local filesystem path to the model directory so it can be
    passed directly to spaCy's NlpEngine as a model name.

    Falls back to en_core_web_lg if the download fails for any reason.
    """
    try:
        if snapshot_download is None:
            raise ImportError("huggingface_hub is not installed")

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

        import logging
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

        first_time = not _is_cached()
        if first_time:
            print("[armos] First-time setup: downloading armos-ner-en (~450 MB).")
            print("[armos] This happens once — model is cached for all future runs.")
            local_path = snapshot_download(
                repo_id=_HF_MODEL_ID,
                cache_dir=str(_CACHE_DIR),
                local_files_only=False,
            )
            print("[armos] Model ready.\n")
        else:
            local_path = snapshot_download(
                repo_id=_HF_MODEL_ID,
                cache_dir=str(_CACHE_DIR),
                local_files_only=True,
            )

        return local_path
    except Exception as exc:
        warnings.warn(
            f"[armos] Could not download {_HF_MODEL_ID!r} from HuggingFace "
            f"({exc!r}). Falling back to {_FALLBACK_MODEL!r}.",
            RuntimeWarning,
            stacklevel=2,
        )
        if not spacy.util.is_package(_FALLBACK_MODEL):
            print(
                f"[armos] Downloading spaCy model {_FALLBACK_MODEL!r} "
                "(first-time setup, ~560 MB)..."
            )
            spacy.cli.download(_FALLBACK_MODEL)
            print("[armos] Model ready.")
        return _FALLBACK_MODEL


ENTITY_TYPES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "AADHAAR_NUMBER",
    "IN_PAN",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "API_KEY",
    "US_SSN",
    "IBAN_CODE",
    "PHYSICAL_ADDRESS",
]

ENTITY_SHORT_CODES = {
    "PERSON":           "NAME",
    "EMAIL_ADDRESS":    "EMAIL",
    "PHONE_NUMBER":     "PHONE",
    "AADHAAR_NUMBER":   "AADHAAR",
    "IN_PAN":           "PAN",
    "CREDIT_CARD":      "CARD",
    "IP_ADDRESS":       "IP",
    "API_KEY":          "APIKEY",
    "US_SSN":           "SSN",
    "IBAN_CODE":        "IBAN",
    "PHYSICAL_ADDRESS": "ADDRESS",
}


class DetectionEngine:
    """
    Orchestrates all PII recognisers.
    Detection runs entirely locally — no text is sent anywhere.
    """

    def __init__(self):
        self._model_path = _load_model_path()
        self._analyzer = self._build_analyzer()

    def _build_analyzer(self) -> AnalyzerEngine:
        # Extend the default entity mapping so that ADDRESS entities produced
        # by armos-ai/en_armos_ner are routed to Presidio's PHYSICAL_ADDRESS type.
        # All other default mappings (PERSON, GPE→LOCATION, etc.) are preserved.
        #
        # PHYSICAL_ADDRESS is added to low_score_entity_names with a reduced
        # multiplier (0.43) so that spaCy-predicted addresses score ~0.37 — above
        # the 0.35 masking threshold but below phone/card regex patterns (~0.4+),
        # preventing the model's occasional phone-as-address false positives from
        # overriding correct PHONE_NUMBER detections in overlap resolution.
        # The high-confidence regex patterns in PhysicalAddressRecogniser (0.82–0.9)
        # always win for well-formed addresses.
        ner_config = NerModelConfiguration()
        mapping = dict(ner_config.model_to_presidio_entity_mapping)
        mapping["ADDRESS"] = "PHYSICAL_ADDRESS"
        ner_config.model_to_presidio_entity_mapping = mapping
        ner_config.low_score_entity_names = {"PHYSICAL_ADDRESS"}
        ner_config.low_confidence_score_multiplier = 0.43

        nlp_engine = SpacyNlpEngine(
            models=[{"lang_code": "en", "model_name": self._model_path}],
            ner_model_configuration=ner_config,
        )
        nlp_engine.load()

        registry = RecognizerRegistry()
        registry.load_predefined_recognizers(nlp_engine=nlp_engine)

        registry.add_recognizer(AadhaarRecogniser())
        registry.add_recognizer(PANRecogniser())
        registry.add_recognizer(APIKeyRecogniser())
        registry.add_recognizer(PhysicalAddressRecogniser())

        return AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)

    def detect(self, text: str, language: str = "en") -> List[DetectedEntity]:
        """Detect all PII in text. Returns entities sorted by position."""
        if not text or not text.strip():
            return []

        results = self._analyzer.analyze(
            text=text,
            entities=ENTITY_TYPES,
            language=language,
        )

        entities = [
            DetectedEntity(
                entity_type=ENTITY_SHORT_CODES.get(r.entity_type, r.entity_type),
                text=text[r.start:r.end],
                start=r.start,
                end=r.end,
                score=r.score,
            )
            for r in results
        ]

        entities = self._filter_entities(entities, text)
        entities.sort(key=lambda e: e.start)
        return self._resolve_overlaps(entities)

    def detect_all(self, text: str, language: str = "en") -> tuple:
        """
        Detect PII at a lower threshold to surface near-misses.
        Returns (certain, uncertain) where:
          certain   — score >= 0.35, will be masked
          uncertain — score 0.1–0.35, detected but not masked
        """
        if not text or not text.strip():
            return [], []

        results = self._analyzer.analyze(
            text=text,
            entities=ENTITY_TYPES,
            language=language,
            score_threshold=0.1,
        )

        certain, uncertain = [], []
        for r in results:
            entity = DetectedEntity(
                entity_type=ENTITY_SHORT_CODES.get(r.entity_type, r.entity_type),
                text=text[r.start:r.end],
                start=r.start,
                end=r.end,
                score=r.score,
            )
            if r.score >= 0.35:
                certain.append(entity)
            else:
                uncertain.append(entity)

        certain = self._filter_entities(certain, text)
        certain.sort(key=lambda e: e.start)
        uncertain.sort(key=lambda e: e.start)
        return self._resolve_overlaps(certain), uncertain

    def _filter_entities(self, entities: List[DetectedEntity], text: str = "") -> List[DetectedEntity]:
        """Post-filter to remove known false positive patterns."""
        filtered = []
        for e in entities:
            # Drop single bare-word ADDRESS — never a real address
            if e.entity_type == "ADDRESS" and " " not in e.text and "," not in e.text:
                continue
            # Drop EMAIL matches inside URL/connection-string credentials (://user:pass@host)
            if e.entity_type == "EMAIL" and text:
                preceding = text[max(0, e.start - 60):e.start]
                if _URL_CREDS_RE.search(preceding):
                    continue
            filtered.append(e)
        return filtered

    def _resolve_overlaps(self, entities: List[DetectedEntity]) -> List[DetectedEntity]:
        """Remove overlapping detections. Higher confidence wins.

        When two entities partially overlap and the lower-confidence entity
        starts clearly before the overlap region (i.e. they are distinct spans
        that merely touch), both are kept — only one is dropped when the overlap
        covers ≥50% of the shorter entity.  This prevents a model's over-wide
        span (e.g. PERSON consuming adjacent digits) from erasing an accurate
        neighbouring detection (e.g. AADHAAR).
        """
        if not entities:
            return entities

        resolved = [entities[0]]
        for current in entities[1:]:
            previous = resolved[-1]
            overlap = previous.end - current.start
            if overlap > 0:
                shorter_len = min(
                    previous.end - previous.start,
                    current.end - current.start,
                )
                if overlap >= shorter_len * 0.5:
                    # Substantial overlap — keep the higher-confidence entity.
                    if current.score > previous.score:
                        resolved[-1] = current
                    # else: keep previous, discard current (fall through without append)
                else:
                    # Minimal overlap (entities are mostly distinct spans) — keep both.
                    resolved.append(current)
            else:
                resolved.append(current)
        return resolved
