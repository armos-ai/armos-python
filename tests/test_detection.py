import pytest
from armos.detection.engine import DetectionEngine


@pytest.fixture(scope="module")
def engine():
    return DetectionEngine()


def test_detects_person_name(engine):
    entities = engine.detect("Please call John Smith immediately")
    assert any(e.entity_type == "NAME" for e in entities)

def test_detects_email(engine):
    entities = engine.detect("Send the report to john.smith@company.com")
    assert any(e.entity_type == "EMAIL" for e in entities)

def test_detects_indian_phone(engine):
    entities = engine.detect("Call me on +91 98765 43210")
    assert any(e.entity_type == "PHONE" for e in entities)

def test_detects_aadhaar_spaces(engine):
    entities = engine.detect("My Aadhaar is 2345 6789 0123")
    assert any(e.entity_type == "AADHAAR" for e in entities)

def test_detects_aadhaar_hyphens(engine):
    entities = engine.detect("UID: 2345-6789-0123")
    assert any(e.entity_type == "AADHAAR" for e in entities)

def test_detects_pan(engine):
    entities = engine.detect("PAN card: ABCDE1234F")
    assert any(e.entity_type == "PAN" for e in entities)

def test_detects_credit_card(engine):
    entities = engine.detect("Card: 4111 1111 1111 1111")
    assert any(e.entity_type == "CARD" for e in entities)

def test_detects_ip_address(engine):
    entities = engine.detect("Request from 192.168.1.100")
    assert any(e.entity_type == "IP" for e in entities)

def test_detects_openai_api_key(engine):
    entities = engine.detect("key is sk-abc123def456ghi789jkl012mno345pqr")
    assert any(e.entity_type == "APIKEY" for e in entities)

def test_detects_aws_key(engine):
    entities = engine.detect("AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE")
    assert any(e.entity_type == "APIKEY" for e in entities)

def test_detects_ssn(engine):
    entities = engine.detect("Employee SSN on file: 371-53-1234.")
    assert any(e.entity_type == "SSN" for e in entities)

def test_detects_iban(engine):
    entities = engine.detect("Please transfer to IBAN GB29NWBK60161331926819.")
    assert any(e.entity_type == "IBAN" for e in entities)

def test_detects_us_street_address(engine):
    entities = engine.detect("My address is 123 Main Street, please come by.")
    assert any(e.entity_type == "ADDRESS" for e in entities)

def test_detects_full_us_address(engine):
    entities = engine.detect("Ship to 456 Oak Avenue, San Francisco, CA 94102")
    assert any(e.entity_type == "ADDRESS" for e in entities)

def test_detects_po_box(engine):
    entities = engine.detect("Mailing address: P.O. Box 4521, New York")
    assert any(e.entity_type == "ADDRESS" for e in entities)

def test_detects_indian_flat_address(engine):
    entities = engine.detect("I live at Flat 4B, Koramangala, Bangalore")
    assert any(e.entity_type == "ADDRESS" for e in entities)

def test_address_masked_in_guard():
    from armos import Armos
    guard = Armos()
    result = guard.mask("Ship to 123 Oak Avenue, please process quickly")
    assert "[PII:ADDRESS:" in result.text

def test_no_false_positives(engine):
    entities = engine.detect("The weather in Bangalore is pleasant today.")
    assert len(entities) == 0

def test_empty_string(engine):
    assert engine.detect("") == []

def test_multiple_entities(engine):
    text = "Patient John Smith, Aadhaar 2345 6789 0123, email john@hospital.com"
    entities = engine.detect(text)
    assert len(entities) >= 3


def test_detect_all_empty_string(engine):
    certain, uncertain = engine.detect_all("")
    assert certain == []
    assert uncertain == []


def test_detect_all_whitespace_only(engine):
    certain, uncertain = engine.detect_all("   ")
    assert certain == []
    assert uncertain == []


def test_detect_all_returns_uncertain_for_low_score():
    """Entities scoring < 0.35 go to the uncertain list, not masked."""
    from unittest.mock import patch, MagicMock
    from armos.detection.engine import DetectionEngine, ENTITY_SHORT_CODES

    engine = DetectionEngine()
    mock_result = MagicMock()
    mock_result.entity_type = "PERSON"
    mock_result.start = 0
    mock_result.end = 4
    mock_result.score = 0.2

    with patch.object(engine._analyzer, "analyze", return_value=[mock_result]):
        certain, uncertain = engine.detect_all("John")

    assert len(certain) == 0
    assert len(uncertain) == 1
    assert uncertain[0].score == 0.2
    assert uncertain[0].entity_type == "NAME"


def test_resolve_overlaps_higher_score_wins(engine):
    """When two entities overlap, the higher-confidence one is kept."""
    from armos.models import DetectedEntity
    entities = [
        DetectedEntity(entity_type="NAME", text="John Smith", start=0, end=10, score=0.5),
        DetectedEntity(entity_type="NAME", text="John",       start=0, end=4,  score=0.85),
    ]
    resolved = engine._resolve_overlaps(entities)
    assert len(resolved) == 1
    assert resolved[0].score == 0.85


def test_detects_openai_sk_proj_key(engine):
    entities = engine.detect("key = sk-proj-abc123def456ghi789jkl012mno345pqr678stu")
    assert any(e.entity_type == "APIKEY" for e in entities)

def test_detects_aws_secret_key(engine):
    entities = engine.detect("aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
    assert any(e.entity_type == "APIKEY" for e in entities)

def test_no_false_positive_email_in_db_connection_string(engine):
    entities = engine.detect("postgresql://admin:password123@db.acme.com:5432/prod")
    assert not any(e.entity_type == "EMAIL" for e in entities)


def test_load_model_path_returns_hf_path_on_success():
    """_load_model_path() returns the HuggingFace snapshot path when download succeeds."""
    from unittest.mock import patch
    from armos.detection.engine import _load_model_path

    with patch("armos.detection.engine.snapshot_download", return_value="/tmp/fake/model") as mock_dl:
        path = _load_model_path()

    mock_dl.assert_called_once()
    assert path == "/tmp/fake/model"


def test_load_model_path_falls_back_to_spacy_on_error():
    """_load_model_path() falls back to en_core_web_lg when HF download fails."""
    from unittest.mock import patch
    from armos.detection.engine import _load_model_path, _FALLBACK_MODEL

    with patch("armos.detection.engine.snapshot_download", side_effect=Exception("network error")), \
         patch("armos.detection.engine.spacy.util.is_package", return_value=True):
        path = _load_model_path()

    assert path == _FALLBACK_MODEL


def test_load_model_path_downloads_spacy_fallback_when_missing():
    """When HF fails and en_core_web_lg is not installed, spacy.cli.download is called."""
    from unittest.mock import patch
    from armos.detection.engine import _load_model_path, _FALLBACK_MODEL

    with patch("armos.detection.engine.snapshot_download", side_effect=Exception("offline")), \
         patch("armos.detection.engine.spacy.util.is_package", return_value=False), \
         patch("armos.detection.engine.spacy.cli.download") as mock_dl:
        path = _load_model_path()

    mock_dl.assert_called_once_with(_FALLBACK_MODEL)
    assert path == _FALLBACK_MODEL
