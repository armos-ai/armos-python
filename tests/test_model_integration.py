"""
Integration tests for armos-ner-en model.

These tests verify that:
  1. The correct model (armos-ner-en) is loaded, not en_core_web_lg.
  2. Indian name detection meets our published accuracy improvement.
  3. Indian and Western ADDRESS detection works via NER (not just regex).
  4. New API key formats (sk-proj-, long ghp_) are caught.
  5. Adjacent entities (PERSON + AADHAAR) are not dropped by overlap resolution.
  6. False positives (room numbers, section refs) are not triggered.

If these tests fail after a model update, it means the new model regressed
on one of our published benchmarks and should NOT be released.
"""
import pytest
from armos.detection.engine import DetectionEngine, _HF_MODEL_ID, _FALLBACK_MODEL
from tests.conftest import using_fallback

_skip_if_fallback = pytest.mark.skipif(
    using_fallback,
    reason=(
        f"armos-ner not available (HuggingFace rate-limited or unreachable); "
        f"engine is using fallback model '{_FALLBACK_MODEL}'"
    ),
)


@pytest.fixture(scope="module")
def engine():
    # Reuse the DetectionEngine already instantiated in conftest to avoid a
    # second slow model load; conftest._engine is the same object.
    from tests.conftest import _engine
    return _engine


# ── Model identity ─────────────────────────────────────────────────────────────

@_skip_if_fallback
def test_correct_model_is_loaded(engine):
    """Armos must use armos-ner-en, not en_core_web_lg."""
    assert engine._model_path != _FALLBACK_MODEL, (
        f"Expected armos-ner-en but got fallback model '{_FALLBACK_MODEL}'. "
        "Check HuggingFace connectivity and model availability."
    )
    assert "armos" in engine._model_path.lower() or _HF_MODEL_ID.replace("/", "--") in engine._model_path


# ── Indian name detection (armos-ner-en improvement over baseline) ────────────

@pytest.mark.parametrize("text,expected_name", [
    # Common Indian names
    ("Patient Priya Sharma was admitted to the ICU.", "Priya Sharma"),
    ("Please contact Rahul Mehta regarding the delay.", "Rahul Mehta"),
    ("KYC documents for Ananya Singh are pending.", "Ananya Singh"),
    # Hard Indian names — en_core_web_lg misses these
    ("The agreement was signed by Venkataraman Subramaniam.", "Venkataraman Subramaniam"),
    ("Aishwarya Rao has been prescribed medication.", "Aishwarya Rao"),
    ("HR file for Harshavardhan Kulkarni updated.", "Harshavardhan Kulkarni"),
    pytest.param(
        "Salary processed for Lakshmipriya Balasubramanian.",
        "Lakshmipriya Balasubramanian",
        marks=_skip_if_fallback,
    ),
])
def test_indian_name_detected(engine, text, expected_name):
    entities = engine.detect(text)
    names = [e.text for e in entities if e.entity_type == "NAME"]
    assert any(expected_name.lower() in n.lower() or n.lower() in expected_name.lower() for n in names), (
        f"Expected '{expected_name}' to be detected in: {text!r}\nGot: {names}"
    )


# ── Western name detection ─────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected_name", [
    ("Employee John Smith has submitted their resignation.", "John Smith"),
    ("The offer letter for Sarah Johnson has been sent.", "Sarah Johnson"),
    ("Background check for Michael Garcia completed.", "Michael Garcia"),
])
def test_western_name_detected(engine, text, expected_name):
    entities = engine.detect(text)
    names = [e.text for e in entities if e.entity_type == "NAME"]
    assert any(expected_name.lower() in n.lower() or n.lower() in expected_name.lower() for n in names), (
        f"Expected '{expected_name}' in: {text!r}\nGot: {names}"
    )


# ── ADDRESS detection (NER-based, not regex) ───────────────────────────────────

@pytest.mark.parametrize("text", [
    "Deliver the package to Flat 4B, Koramangala, Bangalore 560095.",
    "KYC address: H.No. 42, Sector 15, Gurgaon, Haryana 122001.",
    pytest.param(
        "Ship to Unit 5A, Sunrise Society, HSR Layout, Bangalore 560102.",
        marks=_skip_if_fallback,
    ),
    "Ship the order to 123 Oak Avenue, Chicago, IL 60601.",
    "Registered office: 456 Elm Street, Brooklyn, NY 11201.",
    "Property at 22 Baker Street, London, W1U 3BW.",
])
def test_address_detected(engine, text):
    entities = engine.detect(text)
    assert any(e.entity_type == "ADDRESS" for e in entities), (
        f"Expected ADDRESS entity in: {text!r}\nGot: {[e.entity_type for e in entities]}"
    )


# ── API key formats ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    # Classic OpenAI key
    "Using key sk-abc123def456ghi789jkl012mno345pqr678stu901vwx.",
    # New sk-proj- format
    "API key: sk-proj-abc123XYZ456abc123XYZ456abc123XYZ456abc123XYZ.",
    # Anthropic
    "Auth: sk-ant-api03-abc123def456ghi789jkl012mno345pqrst.",
    # AWS
    "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE",
    # GitHub PAT (36+ chars)
    "Token: ghp_abc123XYZ456abc123XYZ456abc123XYZ456.",
])
def test_api_key_detected(engine, text):
    entities = engine.detect(text)
    assert any(e.entity_type == "APIKEY" for e in entities), (
        f"Expected APIKEY entity in: {text!r}\nGot: {[e.entity_type for e in entities]}"
    )


# ── Overlap resolution — adjacent entities must not be dropped ─────────────────

def test_person_and_aadhaar_both_detected(engine):
    """PERSON and AADHAAR in close proximity — both must be masked, neither dropped."""
    entities = engine.detect(
        "Patient Priya Sharma, Aadhaar 2345 6789 0123, is admitted."
    )
    types = {e.entity_type for e in entities}
    assert "NAME" in types, f"NAME not detected. Got: {types}"
    assert "AADHAAR" in types, f"AADHAAR dropped by overlap resolution. Got: {types}"


def test_person_and_pan_both_detected(engine):
    entities = engine.detect(
        "ITR filed by Rahul Mehta under PAN ABCDE1234F."
    )
    types = {e.entity_type for e in entities}
    assert "NAME" in types, f"NAME not detected. Got: {types}"
    assert "PAN" in types, f"PAN dropped by overlap resolution. Got: {types}"


# ── False positive prevention ──────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "The meeting is in Room B.",
    "Please refer to Section 80C for tax details.",
    "Refer to Chapter 12A of the policy manual.",
    pytest.param("The patient is in Ward 7B of the hospital.", marks=_skip_if_fallback),
    "The server is running on Port 8080.",
    "Error 404: resource not found.",
    "Q3 revenue was up 12% year over year.",
    "IP address 192.168.1.1 has been blocked.",
])
def test_no_false_positive(engine, text):
    entities = engine.detect(text)
    addr_or_name = [e for e in entities if e.entity_type in ("ADDRESS", "NAME")]
    assert not addr_or_name, (
        f"False positive in: {text!r}\nGot: {[(e.text, e.entity_type) for e in addr_or_name]}"
    )
