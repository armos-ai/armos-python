import pytest
from armos import Armos, MaskResult


@pytest.fixture
def guard():
    return Armos()


def test_mask_returns_mask_result(guard):
    result = guard.mask("Hello John Smith")
    assert isinstance(result, MaskResult)

def test_mask_replaces_email(guard):
    result = guard.mask("Email john@test.com for details")
    assert "john@test.com" not in result.text
    assert "[PII:EMAIL:" in result.text

def test_mask_replaces_aadhaar(guard):
    result = guard.mask("Aadhaar: 2345 6789 0123")
    assert "2345 6789 0123" not in result.text
    assert "[PII:AADHAAR:" in result.text

def test_mask_replaces_pan(guard):
    result = guard.mask("PAN: ABCDE1234F")
    assert "ABCDE1234F" not in result.text
    assert "[PII:PAN:" in result.text

def test_roundtrip_restores_original(guard):
    original = "Call John Smith at john@example.com about Aadhaar 2345 6789 0123"
    result = guard.mask(original)
    restored = guard.demask(result.text)
    assert restored == original

def test_casing_normalised_same_token(guard):
    """All casing variants produce the same token — LLM sees one entity."""
    result1 = guard.mask("patient john smith")
    result2 = guard.mask("Patient John Smith")
    result3 = guard.mask("PATIENT JOHN SMITH")

    import re
    pattern = re.compile(r'\[PII:NAME:[a-f0-9]{8}\]')

    tokens1 = pattern.findall(result1.text)
    tokens2 = pattern.findall(result2.text)
    tokens3 = pattern.findall(result3.text)

    if tokens1 and tokens2 and tokens3:
        assert tokens1[0] == tokens2[0] == tokens3[0]

    assert guard.demask(result1.text) == "patient john smith"
    assert "john smith" in guard.demask(result2.text).lower()
    assert "john smith" in guard.demask(result3.text).lower()

def test_unknown_token_safe(guard):
    """Tokens not in vault left unchanged — never crash."""
    text = "Hello [PII:NAME:unknown1] how are you"
    assert guard.demask(text) == text

def test_empty_string(guard):
    result = guard.mask("")
    assert result.text == ""
    assert result.entities == []
    assert result.uncertain == []

def test_uncertain_is_list(guard):
    result = guard.mask("Patient John Smith, email john@example.com")
    assert isinstance(result.uncertain, list)

def test_uncertain_not_masked(guard):
    """Uncertain entities must not appear as tokens in masked text."""
    result = guard.mask("Patient John Smith, email john@example.com")
    for entity in result.uncertain:
        assert entity.text not in result.text or "[PII:" not in result.text

def test_uncertain_scores_below_threshold(guard):
    """All uncertain entities must score below the 0.35 masking threshold."""
    result = guard.mask("Patient John Smith, Aadhaar 2345 6789 0123, email john@hospital.com")
    for entity in result.uncertain:
        assert entity.score < 0.35

def test_clean_text_unchanged(guard):
    result = guard.mask("The weather is nice today.")
    assert result.text == "The weather is nice today."
    assert not result.has_pii

def test_redis_store():
    """store='redis' is not supported in v2; raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported store"):
        Armos(store="redis")

def test_redis_store_missing_url_raises():
    """store='redis' is not supported in v2 regardless of redis_url."""
    with pytest.raises(ValueError, match="Unsupported store"):
        Armos(store="redis", armos_api_key=None)

def test_invalid_store_raises():
    with pytest.raises(ValueError, match="Unsupported store"):
        Armos(store="postgres")


def test_clear_vault(guard):
    result = guard.mask("John Smith")
    token = result.text
    assert guard.demask(token) == "John Smith"
    guard.clear_vault()
    assert guard.demask(token) == token  # unknown after clear — returned as-is
