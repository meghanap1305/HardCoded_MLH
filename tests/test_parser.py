"""Unit tests for Gemma 4 prompt engineering and JSON parsing (TASK-04)."""

from hypelens.nlp.parser import (
    EpistemicStatus,
    EventCategory,
    ExtractedClaim,
    clean_and_extract_json,
    parse_gemma_extraction,
)
from hypelens.nlp.prompts import (
    GEMMA_EXTRACTION_SYSTEM_PROMPT,
    build_extraction_prompt,
)


def test_build_extraction_prompt_multimodal():
    """Verify prompt formatting with multimodal text, OCR, and transcript inputs."""
    prompt = build_extraction_prompt(
        content_text="Major partnership rumors on $NVDA",
        ocr_text="LEAKED SLIDE CONFIDENTIAL",
        transcript="We expect a deal announcement this week.",
        has_images=True,
    )
    assert "$NVDA" in prompt
    assert "LEAKED SLIDE CONFIDENTIAL" in prompt
    assert "SPOKEN TRANSCRIPT" in prompt
    assert "ATTACHED VISUALS" in prompt
    assert "output valid JSON" in prompt
    assert len(GEMMA_EXTRACTION_SYSTEM_PROMPT) > 50


def test_clean_and_extract_json_direct():
    """Verify clean JSON extraction from raw JSON string."""
    raw = '{"ticker": "NVDA", "claim": "Cloud deal rumor", "sentiment": 0.85}'
    res = clean_and_extract_json(raw)
    assert res["ticker"] == "NVDA"
    assert res["sentiment"] == 0.85


def test_clean_and_extract_json_markdown_wrapped():
    """Verify extraction from markdown code fences and commentary prose."""
    raw = """Here is the extracted analysis:
```json
{
  "ticker": "$TSLA",
  "claim": "Robotaxi permit filing leak",
  "event_category": "regulatory",
  "status": "rumour_unverified",
  "sentiment": 0.7,
  "intensity": 0.85,
  "aspects": {"growth": 0.9, "valuation": -0.2},
  "is_promotional_spam": false,
  "confidence": 0.88
}
```
Hope this is helpful!"""
    res = clean_and_extract_json(raw)
    assert res["ticker"] == "$TSLA"
    assert res["event_category"] == "regulatory"
    assert res["status"] == "rumour_unverified"


def test_clean_and_extract_json_trailing_commas():
    """Verify heuristic repair of trailing commas."""
    raw = '{"ticker": "AAPL", "sentiment": 0.4, "aspects": {"product": 0.5, }, }'
    res = clean_and_extract_json(raw)
    assert res["ticker"] == "AAPL"
    assert res["aspects"]["product"] == 0.5


def test_parse_gemma_extraction_complete():
    """Verify mapping of clean payload into typed ExtractedClaim."""
    raw = """{
      "ticker": "$NVDA",
      "claim": "Hyperscaler multi-billion partnership",
      "event_category": "merger_acquisition",
      "status": "credible_report",
      "sentiment": 0.75,
      "intensity": 0.9,
      "aspects": {"growth": 0.8},
      "is_promotional_spam": false,
      "confidence": 0.85
    }"""
    claim = parse_gemma_extraction(raw)
    assert isinstance(claim, ExtractedClaim)
    assert claim.ticker == "NVDA"
    assert claim.event_category == EventCategory.MERGER_ACQUISITION
    assert claim.status == EpistemicStatus.CREDIBLE_REPORT
    assert claim.sentiment == 0.75
    assert claim.intensity == 0.9
    assert claim.confidence == 0.85
    assert claim.is_promotional_spam is False


def test_parse_gemma_extraction_clamping():
    """Verify numerical bounds clamping for sentiment, intensity, and confidence."""
    raw = """{
      "ticker": "TSLA",
      "sentiment": 2.5,
      "intensity": -0.5,
      "confidence": 1.5,
      "is_promotional_spam": true
    }"""
    claim = parse_gemma_extraction(raw)
    assert claim.sentiment == 1.0
    assert claim.intensity == 0.0
    assert claim.confidence == 1.0
    assert claim.is_promotional_spam is True


def test_parse_gemma_extraction_garbage_fallback():
    """Verify graceful fallback without exceptions when LLM outputs unparseable text."""
    claim = parse_gemma_extraction("I apologize, I am unable to process this request.", fallback_ticker="AAPL")
    assert isinstance(claim, ExtractedClaim)
    assert claim.ticker == "AAPL"
    assert claim.confidence == 0.0
    assert claim.status == EpistemicStatus.RUMOUR_UNVERIFIED
    assert claim.event_category == EventCategory.OTHER
