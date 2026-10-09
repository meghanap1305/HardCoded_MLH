"""Unit tests for Gemma 4 multimodal extractor client and mock fallback (TASK-05)."""

from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pytest
import requests

from hypelens.config import FIXTURES_DIR, OllamaConfig
from hypelens.models.data_models import AccountTier, AuthorMetadata, PlatformType, SocialPost
from hypelens.nlp.extractor import GemmaExtractor, encode_image_to_base64
from hypelens.nlp.parser import EpistemicStatus, EventCategory


def test_encode_image_to_base64():
    """Verify base64 encoding on real image and missing file handling."""
    img_path = FIXTURES_DIR / "sample_images" / "nvda_leak_chart.png"
    assert img_path.exists()
    encoded = encode_image_to_base64(img_path)
    assert encoded is not None
    assert len(encoded) > 50

    # Missing file
    assert encode_image_to_base64("non_existent_image.png") is None


def test_extractor_forced_mock():
    """Verify deterministic mock extractor outputs explicit mock flags."""
    extractor = GemmaExtractor(force_mock=True)
    res = extractor.extract(
        content_text="Major partnership rumors on $NVDA",
        tickers=["NVDA"],
        ocr_text="LEAKED SLIDE",
    )
    assert res.is_mock is True
    assert res.model_used == "mock-gemma4"
    assert res.claim.ticker == "NVDA"
    assert res.claim.status == EpistemicStatus.RUMOUR_UNVERIFIED
    assert res.claim.event_category == EventCategory.MERGER_ACQUISITION
    assert res.error is None


def test_extractor_from_social_post():
    """Verify extractor helper on a validated SocialPost model."""
    post = SocialPost(
        post_id="p1",
        platform=PlatformType.X_POST,
        timestamp=datetime.now(timezone.utc),
        author=AuthorMetadata(author_id="a1", username="trader", tier=AccountTier.MEDIUM),
        content_text="Hearing huge rumors about $TSLA robotaxi permits",
        tickers=["TSLA"],
    )
    extractor = GemmaExtractor(force_mock=True)
    res = extractor.extract_from_post(post)
    assert res.is_mock is True
    assert res.claim.ticker == "TSLA"
    assert res.claim.event_category == EventCategory.REGULATORY


def test_extractor_mocked_http_success():
    """Verify extractor parses live Gemma 4 JSON response when Ollama responds 200 OK."""
    cfg = OllamaConfig(host="http://localhost:11434", model="gemma4")
    extractor = GemmaExtractor(config=cfg, force_mock=False)

    fake_response = {
        "response": '{"ticker": "NVDA", "claim": "Cloud partnership deal", "event_category": "merger_acquisition", "status": "officially_confirmed", "sentiment": 0.8, "intensity": 0.85, "confidence": 0.9}'
    }

    with mock.patch("requests.post") as mock_post:
        mock_resp = mock.Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_response
        mock_post.return_value = mock_resp

        img_path = FIXTURES_DIR / "sample_images" / "nvda_leak_chart.png"
        res = extractor.extract(
            content_text="Official confirmation on $NVDA",
            tickers=["NVDA"],
            image_paths=[img_path],
        )

        assert res.is_mock is False
        assert res.model_used == "gemma4"
        assert res.claim.ticker == "NVDA"
        assert res.claim.status == EpistemicStatus.OFFICIALLY_CONFIRMED

        # Check call arguments
        assert mock_post.called
        sent_json = mock_post.call_args[1]["json"]
        assert sent_json["model"] == "gemma4"
        assert "images" in sent_json
        assert len(sent_json["images"]) == 1


def test_extractor_fallback_on_network_error():
    """Verify graceful fallback to mock when Ollama connection drops."""
    cfg = OllamaConfig(allow_mock_fallback=True)
    extractor = GemmaExtractor(config=cfg, force_mock=False)

    with mock.patch("requests.post", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        res = extractor.extract(
            content_text="Major rumors on $NVDA",
            tickers=["NVDA"],
        )
        assert res.is_mock is True
        assert res.model_used == "mock-gemma4"
        assert res.error is not None
        assert "communication failure" in res.error.lower()
        assert res.claim.ticker == "NVDA"


def test_extractor_no_fallback_when_disabled():
    """Verify fallback can be disabled to detect hard failures."""
    cfg = OllamaConfig(allow_mock_fallback=False)
    extractor = GemmaExtractor(config=cfg, force_mock=False)

    with mock.patch("requests.post", side_effect=requests.exceptions.Timeout("Timeout")):
        res = extractor.extract(
            content_text="Some text",
            tickers=["NVDA"],
        )
        assert res.is_mock is False
        assert res.claim.confidence == 0.0
        assert res.error is not None
