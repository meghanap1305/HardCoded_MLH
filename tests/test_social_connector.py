"""Unit tests for social ingestion adapter and Gemma 4 integration."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from hypelens.ingestion.social_connector import SocialIngestor
from hypelens.models.data_models import PlatformType
from hypelens.nlp.extractor import GemmaExtractor


def test_social_ingestor_user_post_manual(tmp_path: Path):
    """Verify manual post creation and entity matching with taxonomy."""
    ingestor = SocialIngestor(cache_dir=tmp_path)
    post = ingestor.create_user_post(
        text="Huge rumor: $RELIANCE Jio is teaming up with Tata Motors on EV batteries!",
        author="rumor_hound",
        platform="x_post",
        source_label="USER_UPLOAD",
    )
    assert post is not None
    assert post.platform == PlatformType.X_POST
    assert "RELIANCE" in post.tickers
    assert "TATAMOTORS" in post.tickers
    assert post.author.username == "rumor_hound"
    assert post.cognitive_provenance == "[USER_UPLOAD]"
    assert post.published_at is not None
    assert post.collected_at is not None


def test_social_ingestor_x_api_mocked_response(tmp_path: Path):
    """Verify official X API v2 payload parsing with mocked HTTP response."""
    ingestor = SocialIngestor(cache_dir=tmp_path)

    # Mock response format from X API v2
    fake_x_payload = {
        "data": [
            {
                "id": "1843920192019",
                "text": "Rumours circulating about $INFY securing a huge AI banking contract in Mumbai.",
                "created_at": "2026-10-09T08:30:00Z",
                "author_id": "998877",
                "public_metrics": {"like_count": 45, "retweet_count": 12, "reply_count": 5},
            },
            {
                "id": "1843920192020",
                "text": "Tata Motors EV commercial rollout seems ahead of schedule. $TATAMOTORS",
                "created_at": "2026-10-09T09:00:00Z",
                "author_id": "443322",
                "public_metrics": {"like_count": 120, "retweet_count": 30, "reply_count": 8},
            },
        ]
    }

    posts = ingestor.parse_x_api_json(fake_x_payload)
    assert len(posts) == 2

    infy_post = next(p for p in posts if "INFY" in p.tickers)
    assert infy_post.post_id == "x_1843920192019"
    assert infy_post.engagement["likes"] == 45
    assert infy_post.source_provider == "x_api_v2"
    assert infy_post.cognitive_provenance == "[LIVE_API]"

    tamo_post = next(p for p in posts if "TATAMOTORS" in p.tickers)
    assert tamo_post.post_id == "x_1843920192020"


def test_social_ingestor_x_api_auth_required_when_no_token(monkeypatch, tmp_path: Path):
    """Verify that fetch_x_recent_posts returns AUTH_REQUIRED without token."""
    monkeypatch.delenv("X_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("TWITTER_BEARER_TOKEN", raising=False)

    ingestor = SocialIngestor(cache_dir=tmp_path)
    res = ingestor.fetch_x_recent_posts()
    assert res["success"] is False
    assert res["status"] == "AUTH_REQUIRED"
    assert len(res["items"]) == 0


def test_social_ingestor_deduplication(tmp_path: Path):
    """Verify that duplicate tweets are ignored."""
    ingestor = SocialIngestor(cache_dir=tmp_path)
    payload = {
        "data": [
            {
                "id": "111",
                "text": "Same text repeat",
                "created_at": "2026-10-09T09:00:00Z",
            }
        ]
    }
    p1 = ingestor.parse_x_api_json(payload)
    p2 = ingestor.parse_x_api_json(payload)
    assert len(p1) == 1
    assert len(p2) == 0


def test_gemma_extractor_offline_mock_vs_live():
    """Verify GemmaExtractor returns clear mock flag when forced mock."""
    extractor = GemmaExtractor(force_mock=True)
    ingestor = SocialIngestor()
    post = ingestor.create_user_post(
        text="Whisper that $RELIANCE is buying an AI robotics firm soon.",
        author="tester",
    )
    res = extractor.extract_from_post(post)
    assert res.is_mock is True
    assert res.claim.event_category.value == "merger_acquisition"
    assert res.claim.status.value == "rumour_unverified"
    assert "[MOCK]" in res.claim.claim
