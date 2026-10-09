"""Unit tests for multi-stream data models and validation schemas (TASK-02)."""

from datetime import datetime, timezone

from hypelens.models.data_models import (
    AccountTier,
    PlatformType,
    parse_utc_timestamp,
    validate_financial_news,
    validate_market_bar,
    validate_social_post,
)


def test_parse_utc_timestamp_formats():
    """Verify robust UTC parsing across multiple date and numeric formats."""
    # ISO with Z
    dt1 = parse_utc_timestamp("2026-10-09T10:00:00Z")
    assert dt1 == datetime(2026, 10, 9, 10, 0, 0, tzinfo=timezone.utc)

    # ISO with positive offset (+05:30)
    dt2 = parse_utc_timestamp("2026-10-09T15:30:00+05:30")
    assert dt2 == datetime(2026, 10, 9, 10, 0, 0, tzinfo=timezone.utc)

    # Unix epoch seconds
    dt3 = parse_utc_timestamp(1700000000)
    assert dt3.tzinfo == timezone.utc

    # Unix epoch milliseconds
    dt4 = parse_utc_timestamp(1700000000000)
    assert dt4.tzinfo == timezone.utc

    # Invalid / empty
    assert parse_utc_timestamp(None) is None
    assert parse_utc_timestamp("not-a-timestamp") is None
    assert parse_utc_timestamp("") is None


def test_validate_social_post_x_success():
    """Verify validation of a compliant X post with author metadata."""
    payload = {
        "post_id": "x_101",
        "platform": "x_post",
        "timestamp": "2026-10-09T08:00:00Z",
        "author": {
            "author_id": "auth_1",
            "username": "tech_trader",
            "tier": "large",
            "is_official": False,
            "follower_count": 250000,
        },
        "content_text": "Massive rumors about $NVDA custom AI silicon deals!",
        "tickers": ["$NVDA"],
        "engagement": {"reposts": 450, "likes": 1200},
    }

    result = validate_social_post(payload)
    assert result.is_valid is True
    assert result.quarantined is False
    post = result.record
    assert post is not None
    assert post.post_id == "x_101"
    assert post.platform == PlatformType.X_POST
    assert post.author.tier == AccountTier.LARGE
    assert post.author.is_official is False
    assert post.tickers == ["NVDA"]
    assert post.engagement["likes"] == 1200
    assert post.raw_payload == payload


def test_validate_social_post_instagram_reel_multimodal():
    """Verify validation of an Instagram Reel with video and OCR fields."""
    payload = {
        "post_id": "reel_202",
        "platform": "instagram_reel",
        "timestamp": "2026-10-09T09:15:00Z",
        "author": {
            "username": "daily_stocks",
            "tier": "small",
            "is_official": False,
        },
        "content_text": "Check out this leaked chart! $TSLA next move.",
        "tickers": ["TSLA"],
        "video_path": "fixtures/sample_videos/tsla_reel.mp4",
        "image_paths": ["fixtures/sample_images/frame_01.png"],
        "ocr_text": "BREAKING: TSLA PARTNERSHIP RUMOR",
        "transcript": "Guys look at this volume surge before any announcement.",
    }

    result = validate_social_post(payload)
    assert result.is_valid is True
    post = result.record
    assert post.platform == PlatformType.INSTAGRAM_REEL
    assert post.video_path == "fixtures/sample_videos/tsla_reel.mp4"
    assert post.ocr_text == "BREAKING: TSLA PARTNERSHIP RUMOR"
    assert post.transcript is not None


def test_social_post_quarantine_on_missing_timestamp():
    """Ensure posts missing timestamps are quarantined to avoid leakage."""
    payload = {
        "post_id": "x_bad_ts",
        "platform": "x_post",
        "timestamp": None,
        "content_text": "Undated rumour post",
    }
    result = validate_social_post(payload)
    assert result.is_valid is False
    assert result.quarantined is True
    assert "quarantined" in result.errors[0].lower()


def test_social_post_missing_post_id():
    """Ensure missing required post_id causes validation rejection."""
    payload = {
        "timestamp": "2026-10-09T10:00:00Z",
        "content_text": "Missing ID",
    }
    result = validate_social_post(payload)
    assert result.is_valid is False
    assert result.quarantined is False
    assert any("post_id" in err for err in result.errors)


def test_validate_financial_news_success():
    """Verify compliant financial news item validation."""
    payload = {
        "news_id": "news_901",
        "headline": "Tech Corp announces strategic expansion",
        "publisher": "Reuters",
        "timestamp": "2026-10-09T11:00:00Z",
        "body": "Official press release confirmed early today.",
        "tickers": ["AAPL"],
    }
    result = validate_financial_news(payload)
    assert result.is_valid is True
    assert result.quarantined is False
    item = result.record
    assert item.news_id == "news_901"
    assert item.publisher == "Reuters"
    assert item.tickers == ["AAPL"]


def test_financial_news_quarantine_on_bad_timestamp():
    """Verify news items with missing timestamp are quarantined."""
    payload = {
        "news_id": "news_bad",
        "headline": "No timestamp headline",
        "publisher": "Unknown",
        "timestamp": "invalid_date",
    }
    result = validate_financial_news(payload)
    assert result.is_valid is False
    assert result.quarantined is True


def test_validate_market_bar_success():
    """Verify compliant OHLCV market bar validation."""
    payload = {
        "ticker": "$NVDA",
        "timestamp": "2026-10-09T14:30:00Z",
        "open": 120.5,
        "high": 124.0,
        "low": 120.0,
        "close": 123.5,
        "volume": 55000000.0,
        "benchmark_return": 0.004,
        "abnormal_volume": 2.1,
    }
    result = validate_market_bar(payload)
    assert result.is_valid is True
    bar = result.record
    assert bar.ticker == "NVDA"
    assert bar.close == 123.5
    assert bar.abnormal_volume == 2.1


def test_market_bar_missing_numeric_fields():
    """Verify market bar rejects missing or malformed numeric OHLCV values."""
    payload = {
        "ticker": "AAPL",
        "timestamp": "2026-10-09T14:30:00Z",
        "open": "corrupted_price",
        "high": 150.0,
        "low": 140.0,
        "close": 145.0,
        "volume": 10000,
    }
    result = validate_market_bar(payload)
    assert result.is_valid is False
    assert "numerical" in result.errors[0].lower()
