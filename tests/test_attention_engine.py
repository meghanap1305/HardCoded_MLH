"""Unit tests for attention engine and abnormal baseline multiplier (TASK-06)."""

from datetime import datetime, timedelta, timezone

from hypelens.analytics.attention_engine import (
    AttentionMetrics,
    calculate_attention_metrics,
)
from hypelens.data.loader import load_social_posts
from hypelens.models.data_models import (
    AccountTier,
    AuthorMetadata,
    PlatformType,
    SocialPost,
)


def _make_post(post_id: str, ticker: str, ts: datetime, text: str, author_id: str) -> SocialPost:
    return SocialPost(
        post_id=post_id,
        platform=PlatformType.X_POST,
        timestamp=ts,
        author=AuthorMetadata(author_id=author_id, username=author_id, tier=AccountTier.SMALL),
        content_text=text,
        tickers=[ticker],
    )


def test_attention_metrics_normal_spike():
    """Verify multiplier calculation with known baseline and current activity."""
    now = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
    posts = []

    # 14 baseline posts evenly in the 7 days prior to last 24h
    # Baseline window = 168h; current window = 24h.
    # norm_baseline = 14 * (24 / 168) = 2.0 posts/day.
    for i in range(14):
        ts = now - timedelta(days=2 + (i % 5), hours=i)
        posts.append(_make_post(f"base_{i}", "NVDA", ts, f"Baseline chat {i}", f"user_{i}"))

    # 8 posts in the last 24 hours from distinct authors
    for i in range(8):
        ts = now - timedelta(hours=i * 2 + 1)
        posts.append(_make_post(f"curr_{i}", "NVDA", ts, f"Big news rumor {i}", f"trader_{i}"))

    metrics = calculate_attention_metrics(
        posts=posts,
        ticker="NVDA",
        cutoff_time=now,
        current_window_hours=24.0,
        baseline_window_hours=168.0,
    )

    assert isinstance(metrics, AttentionMetrics)
    assert metrics.current_mentions == 8
    assert metrics.baseline_mentions == 14
    assert metrics.normalized_baseline == 2.0
    assert metrics.attention_multiplier == 4.0  # 8 / 2.0 = 4.0x
    assert metrics.insufficient_history is False
    assert metrics.unique_authors == 8
    assert metrics.author_diversity_ratio == 1.0
    assert metrics.is_coordinated_pump is False


def test_attention_metrics_zero_baseline():
    """Ensure zero historical baseline does not cause division by zero."""
    now = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
    posts = [
        _make_post("c1", "TSLA", now - timedelta(hours=2), "Early whisper", "a1"),
        _make_post("c2", "TSLA", now - timedelta(hours=4), "Another post", "a2"),
    ]

    metrics = calculate_attention_metrics(posts, "TSLA", cutoff_time=now)
    assert metrics.current_mentions == 2
    assert metrics.baseline_mentions == 0
    assert metrics.insufficient_history is True
    assert metrics.attention_multiplier > 1.0  # Safe smoothed multiplier


def test_attention_metrics_zero_current():
    """Verify handling when there are zero current mentions."""
    now = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
    posts = [
        _make_post("b1", "AAPL", now - timedelta(days=3), "Old news", "a1"),
    ]

    metrics = calculate_attention_metrics(posts, "AAPL", cutoff_time=now)
    assert metrics.current_mentions == 0
    assert metrics.attention_multiplier == 0.0
    assert metrics.is_coordinated_pump is False


def test_attention_metrics_coordinated_pump_detection():
    """Verify flagging of repeated spam text and narrow author clusters."""
    now = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
    posts = [
        _make_post("p1", "TSLA", now - timedelta(hours=1), "$TSLA TO THE MOON! BUY NOW!", "bot_1"),
        _make_post("p2", "TSLA", now - timedelta(hours=2), "$TSLA TO THE MOON! BUY NOW!", "bot_2"),
        _make_post("p3", "TSLA", now - timedelta(hours=3), "$TSLA TO THE MOON! BUY NOW!", "bot_3"),
    ]

    metrics = calculate_attention_metrics(posts, "TSLA", cutoff_time=now)
    assert metrics.current_mentions == 3
    assert metrics.duplicate_text_ratio > 0.5
    assert metrics.is_coordinated_pump is True


def test_attention_metrics_with_fixture_data():
    """Verify integration of attention engine with sample_social fixture records."""
    posts, _ = load_social_posts()
    metrics = calculate_attention_metrics(posts, "NVDA")
    assert metrics.ticker == "NVDA"
    assert metrics.current_mentions >= 1
    assert metrics.attention_multiplier >= 1.0

    d = metrics.to_dict()
    assert "attention_multiplier" in d
    assert "is_coordinated_pump" in d
