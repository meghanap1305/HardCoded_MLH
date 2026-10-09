"""Unit tests for rumour-to-news lifecycle classifier and leakage guardrails (TASK-07)."""

from datetime import datetime, timedelta, timezone

from hypelens.analytics.attention_engine import AttentionMetrics
from hypelens.analytics.lifecycle_classifier import (
    LifecycleStage,
    classify_lifecycle_stage,
)
from hypelens.models.data_models import (
    AccountTier,
    AuthorMetadata,
    FinancialNewsItem,
    PlatformType,
    SocialPost,
)


def _make_post(post_id: str, ticker: str, ts: datetime, text: str, tier: AccountTier = AccountTier.SMALL) -> SocialPost:
    return SocialPost(
        post_id=post_id,
        platform=PlatformType.X_POST,
        timestamp=ts,
        author=AuthorMetadata(author_id=f"u_{post_id}", username=f"user_{post_id}", tier=tier),
        content_text=text,
        tickers=[ticker],
    )


def test_stage_1_early_whisper():
    """Verify single unverified rumour from small creator maps to Stage 1."""
    t0 = datetime(2026, 10, 5, 10, 0, tzinfo=timezone.utc)
    posts = [_make_post("p1", "NVDA", t0 - timedelta(hours=1), "Supply chain whisper", AccountTier.SMALL)]

    res = classify_lifecycle_stage("NVDA", cutoff_time=t0, social_posts=posts, news_items=[])
    assert res.stage == LifecycleStage.STAGE_1_EARLY_WHISPER
    assert res.stage_index == 1
    assert res.has_official_confirmation is False


def test_stage_2_hype_buildup():
    """Verify accelerating attention without official news maps to Stage 2."""
    t0 = datetime(2026, 10, 6, 15, 0, tzinfo=timezone.utc)
    posts = [
        _make_post("p1", "NVDA", t0 - timedelta(hours=20), "Early rumor"),
        _make_post("p2", "NVDA", t0 - timedelta(hours=10), "Rumor spreading"),
        _make_post("p3", "NVDA", t0 - timedelta(hours=2), "Big momentum building"),
    ]
    att = AttentionMetrics(
        ticker="NVDA", window_end=t0, current_mentions=3, baseline_mentions=1,
        normalized_baseline=1.0, attention_multiplier=2.0, unique_authors=3,
        author_diversity_ratio=1.0, duplicate_text_ratio=0.0, is_coordinated_pump=False,
        insufficient_history=False,
    )

    res = classify_lifecycle_stage("NVDA", cutoff_time=t0, social_posts=posts, news_items=[], attention_metrics=att)
    assert res.stage == LifecycleStage.STAGE_2_HYPE_BUILDUP
    assert res.stage_index == 2
    assert res.has_official_confirmation is False


def test_stage_3_peak_attention_not_confirmed():
    """Verify high attention with large creators maps to Stage 3 and NEVER infers confirmation."""
    t0 = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)
    posts = [
        _make_post("p1", "NVDA", t0 - timedelta(hours=5), "Everyone talking about this!", AccountTier.LARGE),
        _make_post("p2", "NVDA", t0 - timedelta(hours=2), "Going viral on reels", AccountTier.MEDIUM),
    ]
    att = AttentionMetrics(
        ticker="NVDA", window_end=t0, current_mentions=15, baseline_mentions=3,
        normalized_baseline=3.0, attention_multiplier=5.0, unique_authors=12,
        author_diversity_ratio=0.8, duplicate_text_ratio=0.1, is_coordinated_pump=False,
        insufficient_history=False,
    )

    res = classify_lifecycle_stage("NVDA", cutoff_time=t0, social_posts=posts, news_items=[], attention_metrics=att)
    assert res.stage == LifecycleStage.STAGE_3_PEAK_ATTENTION
    assert res.stage_index == 3
    # Confirmation is NEVER inferred from viral social attention
    assert res.has_official_confirmation is False


def test_stage_4_official_confirmation():
    """Verify recent verified news announcement maps to Stage 4."""
    t0 = datetime(2026, 10, 8, 16, 0, tzinfo=timezone.utc)
    posts = [_make_post("p1", "NVDA", t0 - timedelta(hours=48), "Pre-news rumor")]
    news = [
        FinancialNewsItem(
            news_id="n1", headline="Nvidia Officially Confirms Deal", publisher="Reuters",
            timestamp=t0 - timedelta(hours=2), body="Full press release confirmed.", tickers=["NVDA"],
        )
    ]

    res = classify_lifecycle_stage("NVDA", cutoff_time=t0, social_posts=posts, news_items=news)
    assert res.stage == LifecycleStage.STAGE_4_OFFICIAL_CONFIRMATION
    assert res.stage_index == 4
    assert res.has_official_confirmation is True
    assert res.hours_since_official_news is not None
    assert round(res.hours_since_official_news, 1) == 2.0


def test_stage_5_market_reaction():
    """Verify news older than 24h maps to Stage 5 market reaction window."""
    t0 = datetime(2026, 10, 9, 18, 0, tzinfo=timezone.utc)
    posts = [_make_post("p1", "NVDA", t0 - timedelta(hours=50), "Pre-news rumor")]
    news = [
        FinancialNewsItem(
            news_id="n1", headline="Nvidia Confirms Deal", publisher="Reuters",
            timestamp=t0 - timedelta(hours=30), body="Confirmed yesterday.", tickers=["NVDA"],
        )
    ]

    res = classify_lifecycle_stage("NVDA", cutoff_time=t0, social_posts=posts, news_items=news)
    assert res.stage == LifecycleStage.STAGE_5_MARKET_REACTION
    assert res.stage_index == 5
    assert res.has_official_confirmation is True


def test_leakage_cutoff_guardrail():
    """Ensure future news is strictly excluded when cutoff is set prior to release."""
    t_whisper = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)
    t_news = datetime(2026, 10, 8, 14, 0, tzinfo=timezone.utc)

    posts = [_make_post("p1", "NVDA", t_whisper - timedelta(hours=2), "Early whisper")]
    # News happens 2 days later in the future
    future_news = [
        FinancialNewsItem(
            news_id="n1", headline="Nvidia Confirms Deal", publisher="Reuters",
            timestamp=t_news, body="Confirmed.", tickers=["NVDA"],
        )
    ]

    # Evaluated at t_whisper: future news must NOT be visible
    res = classify_lifecycle_stage("NVDA", cutoff_time=t_whisper, social_posts=posts, news_items=future_news)
    assert res.stage == LifecycleStage.STAGE_1_EARLY_WHISPER
    assert res.has_official_confirmation is False
    assert res.hours_since_official_news is None


def test_social_lead_lag_calculation():
    """Verify calculation of social rumour lead time ahead of official announcement."""
    t_whisper = datetime(2026, 10, 5, 10, 0, tzinfo=timezone.utc)
    t_news = datetime(2026, 10, 8, 10, 0, tzinfo=timezone.utc)
    t_eval = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)

    posts = [_make_post("p1", "TSLA", t_whisper, "First whisper")]
    news = [
        FinancialNewsItem(
            news_id="n1", headline="TSLA Regulatory Permit Review", publisher="Bloomberg",
            timestamp=t_news, body="Notice published.", tickers=["TSLA"],
        )
    ]

    res = classify_lifecycle_stage("TSLA", cutoff_time=t_eval, social_posts=posts, news_items=news)
    # 72 hours difference between Oct 5 10:00 and Oct 8 10:00
    assert res.social_lead_hours is not None
    assert round(res.social_lead_hours, 1) == 72.0
