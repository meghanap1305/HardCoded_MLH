"""Unit tests for benchmark dataset loading and quarantine separation (TASK-03)."""

from hypelens.data.loader import (
    DatasetBundle,
    load_benchmark_dataset,
    load_financial_news,
    load_market_data,
    load_social_posts,
)


def test_load_social_posts_fixtures():
    """Verify loading and segregation of valid vs quarantined social posts."""
    posts, quarantined = load_social_posts()
    assert len(posts) >= 8
    assert len(quarantined) >= 1

    # Check presence of Instagram Reel with video and OCR
    reels = [p for p in posts if p.platform.value == "instagram_reel"]
    assert len(reels) >= 1
    assert any(r.video_path is not None for r in reels)
    assert any(r.ocr_text is not None for r in reels)

    # Check presence of X posts
    x_posts = [p for p in posts if p.platform.value == "x_post"]
    assert len(x_posts) >= 1


def test_load_financial_news_fixtures():
    """Verify loading of financial news and quarantine handling."""
    news, quarantined = load_financial_news()
    assert len(news) >= 3
    assert len(quarantined) >= 1
    assert any(n.publisher == "Reuters" for n in news)


def test_load_market_data_fixtures():
    """Verify loading of market OHLCV CSV and quarantine handling."""
    bars, quarantined = load_market_data()
    assert len(bars) >= 10
    assert len(quarantined) >= 1

    nvda_bars = [b for b in bars if b.ticker == "NVDA"]
    assert len(nvda_bars) >= 4
    assert any(b.abnormal_volume > 2.0 for b in nvda_bars)


def test_load_benchmark_dataset_bundle():
    """Verify full bundle loading and ticker filtering methods."""
    bundle = load_benchmark_dataset()
    assert isinstance(bundle, DatasetBundle)
    assert len(bundle.social_posts) > 0
    assert len(bundle.news_items) > 0
    assert len(bundle.market_bars) > 0
    assert len(bundle.quarantined) >= 3

    nvda_posts = bundle.get_posts_for_ticker("NVDA")
    assert len(nvda_posts) >= 3

    tsla_news = bundle.get_news_for_ticker("TSLA")
    assert len(tsla_news) >= 1

    aapl_market = bundle.get_market_for_ticker("AAPL")
    assert len(aapl_market) >= 3
