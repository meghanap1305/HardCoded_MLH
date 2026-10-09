"""Unit tests for Streamlit UI formatting components and app helpers (TASK-09)."""

import pandas as pd
import pytest

from hypelens.data.loader import load_benchmark_dataset
from hypelens.ml.movement_model import MovementPrediction
from hypelens.nlp.parser import EpistemicStatus, EventCategory, ExtractedClaim
from hypelens.ui.components import (
    RESEARCH_DISCLAIMER,
    format_multimodal_card,
    format_overview_metrics,
    format_signal_card,
    format_timeline_events,
)


def test_research_disclaimer_content():
    """Verify presence and strict disclosures in research disclaimer."""
    assert "RESEARCH DECISION-SUPPORT" in RESEARCH_DISCLAIMER
    assert "NOT FINANCIAL ADVICE" in RESEARCH_DISCLAIMER
    assert "does not guarantee" in RESEARCH_DISCLAIMER


def test_format_overview_metrics_bundle():
    """Verify overview table formatting across all tickers."""
    bundle = load_benchmark_dataset()
    df = format_overview_metrics(bundle)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    expected_cols = [
        "Ticker", "Mentions", "Attention Multiplier",
        "Influencers (S/M/L)", "Lifecycle Stage",
        "Latest Price", "1D Return", "Manipulation Alert",
    ]
    for col in expected_cols:
        assert col in df.columns

    tickers = df["Ticker"].tolist()
    assert "NVDA" in tickers
    assert "TSLA" in tickers
    assert "AAPL" in tickers


def test_format_signal_card_styling():
    """Verify signal badge styling and probability conversion."""
    pred = MovementPrediction(
        ticker="NVDA",
        probability_outperform=0.68,
        signal_class="Pre-News Run-Up Candidate",
        confidence=0.68,
        lifecycle_stage_index=2,
        features_used=["return_1d"],
        explanation=["Early hype buildup with positive whisper."],
    )
    card = format_signal_card(pred)
    assert card["ticker"] == "NVDA"
    assert card["signal_class"] == "Pre-News Run-Up Candidate"
    assert card["badge_color"] == "green"
    assert card["probability_outperform"] == 68.0
    assert len(card["explanation"]) == 1


def test_format_timeline_events_chronological():
    """Verify unified timeline interleaving social posts and news chronologically."""
    bundle = load_benchmark_dataset()
    nvda_posts = bundle.get_posts_for_ticker("NVDA")
    nvda_news = bundle.get_news_for_ticker("NVDA")

    events = format_timeline_events(nvda_posts, nvda_news)
    assert len(events) >= 4
    # Ensure chronological sort order
    times = [e["time_dt"] for e in events]
    assert times == sorted(times)
    assert any(e["stream"] == "Official / News Wire" for e in events)
    assert any(e["stream"] == "Community / Social" for e in events)


def test_format_multimodal_card_provenance():
    """Verify multimodal card formatting and clear mock banner."""
    bundle = load_benchmark_dataset()
    reels = [p for p in bundle.social_posts if p.platform.value == "instagram_reel"]
    assert len(reels) > 0
    reel = reels[0]

    claim = ExtractedClaim(
        ticker="NVDA",
        claim="Major cloud deal",
        event_category=EventCategory.MERGER_ACQUISITION,
        status=EpistemicStatus.RUMOUR_UNVERIFIED,
        sentiment=0.75,
        intensity=0.8,
    )
    card = format_multimodal_card(reel, claim=claim, is_mock=True)

    assert card["platform"] == "instagram_reel"
    assert card["ocr_text"] is not None
    assert card["transcript"] is not None
    assert card["provenance_banner"] == "[DETERMINISTIC MOCK EXTRACTOR]"
    assert card["sentiment"] == 0.75


def test_app_module_imports():
    """Verify that app.py imports cleanly without syntax errors."""
    import app
    assert hasattr(app, "main")
    assert hasattr(app, "get_cached_bundle")
