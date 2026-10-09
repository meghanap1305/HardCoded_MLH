"""Unit tests for sector grouping and lifecycle categorization engine."""

import pandas as pd
import pytest

from hypelens.analytics.sector_engine import (
    build_sector_lifecycle_table,
    categorize_stocks_by_sector,
    get_sector_for_ticker,
)
from hypelens.data.loader import load_benchmark_dataset


def test_get_sector_for_ticker():
    """Verify sector resolution for core tickers."""
    assert get_sector_for_ticker("NVDA") == "Semiconductors & AI"
    assert get_sector_for_ticker("AMD") == "Semiconductors & AI"
    assert get_sector_for_ticker("TSLA") == "EV & Automotive"
    assert get_sector_for_ticker("AAPL") == "Consumer Tech & Hardware"
    assert get_sector_for_ticker("UNKNOWN") == "Diversified Tech / Other"


def test_categorize_stocks_by_sector():
    """Verify stocks grouped into sectors with accurate lifecycle stages."""
    bundle = load_benchmark_dataset()
    grouped = categorize_stocks_by_sector(bundle)

    assert "Semiconductors & AI" in grouped
    assert "EV & Automotive" in grouped
    assert "Consumer Tech & Hardware" in grouped

    # Check NVDA within Semiconductors
    nvda_items = [s for s in grouped["Semiconductors & AI"] if s["ticker"] == "NVDA"]
    assert len(nvda_items) == 1
    nvda = nvda_items[0]
    assert nvda["stage_index"] in [1, 2, 3, 4, 5]
    assert "epistemic_badge" in nvda
    assert "attention_multiplier" in nvda

    # Check AAPL within Consumer Tech
    aapl_items = [s for s in grouped["Consumer Tech & Hardware"] if s["ticker"] == "AAPL"]
    assert len(aapl_items) == 1


def test_build_sector_lifecycle_table():
    """Verify tabular formatting of sector and lifecycle matrix."""
    bundle = load_benchmark_dataset()
    df = build_sector_lifecycle_table(bundle)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    expected_cols = ["Sector", "Ticker", "Lifecycle Stage", "Information Type", "Attention Spike", "Opportunity Signal"]
    for col in expected_cols:
        assert col in df.columns
