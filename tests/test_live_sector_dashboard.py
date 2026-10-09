"""Unit tests for Indian date parsing, sector aggregation, and no-company dashboard states."""

from datetime import datetime, timezone
import pytest

from hypelens.analytics.sector_engine import aggregate_live_items_by_sector
from hypelens.models.data_models import FinancialNewsItem
from hypelens.utils.datetime_utils import format_ist, to_ist, to_utc


def test_sebi_and_nse_date_parsing_offline():
    """Verify parsing SEBI and NSE custom date strings to strict UTC."""
    # SEBI format: '08 Oct, 2026 +0530'
    sebi_dt = to_utc("08 Oct, 2026 +0530")
    assert sebi_dt is not None
    assert sebi_dt.tzinfo == timezone.utc
    assert sebi_dt.year == 2026
    assert sebi_dt.month == 10
    assert sebi_dt.day == 7  # 8th Oct 00:00 +0530 is 7th Oct 18:30 UTC
    assert sebi_dt.hour == 18
    assert sebi_dt.minute == 30

    # SEBI format without offset: '08 Oct, 2026' (assumed IST)
    sebi_dt2 = to_utc("08 Oct, 2026")
    assert sebi_dt2 is not None
    assert sebi_dt2.tzinfo == timezone.utc
    assert sebi_dt2.hour == 18
    assert sebi_dt2.minute == 30

    # NSE format: '09-Oct-2026 15:32:52'
    nse_dt = to_utc("09-Oct-2026 15:32:52")
    assert nse_dt is not None
    assert nse_dt.tzinfo == timezone.utc
    assert nse_dt.year == 2026
    assert nse_dt.month == 10
    assert nse_dt.day == 9
    assert nse_dt.hour == 10  # 15:32 IST is 10:02 UTC
    assert nse_dt.minute == 2
    assert nse_dt.second == 52

    # Formatting in Asia/Kolkata
    ist_display = format_ist(nse_dt)
    assert "09 Oct 2026" in ist_display
    assert "03:32 PM IST" in ist_display


def test_aggregate_live_items_by_sector():
    """Verify aggregation of live items across sectors using taxonomy."""
    items = [
        FinancialNewsItem(
            news_id="rel_1",
            headline="Reliance Jio 5G expansion",
            publisher="NSE",
            timestamp=to_utc("09-Oct-2026 10:00:00"),
            tickers=["RELIANCE"],
            source_provider="official_nse",
        ),
        FinancialNewsItem(
            news_id="tamo_1",
            headline="Tata Motors commercial vehicles update",
            publisher="NSE",
            timestamp=to_utc("09-Oct-2026 11:00:00"),
            tickers=["TATAMOTORS"],
            source_provider="official_nse",
        ),
        FinancialNewsItem(
            news_id="sebi_1",
            headline="Adjudication order for broker",
            publisher="SEBI",
            timestamp=to_utc("08 Oct, 2026 +0530"),
            tickers=[],  # Unmatched general market item
            source_provider="official_sebi",
        ),
    ]

    agg = aggregate_live_items_by_sector(items)

    assert "Energy & Conglomerate" in agg
    assert agg["Energy & Conglomerate"]["event_count"] == 1
    assert "RELIANCE" in agg["Energy & Conglomerate"]["companies"]

    assert "EV & Automotive" in agg
    assert agg["EV & Automotive"]["event_count"] == 1
    assert "TATAMOTORS" in agg["EV & Automotive"]["companies"]

    # General unmatched item goes into Market-wide / Unclassified
    assert "Market-wide / Unclassified" in agg
    assert agg["Market-wide / Unclassified"]["event_count"] == 1
    assert len(agg["Market-wide / Unclassified"]["companies"]) == 0


def test_no_company_selected_overview_logic():
    """Verify sector aggregation handles empty items or all sectors without crash."""
    empty_agg = aggregate_live_items_by_sector([])
    assert empty_agg == {}

    # Unclassified only
    items = [
        FinancialNewsItem(
            news_id="gen_1",
            headline="General exchange circular",
            publisher="NSE",
            timestamp=to_utc("09-Oct-2026 12:00:00"),
            tickers=[],
            source_provider="official_nse",
        )
    ]
    agg = aggregate_live_items_by_sector(items)
    assert "Market-wide / Unclassified" in agg
    assert agg["Market-wide / Unclassified"]["event_count"] == 1
