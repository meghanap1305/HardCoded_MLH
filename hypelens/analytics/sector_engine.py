"""Sector taxonomy and lifecycle categorization engine for HypeLens.

Groups equities by industrial/thematic sector and categorizes each stock's
current position in the rumour-to-news lifecycle (Early Rumour, Hype Build-up,
Peak Attention, Official Confirmation, Market Reaction).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from hypelens.analytics.attention_engine import calculate_attention_metrics
from hypelens.analytics.lifecycle_classifier import LifecycleStage, classify_lifecycle_stage
from hypelens.config import get_config
from hypelens.data.loader import DatasetBundle


# Default Sector Taxonomy mapping
STOCK_SECTORS: Dict[str, str] = {
    "NVDA": "Semiconductors & AI",
    "AMD": "Semiconductors & AI",
    "TSLA": "EV & Automotive",
    "RIVN": "EV & Automotive",
    "AAPL": "Consumer Tech & Hardware",
    "MSFT": "Cloud & Enterprise Software",
    "GOOGL": "Cloud & Enterprise Software",
}

SECTOR_DESCRIPTIONS: Dict[str, str] = {
    "Semiconductors & AI": "Hardware accelerators, hyperscaler chip deals, and supply chain whisper chatter.",
    "EV & Automotive": "Autonomous permits, battery breakthroughs, and retail momentum speculation.",
    "Consumer Tech & Hardware": "Product launches, keynote leaks, and subscription tier rumors.",
    "Cloud & Enterprise Software": "Enterprise AI contracts, M&A acquisitions, and cloud infrastructure.",
}


def get_sector_for_ticker(ticker: str) -> str:
    """Retrieve thematic sector for a stock ticker."""
    clean = ticker.upper().replace("$", "").strip()
    try:
        from hypelens.models.taxonomy import get_taxonomy
        resolved_sector = get_taxonomy().get_sector(clean)
        if resolved_sector and resolved_sector != "Diversified Tech / Other":
            return resolved_sector
    except Exception:
        pass
    return STOCK_SECTORS.get(clean, "Diversified Tech / Other")


def categorize_stocks_by_sector(
    bundle: DatasetBundle,
    cutoff_time: Optional[datetime] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Categorize all stocks by sector and evaluate their point-in-time lifecycle stage.

    Returns:
        Dictionary mapping sector name to list of stock lifecycle analysis dicts.
    """
    tickers = sorted(list({b.ticker for b in bundle.market_bars} | {
        t for p in bundle.social_posts for t in p.tickers
    }))

    sector_groups: Dict[str, List[Dict[str, Any]]] = {}

    for ticker in tickers:
        sector = get_sector_for_ticker(ticker)
        if sector not in sector_groups:
            sector_groups[sector] = []

        posts = bundle.get_posts_for_ticker(ticker)
        bars = bundle.get_market_for_ticker(ticker)
        news = bundle.get_news_for_ticker(ticker)

        eval_ts = cutoff_time
        if eval_ts is None:
            eval_ts = bars[-1].timestamp if bars else (posts[-1].timestamp if posts else datetime.now(timezone.utc))

        att = calculate_attention_metrics(posts, ticker, cutoff_time=eval_ts)
        lifecycle = classify_lifecycle_stage(ticker, eval_ts, posts, news, attention_metrics=att)

        # Classify epistemic state (Rumour vs Confirmed)
        if lifecycle.has_official_confirmation:
            epistemic_type = "Officially Confirmed PR"
            epistemic_badge = "CONFIRMED NEWS"
        else:
            epistemic_type = "Unverified Social Rumour"
            epistemic_badge = "UNVERIFIED RUMOUR"

        # Signal summary
        if att.is_coordinated_pump:
            signal = "Avoid // Pump Manipulation"
        elif lifecycle.stage == LifecycleStage.STAGE_1_EARLY_WHISPER:
            signal = "Early Whisper (Stage 1)"
        elif lifecycle.stage == LifecycleStage.STAGE_2_HYPE_BUILDUP:
            signal = "Pre-News Run-Up Candidate (Stage 2)"
        elif lifecycle.stage == LifecycleStage.STAGE_3_PEAK_ATTENTION:
            signal = "Crowded Peak FOMO (Stage 3)"
        elif lifecycle.stage == LifecycleStage.STAGE_4_OFFICIAL_CONFIRMATION:
            signal = "Official Confirmation Released (Stage 4)"
        else:
            signal = "Sell-the-News Reversal Watch (Stage 5)"

        latest_price = bars[-1].close if bars else None
        ret_1d = None
        if len(bars) >= 2 and bars[-2].close > 0:
            ret_1d = round((bars[-1].close - bars[-2].close) / bars[-2].close * 100.0, 2)

        stock_info = {
            "ticker": ticker,
            "sector": sector,
            "stage_name": lifecycle.stage.value.replace("_", " ").title(),
            "stage_index": lifecycle.stage_index,
            "stage_key": lifecycle.stage.value,
            "epistemic_type": epistemic_type,
            "epistemic_badge": epistemic_badge,
            "has_official_confirmation": lifecycle.has_official_confirmation,
            "attention_multiplier": round(att.attention_multiplier, 1),
            "current_mentions": att.current_mentions,
            "is_pump_flag": att.is_coordinated_pump,
            "social_lead_hours": round(lifecycle.social_lead_hours, 1) if lifecycle.social_lead_hours else None,
            "opportunity_signal": signal,
            "latest_price": latest_price,
            "return_1d": ret_1d,
            "rationale": lifecycle.rationale[0] if lifecycle.rationale else "Normal trading baseline",
        }
        sector_groups[sector].append(stock_info)

    return sector_groups


def build_sector_lifecycle_table(bundle: DatasetBundle) -> pd.DataFrame:
    """Build a consolidated tabular DataFrame of all stocks with sector and lifecycle stages."""
    grouped = categorize_stocks_by_sector(bundle)
    rows: List[Dict[str, Any]] = []

    for sector, stocks in grouped.items():
        for s in stocks:
            rows.append({
                "Sector": sector,
                "Ticker": s["ticker"],
                "Lifecycle Stage": f"Stage {s['stage_index']}: {s['stage_name']}",
                "Information Type": s["epistemic_badge"],
                "Attention Spike": f"{s['attention_multiplier']}x",
                "Mentions": s["current_mentions"],
                "Social Lead": f"{s['social_lead_hours']}h" if s["social_lead_hours"] is not None else "Pending News",
                "Opportunity Signal": s["opportunity_signal"],
                "Latest Price": f"${s['latest_price']:.2f}" if s["latest_price"] is not None else "N/A",
                "1D Return": f"{s['return_1d']:+.2f}%" if s["return_1d"] is not None else "N/A",
            })

    return pd.DataFrame(rows)


def aggregate_live_items_by_sector(
    items: List[Any],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate live news or exchange items by industrial sector using taxonomy.

    Keeps unmatched items in a 'Market-wide / Unclassified' category.
    Does not fabricate prices, sentiment, or events.
    """
    from hypelens.models.taxonomy import get_taxonomy
    tax = get_taxonomy()

    sector_agg: Dict[str, Dict[str, Any]] = {}

    for item in items:
        tickers = getattr(item, "tickers", [])
        matched_sectors = set()

        for t in tickers:
            sec = tax.get_sector(t)
            matched_sectors.add(sec)

        if not matched_sectors:
            matched_sectors = {"Market-wide / Unclassified"}

        for sec in matched_sectors:
            if sec not in sector_agg:
                sector_agg[sec] = {
                    "sector": sec,
                    "event_count": 0,
                    "latest_event_ts": None,
                    "items": [],
                    "companies": set(),
                }
            sector_agg[sec]["event_count"] += 1
            sector_agg[sec]["items"].append(item)
            for t in tickers:
                sector_agg[sec]["companies"].add(t)

            ts = getattr(item, "published_at", None) or getattr(item, "timestamp", None)
            if ts:
                current_latest = sector_agg[sec]["latest_event_ts"]
                if current_latest is None or ts > current_latest:
                    sector_agg[sec]["latest_event_ts"] = ts

    # Convert companies set to sorted list
    for sec_data in sector_agg.values():
        sec_data["companies"] = sorted(list(sec_data["companies"]))

    return sector_agg

