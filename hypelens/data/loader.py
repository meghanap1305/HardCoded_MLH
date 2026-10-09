"""Dataset loader for HypeLens multimodal fixtures.

Parses JSONL social media records, financial news feeds, and CSV market prices,
strictly separating valid records from quarantined records.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hypelens.config import FIXTURES_DIR
from hypelens.models.data_models import (
    FinancialNewsItem,
    MarketBar,
    SocialPost,
    validate_financial_news,
    validate_market_bar,
    validate_social_post,
)


@dataclass
class DatasetBundle:
    """Container holding parsed multi-stream data and quarantined items."""
    social_posts: List[SocialPost] = field(default_factory=list)
    news_items: List[FinancialNewsItem] = field(default_factory=list)
    market_bars: List[MarketBar] = field(default_factory=list)
    quarantined: List[Dict[str, Any]] = field(default_factory=list)

    def get_posts_for_ticker(self, ticker: str) -> List[SocialPost]:
        """Filter social posts mentioning a specific ticker (case-insensitive)."""
        clean_t = ticker.upper().replace("$", "")
        return [p for p in self.social_posts if clean_t in p.tickers]

    def get_news_for_ticker(self, ticker: str) -> List[FinancialNewsItem]:
        """Filter news items mentioning a specific ticker."""
        clean_t = ticker.upper().replace("$", "")
        return [n for n in self.news_items if clean_t in n.tickers]

    def get_market_for_ticker(self, ticker: str) -> List[MarketBar]:
        """Retrieve chronological market bars for a specific ticker."""
        clean_t = ticker.upper().replace("$", "")
        bars = [b for b in self.market_bars if b.ticker == clean_t]
        return sorted(bars, key=lambda b: b.timestamp)


def load_social_posts(
    path: Optional[str | Path] = None,
) -> Tuple[List[SocialPost], List[Dict[str, Any]]]:
    """Load social posts from a JSONL file.

    Returns:
        Tuple of (valid_posts, quarantined_records).
    """
    target = Path(path) if path else FIXTURES_DIR / "sample_social.jsonl"
    valid: List[SocialPost] = []
    quarantined: List[Dict[str, Any]] = []

    if not target.exists():
        return valid, quarantined

    with open(target, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                raw = json.loads(line_str)
            except json.JSONDecodeError as e:
                quarantined.append({"line_no": line_no, "error": f"JSON decode error: {e}", "raw": line_str})
                continue

            result = validate_social_post(raw)
            if result.is_valid and result.record is not None:
                valid.append(result.record)
            else:
                quarantined.append({"line_no": line_no, "errors": result.errors, "raw": raw})

    # Sort valid posts chronologically
    valid.sort(key=lambda p: p.timestamp)
    return valid, quarantined


def load_financial_news(
    path: Optional[str | Path] = None,
) -> Tuple[List[FinancialNewsItem], List[Dict[str, Any]]]:
    """Load financial news items from a JSONL file.

    Returns:
        Tuple of (valid_news, quarantined_records).
    """
    target = Path(path) if path else FIXTURES_DIR / "sample_news.jsonl"
    valid: List[FinancialNewsItem] = []
    quarantined: List[Dict[str, Any]] = []

    if not target.exists():
        return valid, quarantined

    with open(target, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                raw = json.loads(line_str)
            except json.JSONDecodeError as e:
                quarantined.append({"line_no": line_no, "error": f"JSON decode error: {e}", "raw": line_str})
                continue

            result = validate_financial_news(raw)
            if result.is_valid and result.record is not None:
                valid.append(result.record)
            else:
                quarantined.append({"line_no": line_no, "errors": result.errors, "raw": raw})

    valid.sort(key=lambda n: n.timestamp)
    return valid, quarantined


def load_market_data(
    path: Optional[str | Path] = None,
) -> Tuple[List[MarketBar], List[Dict[str, Any]]]:
    """Load historical market bars from a CSV file.

    Returns:
        Tuple of (valid_bars, quarantined_records).
    """
    target = Path(path) if path else FIXTURES_DIR / "sample_market.csv"
    valid: List[MarketBar] = []
    quarantined: List[Dict[str, Any]] = []

    if not target.exists():
        return valid, quarantined

    with open(target, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for line_no, row in enumerate(reader, start=2):
            result = validate_market_bar(row)
            if result.is_valid and result.record is not None:
                valid.append(result.record)
            else:
                quarantined.append({"line_no": line_no, "errors": result.errors, "raw": row})

    valid.sort(key=lambda b: (b.ticker, b.timestamp))
    return valid, quarantined


def load_benchmark_dataset(
    fixtures_dir: Optional[str | Path] = None,
) -> DatasetBundle:
    """Load full benchmark dataset bundle from fixtures directory."""
    dir_path = Path(fixtures_dir) if fixtures_dir else FIXTURES_DIR

    posts, q_posts = load_social_posts(dir_path / "sample_social.jsonl")
    news, q_news = load_financial_news(dir_path / "sample_news.jsonl")
    market, q_market = load_market_data(dir_path / "sample_market.csv")

    all_quarantined = [
        {"type": "social", **q} for q in q_posts
    ] + [
        {"type": "news", **q} for q in q_news
    ] + [
        {"type": "market", **q} for q in q_market
    ]

    return DatasetBundle(
        social_posts=posts,
        news_items=news,
        market_bars=market,
        quarantined=all_quarantined,
    )
