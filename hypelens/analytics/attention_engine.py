"""Attention engine and abnormal hype multiplier calculator.

Computes windowed mention volume against rolling historical baselines,
author diversity ratios, and coordinated pump manipulation signatures.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from hypelens.models.data_models import SocialPost


@dataclass
class AttentionMetrics:
    """Calculated attention, baseline multiplier, and manipulation metrics."""
    ticker: str
    window_end: datetime
    current_mentions: int
    baseline_mentions: int
    normalized_baseline: float
    attention_multiplier: float
    unique_authors: int
    author_diversity_ratio: float
    duplicate_text_ratio: float
    is_coordinated_pump: bool
    insufficient_history: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "window_end": self.window_end.isoformat(),
            "current_mentions": self.current_mentions,
            "baseline_mentions": self.baseline_mentions,
            "normalized_baseline": round(self.normalized_baseline, 2),
            "attention_multiplier": round(self.attention_multiplier, 2),
            "unique_authors": self.unique_authors,
            "author_diversity_ratio": round(self.author_diversity_ratio, 3),
            "duplicate_text_ratio": round(self.duplicate_text_ratio, 3),
            "is_coordinated_pump": self.is_coordinated_pump,
            "insufficient_history": self.insufficient_history,
        }


def _normalize_text_for_hash(text: str) -> str:
    """Normalize text removing case, punctuation, and excess whitespace."""
    cleaned = re.sub(r"[^\w\s]", "", text.lower())
    return " ".join(cleaned.split())


def calculate_attention_metrics(
    posts: List[SocialPost],
    ticker: str,
    cutoff_time: Optional[datetime] = None,
    current_window_hours: float = 24.0,
    baseline_window_hours: float = 168.0,  # 7 days baseline
) -> AttentionMetrics:
    """Calculate rolling abnormal attention and manipulation metrics for a ticker.

    Args:
        posts: List of validated SocialPost items.
        ticker: Target stock ticker.
        cutoff_time: End of analysis window (defaults to latest post timestamp or UTC now).
        current_window_hours: Duration of current active hype window in hours.
        baseline_window_hours: Duration of prior historical baseline window in hours.

    Returns:
        AttentionMetrics object with calculated multipliers and flags.
    """
    clean_ticker = ticker.upper().replace("$", "").strip()

    # Filter posts mentioning ticker with valid timestamps
    ticker_posts = [
        p for p in posts
        if clean_ticker in p.tickers and p.timestamp is not None
    ]

    # Establish cutoff time
    if cutoff_time is None:
        if ticker_posts:
            cutoff_time = max(p.timestamp for p in ticker_posts)
        else:
            cutoff_time = datetime.now(timezone.utc)
    elif cutoff_time.tzinfo is None:
        cutoff_time = cutoff_time.replace(tzinfo=timezone.utc)

    curr_start = cutoff_time - timedelta(hours=current_window_hours)
    base_start = curr_start - timedelta(hours=baseline_window_hours)

    current_posts = [p for p in ticker_posts if curr_start < p.timestamp <= cutoff_time]
    baseline_posts = [p for p in ticker_posts if base_start < p.timestamp <= curr_start]

    curr_count = len(current_posts)
    base_count = len(baseline_posts)

    # Normalize baseline to match current window length
    norm_ratio = (current_window_hours / baseline_window_hours) if baseline_window_hours > 0 else 1.0
    norm_baseline = base_count * norm_ratio

    # Calculate attention multiplier safely
    if norm_baseline > 0:
        multiplier = curr_count / norm_baseline
        insufficient_history = False
    else:
        # Zero baseline handling: if current mentions exist, flag insufficient history
        insufficient_history = True
        if curr_count > 0:
            multiplier = 1.0 + float(curr_count)
        else:
            multiplier = 1.0

    # Calculate author diversity and duplication ratios
    if curr_count > 0:
        unique_authors = len({p.author.author_id for p in current_posts})
        diversity_ratio = unique_authors / curr_count

        # Duplicate text analysis
        text_hashes = [_normalize_text_for_hash(p.content_text) for p in current_posts]
        duplicate_count = curr_count - len(set(text_hashes))
        dup_ratio = duplicate_count / curr_count

        # Pump detection: low author diversity (< 0.5) OR high duplicate ratio (>= 0.4)
        is_pump = (curr_count >= 2) and (diversity_ratio < 0.6 or dup_ratio >= 0.4)
    else:
        unique_authors = 0
        diversity_ratio = 1.0
        dup_ratio = 0.0
        is_pump = False

    return AttentionMetrics(
        ticker=clean_ticker,
        window_end=cutoff_time,
        current_mentions=curr_count,
        baseline_mentions=base_count,
        normalized_baseline=norm_baseline,
        attention_multiplier=round(multiplier, 3),
        unique_authors=unique_authors,
        author_diversity_ratio=round(diversity_ratio, 3),
        duplicate_text_ratio=round(dup_ratio, 3),
        is_coordinated_pump=is_pump,
        insufficient_history=insufficient_history,
    )
