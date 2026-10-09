"""Rumour-to-news lifecycle stage classifier.

Maps equity intelligence states into five point-in-time lifecycle stages:
Stage 1: Early Whisper
Stage 2: Hype Build-up
Stage 3: Peak Attention
Stage 4: Official Confirmation
Stage 5: Market Reaction / Reversal

Enforces strict timestamp cutoff to prevent look-ahead data leakage.
Official confirmation is never inferred from social media popularity alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from hypelens.analytics.attention_engine import (
    AttentionMetrics,
    calculate_attention_metrics,
)
from hypelens.models.data_models import (
    AccountTier,
    FinancialNewsItem,
    MarketBar,
    SocialPost,
)
from hypelens.nlp.parser import EpistemicStatus, ExtractedClaim


class LifecycleStage(str, Enum):
    """The five distinct phases of the rumour-to-news lifecycle."""
    STAGE_1_EARLY_WHISPER = "early_whisper"
    STAGE_2_HYPE_BUILDUP = "hype_buildup"
    STAGE_3_PEAK_ATTENTION = "peak_attention"
    STAGE_4_OFFICIAL_CONFIRMATION = "official_confirmation"
    STAGE_5_MARKET_REACTION = "market_reaction"


@dataclass
class LifecycleClassification:
    """Point-in-time lifecycle classification result."""
    ticker: str
    cutoff_time: datetime
    stage: LifecycleStage
    stage_index: int
    confidence: float
    has_official_confirmation: bool
    attention_multiplier: float
    hours_since_first_rumour: Optional[float] = None
    hours_since_official_news: Optional[float] = None
    social_lead_hours: Optional[float] = None
    rationale: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "cutoff_time": self.cutoff_time.isoformat(),
            "stage": self.stage.value,
            "stage_index": self.stage_index,
            "confidence": round(self.confidence, 3),
            "has_official_confirmation": self.has_official_confirmation,
            "attention_multiplier": round(self.attention_multiplier, 2),
            "hours_since_first_rumour": (
                round(self.hours_since_first_rumour, 2)
                if self.hours_since_first_rumour is not None
                else None
            ),
            "hours_since_official_news": (
                round(self.hours_since_official_news, 2)
                if self.hours_since_official_news is not None
                else None
            ),
            "social_lead_hours": (
                round(self.social_lead_hours, 2)
                if self.social_lead_hours is not None
                else None
            ),
            "rationale": self.rationale,
        }


def classify_lifecycle_stage(
    ticker: str,
    cutoff_time: datetime,
    social_posts: List[SocialPost],
    news_items: List[FinancialNewsItem],
    attention_metrics: Optional[AttentionMetrics] = None,
    extracted_claims: Optional[List[ExtractedClaim]] = None,
    market_bars: Optional[List[MarketBar]] = None,
    news_lookback_hours: float = 72.0,
) -> LifecycleClassification:
    """Classify the rumour-to-news lifecycle stage for a ticker strictly at cutoff_time.

    All inputs are strictly filtered to timestamps <= cutoff_time to avoid leakage.
    """
    clean_ticker = ticker.upper().replace("$", "").strip()

    # Enforce UTC timezone
    if cutoff_time.tzinfo is None:
        cutoff_time = cutoff_time.replace(tzinfo=timezone.utc)

    # 1. Strict point-in-time filtering
    point_posts = [
        p for p in social_posts
        if clean_ticker in p.tickers and p.timestamp is not None and p.timestamp <= cutoff_time
    ]
    point_news = [
        n for n in news_items
        if clean_ticker in n.tickers and n.timestamp is not None and n.timestamp <= cutoff_time
    ]

    # Calculate attention if not provided
    if attention_metrics is None:
        attention_metrics = calculate_attention_metrics(
            posts=point_posts,
            ticker=clean_ticker,
            cutoff_time=cutoff_time,
        )

    att_mult = attention_metrics.attention_multiplier
    rationale: List[str] = []

    # 2. Check for official news confirmation
    # Official confirmation NEVER comes from social volume alone; requires verified news
    official_news = [
        n for n in point_news
        if (cutoff_time - n.timestamp) <= timedelta(hours=news_lookback_hours)
    ]

    # Check claims for explicit official status
    claims_official = (
        any(c.status == EpistemicStatus.OFFICIALLY_CONFIRMED for c in (extracted_claims or []))
    )
    has_official = bool(official_news or claims_official)

    # Compute timeline offsets
    hours_first_rumour: Optional[float] = None
    if point_posts:
        first_post_ts = min(p.timestamp for p in point_posts)
        hours_first_rumour = (cutoff_time - first_post_ts).total_seconds() / 3600.0

    hours_official_news: Optional[float] = None
    if point_news:
        latest_news_ts = max(n.timestamp for n in point_news)
        hours_official_news = (cutoff_time - latest_news_ts).total_seconds() / 3600.0

    # Cross-stream lead–lag
    social_lead: Optional[float] = None
    if point_posts and point_news:
        first_post_ts = min(p.timestamp for p in point_posts)
        first_news_ts = min(n.timestamp for n in point_news)
        social_lead = (first_news_ts - first_post_ts).total_seconds() / 3600.0

    # 3. Decision Logic
    if has_official and hours_official_news is not None:
        if hours_official_news <= 24.0:
            # Stage 4: Official announcement released recently (<= 24h)
            stage = LifecycleStage.STAGE_4_OFFICIAL_CONFIRMATION
            stage_idx = 4
            confidence = 0.85
            rationale.append(f"Official confirmation detected {hours_official_news:.1f}h ago.")
        else:
            # Stage 5: Post-news market digestion and price reaction window (> 24h)
            stage = LifecycleStage.STAGE_5_MARKET_REACTION
            stage_idx = 5
            confidence = 0.80
            rationale.append(f"Official news released {hours_official_news:.1f}h ago; in reaction window.")
    else:
        # Pre-confirmation phases (Stages 1 - 3)
        has_large_creators = any(
            p.author.tier in (AccountTier.LARGE, AccountTier.MEDIUM)
            for p in point_posts
        )
        if att_mult >= 3.0 and has_large_creators:
            # Stage 3: Extreme attention, widespread social buzz before official confirmation
            stage = LifecycleStage.STAGE_3_PEAK_ATTENTION
            stage_idx = 3
            confidence = 0.75
            rationale.append(f"Peak attention detected: {att_mult:.1f}x baseline with large creators.")
        elif (att_mult >= 1.8 and len(point_posts) >= 2 and not attention_metrics.insufficient_history) or len(point_posts) >= 3:
            # Stage 2: Attention building, accelerating discussion, unverified speculation
            stage = LifecycleStage.STAGE_2_HYPE_BUILDUP
            stage_idx = 2
            confidence = 0.70
            rationale.append(f"Hype build-up: {att_mult:.1f}x baseline attention; rumor gaining spread.")
        else:
            # Stage 1: Whispers, early unverified mentions, small creators
            stage = LifecycleStage.STAGE_1_EARLY_WHISPER
            stage_idx = 1
            confidence = 0.65
            rationale.append("Early whisper: low volume of unconfirmed rumors; no broad attention yet.")

    return LifecycleClassification(
        ticker=clean_ticker,
        cutoff_time=cutoff_time,
        stage=stage,
        stage_index=stage_idx,
        confidence=confidence,
        has_official_confirmation=has_official,
        attention_multiplier=att_mult,
        hours_since_first_rumour=hours_first_rumour,
        hours_since_official_news=hours_official_news,
        social_lead_hours=social_lead,
        rationale=rationale,
    )
