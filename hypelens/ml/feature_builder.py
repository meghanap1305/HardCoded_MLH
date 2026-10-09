"""Leakage-safe feature builder and forward CAR target computation.

Enforces strict timestamp cutoff: every feature vector for prediction point t
uses ONLY information timestamped <= t. Future prices are used exclusively for
calculating the forward target CAR(t, t+H).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from hypelens.analytics.attention_engine import calculate_attention_metrics
from hypelens.analytics.lifecycle_classifier import classify_lifecycle_stage
from hypelens.data.loader import DatasetBundle
from hypelens.models.data_models import FinancialNewsItem, MarketBar, SocialPost
from hypelens.nlp.extractor import GemmaExtractor
from hypelens.nlp.parser import ExtractedClaim


FEATURE_SETS = {
    "market": ["return_1d", "abnormal_volume", "price_volatility"],
    "sentiment": ["avg_sentiment", "sentiment_intensity", "bullish_post_ratio"],
    "attention": ["attention_multiplier", "current_mentions", "author_diversity", "is_pump_flag"],
    "lifecycle": ["stage_index", "has_official_news", "hours_since_first_rumour"],
}


def compute_forward_car(
    bars: List[MarketBar],
    cutoff_time: datetime,
    horizon_steps: int = 1,
) -> Optional[float]:
    """Calculate forward Cumulative Abnormal Return CAR(t, t+H).

    CAR = sum_{tau=1}^{H} (R_{stock, tau} - R_{benchmark, tau})
    Returns None if fewer than horizon_steps forward observations exist.
    """
    sorted_bars = sorted(bars, key=lambda b: b.timestamp)
    # Find base bar at or immediately prior to cutoff_time
    prior_indices = [i for i, b in enumerate(sorted_bars) if b.timestamp <= cutoff_time]
    if not prior_indices:
        return None

    base_idx = prior_indices[-1]
    target_idx = base_idx + horizon_steps
    if target_idx >= len(sorted_bars):
        return None

    car = 0.0
    for tau in range(base_idx + 1, target_idx + 1):
        prev_p = sorted_bars[tau - 1].close
        curr_p = sorted_bars[tau].close
        if prev_p <= 0:
            continue
        stock_ret = (curr_p - prev_p) / prev_p
        bench_ret = sorted_bars[tau].benchmark_return
        car += (stock_ret - bench_ret)

    return float(car)


def build_feature_row(
    ticker: str,
    cutoff_time: datetime,
    social_posts: List[SocialPost],
    news_items: List[FinancialNewsItem],
    market_bars: List[MarketBar],
    claims: Optional[List[ExtractedClaim]] = None,
    forward_horizon: int = 1,
) -> Dict[str, Any]:
    """Build a point-in-time feature dictionary strictly using data available by cutoff_time."""
    clean_ticker = ticker.upper().replace("$", "").strip()
    if cutoff_time.tzinfo is None:
        cutoff_time = cutoff_time.replace(tzinfo=timezone.utc)

    # 1. Point-in-time filtering (zero look-ahead leakage)
    pt_posts = [p for p in social_posts if clean_ticker in p.tickers and p.timestamp <= cutoff_time]
    pt_news = [n for n in news_items if clean_ticker in n.tickers and n.timestamp <= cutoff_time]
    pt_bars = sorted(
        [b for b in market_bars if b.ticker == clean_ticker and b.timestamp <= cutoff_time],
        key=lambda b: b.timestamp,
    )

    # 2. Market features
    if len(pt_bars) >= 2:
        p0 = pt_bars[-2].close
        p1 = pt_bars[-1].close
        ret_1d = (p1 - p0) / p0 if p0 > 0 else 0.0
        abnormal_vol = pt_bars[-1].abnormal_volume
        highs = [b.high for b in pt_bars[-5:]]
        lows = [b.low for b in pt_bars[-5:]]
        volatility = (max(highs) - min(lows)) / pt_bars[-1].close if pt_bars[-1].close > 0 else 0.0
    elif pt_bars:
        ret_1d = 0.0
        abnormal_vol = pt_bars[-1].abnormal_volume
        volatility = 0.0
    else:
        ret_1d = 0.0
        abnormal_vol = 1.0
        volatility = 0.0

    # 3. Attention features
    att = calculate_attention_metrics(
        posts=pt_posts,
        ticker=clean_ticker,
        cutoff_time=cutoff_time,
    )

    # 4. Sentiment features
    # Derive sentiment from claims or deterministic mock fallback
    if claims:
        pt_claims = claims
    else:
        extractor = GemmaExtractor(force_mock=True)
        pt_claims = [extractor.extract_from_post(p).claim for p in pt_posts]

    if pt_claims:
        avg_sentiment = float(np.mean([c.sentiment for c in pt_claims]))
        avg_intensity = float(np.mean([c.intensity for c in pt_claims]))
        bullish_count = sum(1 for c in pt_claims if c.sentiment > 0.1)
        bullish_ratio = bullish_count / len(pt_claims)
    else:
        avg_sentiment = 0.0
        avg_intensity = 0.0
        bullish_ratio = 0.0

    # 5. Lifecycle features
    lifecycle = classify_lifecycle_stage(
        ticker=clean_ticker,
        cutoff_time=cutoff_time,
        social_posts=pt_posts,
        news_items=pt_news,
        attention_metrics=att,
        extracted_claims=pt_claims,
    )

    # 6. Forward CAR Target (only for training / evaluation, uses future bars)
    all_ticker_bars = [b for b in market_bars if b.ticker == clean_ticker]
    car_target = compute_forward_car(all_ticker_bars, cutoff_time, horizon_steps=forward_horizon)
    binary_target = 1 if (car_target is not None and car_target > 0.0) else 0

    return {
        "ticker": clean_ticker,
        "cutoff_time": cutoff_time.isoformat(),
        # Market
        "return_1d": round(ret_1d, 4),
        "abnormal_volume": round(abnormal_vol, 2),
        "price_volatility": round(volatility, 4),
        # Sentiment
        "avg_sentiment": round(avg_sentiment, 3),
        "sentiment_intensity": round(avg_intensity, 3),
        "bullish_post_ratio": round(bullish_ratio, 3),
        # Attention
        "attention_multiplier": round(att.attention_multiplier, 2),
        "current_mentions": att.current_mentions,
        "author_diversity": round(att.author_diversity_ratio, 3),
        "is_pump_flag": 1 if att.is_coordinated_pump else 0,
        # Lifecycle
        "stage_index": lifecycle.stage_index,
        "has_official_news": 1 if lifecycle.has_official_confirmation else 0,
        "hours_since_first_rumour": round(lifecycle.hours_since_first_rumour or 0.0, 1),
        # Target
        "forward_car": round(car_target, 4) if car_target is not None else None,
        "target_outperform": binary_target if car_target is not None else None,
    }


def build_time_series_dataset(
    bundle: DatasetBundle,
    horizon_steps: int = 1,
) -> pd.DataFrame:
    """Construct complete tabular dataset across all unique bar timestamps and tickers."""
    rows = []
    tickers = sorted(list({b.ticker for b in bundle.market_bars}))

    for t in tickers:
        t_bars = sorted(bundle.get_market_for_ticker(t), key=lambda b: b.timestamp)
        for bar in t_bars:
            row = build_feature_row(
                ticker=t,
                cutoff_time=bar.timestamp,
                social_posts=bundle.social_posts,
                news_items=bundle.news_items,
                market_bars=bundle.market_bars,
                forward_horizon=horizon_steps,
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["cutoff_time"] = pd.to_datetime(df["cutoff_time"])
        df = df.sort_values(by="cutoff_time").reset_index(drop=True)
    return df
