"""Unit tests for ML feature builder, leakage guardrails, and movement predictor (TASK-08)."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from hypelens.analytics.analogues import find_historical_analogues
from hypelens.data.loader import load_benchmark_dataset
from hypelens.ml.feature_builder import (
    FEATURE_SETS,
    build_feature_row,
    build_time_series_dataset,
    compute_forward_car,
)
from hypelens.ml.movement_model import MovementPrediction, MovementPredictor
from hypelens.models.data_models import MarketBar


def _make_bar(ticker: str, ts: datetime, close: float, bench_ret: float = 0.0) -> MarketBar:
    return MarketBar(
        ticker=ticker,
        timestamp=ts,
        open=close * 0.99,
        high=close * 1.01,
        low=close * 0.98,
        close=close,
        volume=100000.0,
        benchmark_return=bench_ret,
        abnormal_volume=1.2,
    )


def test_compute_forward_car_math():
    """Verify forward CAR cumulative abnormal return arithmetic."""
    t0 = datetime(2026, 10, 5, 16, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 10, 6, 16, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 10, 7, 16, 0, tzinfo=timezone.utc)

    # Bar 0: P=100
    # Bar 1: P=110 (+10%), bench = +2% -> abnormal = +8% (0.08)
    # Bar 2: P=121 (+10%), bench = +1% -> abnormal = +9% (0.09)
    # CAR(t0, t1) = 0.08
    # CAR(t0, t2) = 0.08 + 0.09 = 0.17
    bars = [
        _make_bar("NVDA", t0, 100.0, 0.0),
        _make_bar("NVDA", t1, 110.0, 0.02),
        _make_bar("NVDA", t2, 121.0, 0.01),
    ]

    car_1 = compute_forward_car(bars, cutoff_time=t0, horizon_steps=1)
    assert car_1 is not None
    assert round(car_1, 4) == 0.08

    car_2 = compute_forward_car(bars, cutoff_time=t0, horizon_steps=2)
    assert car_2 is not None
    assert round(car_2, 4) == 0.17

    # Horizon exceeds available future observations
    assert compute_forward_car(bars, cutoff_time=t0, horizon_steps=5) is None


def test_feature_builder_zero_lookahead_leakage():
    """Ensure features at cutoff t are strictly identical regardless of future additions."""
    bundle = load_benchmark_dataset()
    cutoff = datetime(2026, 10, 6, 16, 0, tzinfo=timezone.utc)

    # Base feature row
    row_original = build_feature_row(
        ticker="NVDA",
        cutoff_time=cutoff,
        social_posts=bundle.social_posts,
        news_items=bundle.news_items,
        market_bars=bundle.market_bars,
    )

    # Inject a massive future news item and future 10x price bar on Oct 9
    future_news = bundle.news_items.copy()
    future_news.append(
        type(bundle.news_items[0])(
            news_id="future_n", headline="FUTURE LEAK", publisher="Wire",
            timestamp=cutoff + timedelta(days=3), body="Future", tickers=["NVDA"],
        )
    )
    future_bars = bundle.market_bars.copy()
    future_bars.append(_make_bar("NVDA", cutoff + timedelta(days=3), 999.0))

    row_with_future = build_feature_row(
        ticker="NVDA",
        cutoff_time=cutoff,
        social_posts=bundle.social_posts,
        news_items=future_news,
        market_bars=future_bars,
    )

    # All predictive feature values must remain strictly identical
    for feat_tier in FEATURE_SETS.values():
        for feat in feat_tier:
            assert row_original[feat] == row_with_future[feat], f"Leakage detected in {feat}"


def test_build_time_series_dataset_bundle():
    """Verify synthesis of tabular DataFrame across all benchmark fixtures."""
    bundle = load_benchmark_dataset()
    df = build_time_series_dataset(bundle, horizon_steps=1)
    assert not df.empty
    assert "return_1d" in df.columns
    assert "stage_index" in df.columns
    assert "target_outperform" in df.columns
    assert df["cutoff_time"].is_monotonic_increasing


def test_movement_predictor_fit_and_signal():
    """Verify training and prediction with decision-support explanation output."""
    bundle = load_benchmark_dataset()
    df = build_time_series_dataset(bundle, horizon_steps=1)

    predictor = MovementPredictor()
    predictor.fit(df)
    assert predictor.is_fitted is True

    sample_row = df.iloc[0].to_dict()
    pred = predictor.predict_row(sample_row)

    assert isinstance(pred, MovementPrediction)
    assert 0.0 <= pred.probability_outperform <= 1.0
    assert pred.signal_class in [
        "Pre-News Run-Up Candidate", "Crowded / Late Hype (Reversal Risk)",
        "Sell-the-News Reversal Risk", "Continuation Candidate",
        "Avoid / Suspected Manipulation", "Neutral / Low Opportunity",
    ]
    assert len(pred.explanation) > 0


def test_historical_analogues_retrieval():
    """Verify nearest-neighbor setup retrieval."""
    bundle = load_benchmark_dataset()
    df = build_time_series_dataset(bundle, horizon_steps=1)
    query = df.iloc[-1].to_dict()

    analogues = find_historical_analogues(query, df, top_k=2)
    assert len(analogues) == 2
    assert analogues[0].similarity >= 0.0
    assert analogues[0].ticker in ["NVDA", "TSLA", "AAPL"]
