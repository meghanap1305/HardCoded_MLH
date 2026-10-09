"""The six-experiment evaluation suite from the HypeLens research foundation.

Implements:
  Experiment A: Market only
  Experiment B: Sentiment only
  Experiment C: Market + sentiment
  Experiment D: Market + sentiment + attention
  Experiment E: Full multi-source with rumour-lifecycle features
  Experiment F: Event study around official announcements
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from hypelens.data.loader import DatasetBundle
from hypelens.ml.feature_builder import FEATURE_SETS, compute_forward_car
from hypelens.ml.movement_model import MovementPredictor


@dataclass
class ExperimentResult:
    """Outcome and evaluation metrics for a specific experimental tier."""
    experiment_id: str
    experiment_name: str
    features_used: List[str]
    sample_size: int
    train_size: int
    test_size: int
    accuracy: Optional[float] = None
    f1: Optional[float] = None
    mean_forward_car: Optional[float] = None
    status: str = "success"
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "features_used": self.features_used,
            "sample_size": self.sample_size,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "accuracy": round(self.accuracy, 3) if self.accuracy is not None else None,
            "f1": round(self.f1, 3) if self.f1 is not None else None,
            "mean_forward_car": (
                round(self.mean_forward_car, 4) if self.mean_forward_car is not None else None
            ),
            "status": self.status,
            "extra_metrics": self.extra_metrics,
        }


def _run_tabular_experiment(
    df: pd.DataFrame,
    features: List[str],
    exp_id: str,
    exp_name: str,
    train_ratio: float = 0.65,
) -> ExperimentResult:
    """Evaluate a tabular feature tier using chronological forward splitting."""
    valid_df = df.dropna(subset=["target_outperform"]).copy()
    n = len(valid_df)

    if n < 2:
        return ExperimentResult(
            experiment_id=exp_id,
            experiment_name=exp_name,
            features_used=features,
            sample_size=n,
            train_size=n,
            test_size=0,
            status="insufficient_data",
        )

    # Chronological forward split (no random shuffling)
    split_idx = max(1, int(n * train_ratio))
    train_df = valid_df.iloc[:split_idx]
    test_df = valid_df.iloc[split_idx:] if split_idx < n else valid_df.iloc[split_idx - 1 :]

    predictor = MovementPredictor(feature_columns=features)
    predictor.fit(train_df)

    # Evaluate on test set
    X_test = test_df[features].fillna(0.0)
    y_test = test_df["target_outperform"].astype(int)

    preds = predictor.model.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    mean_car = float(test_df["forward_car"].mean()) if "forward_car" in test_df else None

    return ExperimentResult(
        experiment_id=exp_id,
        experiment_name=exp_name,
        features_used=features,
        sample_size=n,
        train_size=len(train_df),
        test_size=len(test_df),
        accuracy=acc,
        f1=f1,
        mean_forward_car=mean_car,
        status="success",
    )


def run_experiment_a_market(df: pd.DataFrame) -> ExperimentResult:
    """Experiment A: Market-only baseline."""
    return _run_tabular_experiment(
        df, FEATURE_SETS["market"], "exp_a", "Experiment A: Market Only"
    )


def run_experiment_b_sentiment(df: pd.DataFrame) -> ExperimentResult:
    """Experiment B: Sentiment-only features."""
    return _run_tabular_experiment(
        df, FEATURE_SETS["sentiment"], "exp_b", "Experiment B: Sentiment Only"
    )


def run_experiment_c_market_sentiment(df: pd.DataFrame) -> ExperimentResult:
    """Experiment C: Market + Sentiment features."""
    feats = FEATURE_SETS["market"] + FEATURE_SETS["sentiment"]
    return _run_tabular_experiment(
        df, feats, "exp_c", "Experiment C: Market + Sentiment"
    )


def run_experiment_d_market_sentiment_attention(df: pd.DataFrame) -> ExperimentResult:
    """Experiment D: Market + Sentiment + Attention features."""
    feats = FEATURE_SETS["market"] + FEATURE_SETS["sentiment"] + FEATURE_SETS["attention"]
    return _run_tabular_experiment(
        df, feats, "exp_d", "Experiment D: Market + Sentiment + Attention"
    )


def run_experiment_e_full_lifecycle(df: pd.DataFrame) -> ExperimentResult:
    """Experiment E: Full multi-source with rumour-lifecycle features."""
    feats = (
        FEATURE_SETS["market"]
        + FEATURE_SETS["sentiment"]
        + FEATURE_SETS["attention"]
        + FEATURE_SETS["lifecycle"]
    )
    return _run_tabular_experiment(
        df, feats, "exp_e", "Experiment E: Full Multi-Source + Lifecycle"
    )


def run_experiment_f_event_study(
    bundle: DatasetBundle,
    pre_window_bars: int = 2,
    post_window_bars: int = 2,
) -> ExperimentResult:
    """Experiment F: Event study examining abnormal returns around official news events."""
    event_studies: List[Dict[str, Any]] = []

    for news in bundle.news_items:
        for ticker in news.tickers:
            t_bars = sorted(bundle.get_market_for_ticker(ticker), key=lambda b: b.timestamp)
            if not t_bars:
                continue

            # Identify announcement bar (nearest bar on or immediately after news)
            ann_indices = [i for i, b in enumerate(t_bars) if b.timestamp >= news.timestamp]
            if not ann_indices:
                continue
            ann_idx = ann_indices[0]

            # Pre-event CAR: [ann_idx - pre_window, ann_idx]
            pre_start = max(0, ann_idx - pre_window_bars)
            pre_car = 0.0
            for i in range(pre_start + 1, ann_idx + 1):
                p_prev = t_bars[i - 1].close
                p_curr = t_bars[i].close
                ret = (p_curr - p_prev) / p_prev if p_prev > 0 else 0.0
                pre_car += (ret - t_bars[i].benchmark_return)

            # Post-event CAR: [ann_idx, ann_idx + post_window]
            post_end = min(len(t_bars) - 1, ann_idx + post_window_bars)
            post_car = 0.0
            for i in range(ann_idx + 1, post_end + 1):
                p_prev = t_bars[i - 1].close
                p_curr = t_bars[i].close
                ret = (p_curr - p_prev) / p_prev if p_prev > 0 else 0.0
                post_car += (ret - t_bars[i].benchmark_return)

            event_studies.append({
                "ticker": ticker,
                "news_id": news.news_id,
                "pre_car": pre_car,
                "post_car": post_car,
                "reversal": (pre_car > 0.0 and post_car < 0.0),
            })

    if not event_studies:
        return ExperimentResult(
            experiment_id="exp_f",
            experiment_name="Experiment F: Official News Event Study",
            features_used=["pre_event_car", "post_event_car"],
            sample_size=0,
            train_size=0,
            test_size=0,
            status="insufficient_events",
        )

    mean_pre = float(np.mean([e["pre_car"] for e in event_studies]))
    mean_post = float(np.mean([e["post_car"] for e in event_studies]))
    reversal_rate = float(np.mean([1.0 if e["reversal"] else 0.0 for e in event_studies]))

    return ExperimentResult(
        experiment_id="exp_f",
        experiment_name="Experiment F: Official News Event Study",
        features_used=["pre_event_car", "post_event_car"],
        sample_size=len(event_studies),
        train_size=len(event_studies),
        test_size=len(event_studies),
        mean_forward_car=mean_post,
        extra_metrics={
            "mean_pre_event_car": round(mean_pre, 4),
            "mean_post_event_car": round(mean_post, 4),
            "reversal_frequency": round(reversal_rate, 3),
            "events_evaluated": len(event_studies),
        },
        status="success",
    )


def run_all_experiments(
    df: pd.DataFrame,
    bundle: Optional[DatasetBundle] = None,
) -> Dict[str, ExperimentResult]:
    """Execute all six comparative experiments across the dataset."""
    results = {
        "exp_a": run_experiment_a_market(df),
        "exp_b": run_experiment_b_sentiment(df),
        "exp_c": run_experiment_c_market_sentiment(df),
        "exp_d": run_experiment_d_market_sentiment_attention(df),
        "exp_e": run_experiment_e_full_lifecycle(df),
    }
    if bundle is not None:
        results["exp_f"] = run_experiment_f_event_study(bundle)
    return results
