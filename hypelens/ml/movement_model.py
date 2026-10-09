"""Baseline short-term movement predictor and opportunity signal generator.

Trains simple, explainable classifiers on leakage-safe features and maps outputs
into decision-support opportunity classifications with risk warnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression


@dataclass
class MovementPrediction:
    """Decision-support movement prediction output."""
    ticker: str
    probability_outperform: float
    signal_class: str
    confidence: float
    lifecycle_stage_index: int
    features_used: List[str]
    explanation: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "probability_outperform": round(self.probability_outperform, 3),
            "signal_class": self.signal_class,
            "confidence": round(self.confidence, 3),
            "lifecycle_stage_index": self.lifecycle_stage_index,
            "features_used": self.features_used,
            "explanation": self.explanation,
        }


class MovementPredictor:
    """Classifier predicting short-term outperformance probability."""

    def __init__(self, feature_columns: Optional[List[str]] = None):
        self.feature_columns = feature_columns or [
            "return_1d", "abnormal_volume", "avg_sentiment",
            "attention_multiplier", "stage_index",
        ]
        self.model: Any = None
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> MovementPredictor:
        """Fit model chronologically using valid rows with defined targets."""
        train_df = df.dropna(subset=["target_outperform"]).copy()
        if len(train_df) < 2 or train_df["target_outperform"].nunique() < 2:
            # Fall back to prior DummyClassifier for tiny sample or single class
            self.model = DummyClassifier(strategy="most_frequent")
            if len(train_df) > 0:
                X = train_df[self.feature_columns].fillna(0.0)
                y = train_df["target_outperform"].astype(int)
                self.model.fit(X, y)
            self.is_fitted = True
            return self

        X = train_df[self.feature_columns].fillna(0.0)
        y = train_df["target_outperform"].astype(int)

        self.model = LogisticRegression(max_iter=200, C=1.0)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_row(self, row: Dict[str, Any]) -> MovementPrediction:
        """Generate opportunity signal and probability for a single point-in-time feature row."""
        ticker = row.get("ticker", "UNKNOWN")
        stage_idx = int(row.get("stage_index", 1))
        is_pump = bool(row.get("is_pump_flag", 0))
        att_mult = float(row.get("attention_multiplier", 1.0))
        sent = float(row.get("avg_sentiment", 0.0))

        # Default probability if not fitted
        if not self.is_fitted or self.model is None:
            prob = 0.5
        else:
            X_vec = pd.DataFrame([{col: row.get(col, 0.0) for col in self.feature_columns}])
            try:
                probs = self.model.predict_proba(X_vec)
                prob = float(probs[0][1]) if probs.shape[1] > 1 else float(probs[0][0])
            except Exception:
                prob = 0.5

        explanation: List[str] = []

        # Map to signal classes using lifecycle stage and probability
        if is_pump:
            signal_class = "Avoid / Suspected Manipulation"
            explanation.append("Coordinated spam patterns or repetitive bot chatter detected.")
        elif stage_idx in (1, 2) and sent > 0.2 and prob >= 0.5:
            signal_class = "Pre-News Run-Up Candidate"
            explanation.append(f"Early rumour phase (Stage {stage_idx}) with positive sentiment ({sent:+.2f}).")
            explanation.append("Price has not yet reacted to official confirmation.")
        elif stage_idx == 3 and att_mult >= 3.0:
            signal_class = "Crowded / Late Hype (Reversal Risk)"
            explanation.append(f"Peak attention spike ({att_mult:.1f}x) without formal confirmation.")
        elif stage_idx in (4, 5) and prob < 0.45:
            signal_class = "Sell-the-News Reversal Risk"
            explanation.append("Official announcement digested; model anticipates reversal / profit taking.")
        elif stage_idx in (4, 5) and prob >= 0.55:
            signal_class = "Continuation Candidate"
            explanation.append("Official announcement confirmed with sustained forward momentum.")
        else:
            signal_class = "Neutral / Low Opportunity"
            explanation.append("Signals within normal historical volatility thresholds.")

        return MovementPrediction(
            ticker=ticker,
            probability_outperform=prob,
            signal_class=signal_class,
            confidence=max(prob, 1.0 - prob),
            lifecycle_stage_index=stage_idx,
            features_used=self.feature_columns,
            explanation=explanation,
        )
