"""Historical analogue matching engine (data-dependent stretch feature).

Retrieves top-k historical setup analogues based on feature vector similarity
to provide explainable context on how similar past hype episodes resolved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class HistoricalAnalogue:
    """A matched historical market analogue."""
    ticker: str
    date: str
    similarity: float
    forward_car: Optional[float]
    stage_index: int
    signal: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "date": self.date,
            "similarity": round(self.similarity, 3),
            "forward_car": round(self.forward_car, 4) if self.forward_car is not None else None,
            "stage_index": self.stage_index,
            "signal": self.signal,
        }


def find_historical_analogues(
    query_row: Dict[str, Any],
    historical_df: pd.DataFrame,
    features: Optional[List[str]] = None,
    top_k: int = 3,
) -> List[HistoricalAnalogue]:
    """Find top-k closest historical setups based on Euclidean feature distance.

    Args:
        query_row: Current point-in-time feature dictionary.
        historical_df: Tabular DataFrame of past setups with historical forward returns.
        features: Subset of numerical feature names to compare.
        top_k: Number of nearest matches to return.

    Returns:
        List of HistoricalAnalogue items.
    """
    if historical_df.empty:
        return []

    feat_cols = features or [
        "return_1d", "abnormal_volume", "avg_sentiment",
        "attention_multiplier", "stage_index",
    ]
    # Filter to existing columns in historical_df
    available_feats = [c for c in feat_cols if c in historical_df.columns]
    if not available_feats:
        return []

    query_vec = np.array([float(query_row.get(c, 0.0)) for c in available_feats], dtype=float)

    hist_matrix = historical_df[available_feats].fillna(0.0).to_numpy(dtype=float)

    # Standardize scale
    stds = np.std(hist_matrix, axis=0)
    stds[stds == 0] = 1.0
    norm_query = query_vec / stds
    norm_hist = hist_matrix / stds

    # Compute Euclidean distances
    dists = np.linalg.norm(norm_hist - norm_query, axis=1)
    similarities = 1.0 / (1.0 + dists)

    top_indices = np.argsort(dists)[:top_k]
    analogues: List[HistoricalAnalogue] = []

    for idx in top_indices:
        row = historical_df.iloc[idx]
        car = float(row["forward_car"]) if pd.notna(row.get("forward_car")) else None
        signal = "Outperformed" if (car is not None and car > 0) else "Underperformed"
        analogues.append(
            HistoricalAnalogue(
                ticker=str(row.get("ticker", "UNKNOWN")),
                date=str(row.get("cutoff_time", "N/A"))[:10],
                similarity=float(similarities[idx]),
                forward_car=car,
                stage_index=int(row.get("stage_index", 1)),
                signal=signal,
            )
        )

    return analogues
