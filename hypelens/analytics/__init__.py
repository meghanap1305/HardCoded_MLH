"""Analytics and signal detection engines for HypeLens."""

from hypelens.analytics.analogues import (
    HistoricalAnalogue,
    find_historical_analogues,
)
from hypelens.analytics.attention_engine import (
    AttentionMetrics,
    calculate_attention_metrics,
)
from hypelens.analytics.lifecycle_classifier import (
    LifecycleClassification,
    LifecycleStage,
    classify_lifecycle_stage,
)
from hypelens.analytics.sector_engine import (
    SECTOR_DESCRIPTIONS,
    STOCK_SECTORS,
    build_sector_lifecycle_table,
    categorize_stocks_by_sector,
    get_sector_for_ticker,
)

__all__ = [
    "AttentionMetrics",
    "calculate_attention_metrics",
    "LifecycleStage",
    "LifecycleClassification",
    "classify_lifecycle_stage",
    "HistoricalAnalogue",
    "find_historical_analogues",
    "STOCK_SECTORS",
    "SECTOR_DESCRIPTIONS",
    "get_sector_for_ticker",
    "categorize_stocks_by_sector",
    "build_sector_lifecycle_table",
]
