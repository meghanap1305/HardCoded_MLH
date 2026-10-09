"""Machine learning and evaluation experiment modules for HypeLens."""

from hypelens.ml.experiments import (
    ExperimentResult,
    run_all_experiments,
    run_experiment_a_market,
    run_experiment_b_sentiment,
    run_experiment_c_market_sentiment,
    run_experiment_d_market_sentiment_attention,
    run_experiment_e_full_lifecycle,
    run_experiment_f_event_study,
)
from hypelens.ml.feature_builder import (
    FEATURE_SETS,
    build_feature_row,
    build_time_series_dataset,
    compute_forward_car,
)
from hypelens.ml.movement_model import MovementPrediction, MovementPredictor

__all__ = [
    "FEATURE_SETS",
    "build_feature_row",
    "compute_forward_car",
    "build_time_series_dataset",
    "MovementPredictor",
    "MovementPrediction",
    "ExperimentResult",
    "run_experiment_a_market",
    "run_experiment_b_sentiment",
    "run_experiment_c_market_sentiment",
    "run_experiment_d_market_sentiment_attention",
    "run_experiment_e_full_lifecycle",
    "run_experiment_f_event_study",
    "run_all_experiments",
]
