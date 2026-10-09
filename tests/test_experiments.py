"""Unit tests for the six-experiment comparative evaluation suite (TASK-08)."""

from hypelens.data.loader import load_benchmark_dataset
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
from hypelens.ml.feature_builder import build_time_series_dataset


def test_experiments_a_through_e_execution():
    """Verify execution of tabular feature experiments A through E."""
    bundle = load_benchmark_dataset()
    df = build_time_series_dataset(bundle, horizon_steps=1)

    exp_a = run_experiment_a_market(df)
    assert isinstance(exp_a, ExperimentResult)
    assert exp_a.experiment_id == "exp_a"
    assert "return_1d" in exp_a.features_used
    assert exp_a.accuracy is not None

    exp_b = run_experiment_b_sentiment(df)
    assert exp_b.experiment_id == "exp_b"
    assert "avg_sentiment" in exp_b.features_used

    exp_c = run_experiment_c_market_sentiment(df)
    assert exp_c.experiment_id == "exp_c"
    assert len(exp_c.features_used) == len(exp_a.features_used) + len(exp_b.features_used)

    exp_d = run_experiment_d_market_sentiment_attention(df)
    assert exp_d.experiment_id == "exp_d"
    assert "attention_multiplier" in exp_d.features_used

    exp_e = run_experiment_e_full_lifecycle(df)
    assert exp_e.experiment_id == "exp_e"
    assert "stage_index" in exp_e.features_used


def test_experiment_f_event_study():
    """Verify event study around official news announcements."""
    bundle = load_benchmark_dataset()
    exp_f = run_experiment_f_event_study(bundle, pre_window_bars=1, post_window_bars=1)

    assert isinstance(exp_f, ExperimentResult)
    assert exp_f.experiment_id == "exp_f"
    assert exp_f.status == "success"
    assert "mean_pre_event_car" in exp_f.extra_metrics
    assert "mean_post_event_car" in exp_f.extra_metrics
    assert "reversal_frequency" in exp_f.extra_metrics
    assert exp_f.extra_metrics["events_evaluated"] >= 1


def test_run_all_experiments_bundle():
    """Verify execution of full suite via run_all_experiments."""
    bundle = load_benchmark_dataset()
    df = build_time_series_dataset(bundle, horizon_steps=1)

    results = run_all_experiments(df, bundle)
    assert len(results) == 6
    for exp_key in ["exp_a", "exp_b", "exp_c", "exp_d", "exp_e", "exp_f"]:
        assert exp_key in results
        assert results[exp_key].status == "success"
