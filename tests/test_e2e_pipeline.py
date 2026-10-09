"""End-to-end workflow verification for HypeLens (TASK-10).

Validates the full multimodal pipeline from data ingestion to cognitive extraction,
attention modeling, rumour lifecycle classification, ML experiments, and UI rendering.
Does not require live APIs or a running Ollama daemon.
"""

from datetime import datetime, timezone

import pandas as pd
import pytest

from hypelens.analytics.analogues import find_historical_analogues
from hypelens.analytics.attention_engine import calculate_attention_metrics
from hypelens.analytics.lifecycle_classifier import (
    LifecycleStage,
    classify_lifecycle_stage,
)
from hypelens.config import FIXTURES_DIR, get_config
from hypelens.data.loader import load_benchmark_dataset
from hypelens.data.video import extract_keyframes
from hypelens.ml.experiments import run_all_experiments
from hypelens.ml.feature_builder import (
    build_feature_row,
    build_time_series_dataset,
    compute_forward_car,
)
from hypelens.ml.movement_model import MovementPredictor
from hypelens.nlp.extractor import GemmaExtractor
from hypelens.ui.components import (
    format_multimodal_card,
    format_overview_metrics,
    format_signal_card,
    format_timeline_events,
)


def test_full_pipeline_end_to_end(tmp_path):
    """Execute complete offline end-to-end integration workflow."""
    # 1. Configuration & Dependency Checks
    cfg = get_config()
    assert "NVDA" in cfg.stock_universe

    # 2. Video frame extraction
    clip_path = FIXTURES_DIR / "sample_clip.mp4"
    if clip_path.exists():
        video_res = extract_keyframes(clip_path, max_frames=2, output_dir=tmp_path / "frames")
        assert video_res.success is True
        assert len(video_res.frames) > 0

    # 3. Load Multi-Stream Benchmark Fixtures
    bundle = load_benchmark_dataset()
    assert len(bundle.social_posts) > 0
    assert len(bundle.news_items) > 0
    assert len(bundle.market_bars) > 0
    assert len(bundle.quarantined) >= 3  # Quarantines missing timestamps

    # 4. Multimodal Cognitive Extraction (Deterministic Offline Mode)
    extractor = GemmaExtractor(force_mock=True)
    post_nvda = bundle.get_posts_for_ticker("NVDA")[0]
    extraction_res = extractor.extract_from_post(post_nvda)
    assert extraction_res.is_mock is True
    assert extraction_res.claim.ticker == "NVDA"
    assert extraction_res.claim.status.value in [
        "rumour_unverified", "credible_report", "officially_confirmed", "denied_retracted"
    ]

    # 5. Attention Modeling & Pump Detection
    att = calculate_attention_metrics(bundle.social_posts, "NVDA")
    assert att.current_mentions >= 1
    assert att.attention_multiplier >= 1.0

    # 6. Point-in-Time Rumour Lifecycle Classification
    cutoff_ts = max(b.timestamp for b in bundle.get_market_for_ticker("NVDA"))
    lifecycle = classify_lifecycle_stage(
        ticker="NVDA",
        cutoff_time=cutoff_ts,
        social_posts=bundle.social_posts,
        news_items=bundle.news_items,
        attention_metrics=att,
    )
    assert isinstance(lifecycle.stage, LifecycleStage)
    assert 1 <= lifecycle.stage_index <= 5

    # 7. Leakage-Safe Feature Synthesis & Movement Prediction
    df = build_time_series_dataset(bundle, horizon_steps=1)
    assert not df.empty
    assert "return_1d" in df.columns
    assert "stage_index" in df.columns

    predictor = MovementPredictor()
    predictor.fit(df)
    assert predictor.is_fitted is True

    test_row = df.iloc[-1].to_dict()
    pred = predictor.predict_row(test_row)
    assert 0.0 <= pred.probability_outperform <= 1.0

    # 8. Historical Analogue Matching (Stretch Feature)
    analogues = find_historical_analogues(test_row, df, top_k=2)
    assert len(analogues) == 2

    # 9. Six-Experiment Comparative Suite Execution
    exp_results = run_all_experiments(df, bundle)
    assert len(exp_results) == 6
    for exp_id in ["exp_a", "exp_b", "exp_c", "exp_d", "exp_e", "exp_f"]:
        assert exp_id in exp_results
        assert exp_results[exp_id].status == "success"

    # 10. UI Component Formatting
    overview_table = format_overview_metrics(bundle)
    assert not overview_table.empty

    signal_card = format_signal_card(pred)
    assert signal_card["ticker"] == test_row["ticker"]

    timeline = format_timeline_events(bundle.social_posts, bundle.news_items)
    assert len(timeline) > 0

    card = format_multimodal_card(post_nvda, claim=extraction_res.claim, is_mock=True)
    assert card["provenance_banner"] == "[DETERMINISTIC MOCK EXTRACTOR]"
