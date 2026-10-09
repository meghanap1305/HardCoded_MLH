"""Dashboard components, formatting helpers, and research guardrails for HypeLens.

Provides deterministic data preparation functions for the Streamlit interface,
enforcing explicit research disclaimers, clear model provenance (mock vs Gemma 4),
and missing-data alerts.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from hypelens.analytics.attention_engine import calculate_attention_metrics
from hypelens.analytics.lifecycle_classifier import classify_lifecycle_stage
from hypelens.data.loader import DatasetBundle
from hypelens.ml.movement_model import MovementPrediction
from hypelens.models.data_models import (
    AccountTier,
    FinancialNewsItem,
    PlatformType,
    SocialPost,
)
from hypelens.nlp.parser import ExtractedClaim


RESEARCH_DISCLAIMER = (
    "RESEARCH DECISION-SUPPORT PROTOTYPE ONLY. NOT FINANCIAL ADVICE. "
    "HypeLens detects social narrative patterns, attention spikes, and rumour-to-news lifecycles. "
    "It does not guarantee future stock movements or execute automated trades."
)


def get_terminal_css() -> str:
    """Cyberpunk / Bloomberg-Terminal aesthetic custom CSS for Streamlit."""
    return """
    <style>
    /* Dark Cyberpunk / Fintech Terminal Styling */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre, .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Top Ticker Tape Marquee */
    .ticker-tape-container {
        width: 100%;
        overflow: hidden;
        background: linear-gradient(90deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.95));
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 8px;
        padding: 8px 0;
        margin-bottom: 20px;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.15);
    }

    .ticker-tape {
        display: flex;
        gap: 32px;
        white-space: nowrap;
        animation: marquee 25s linear infinite;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        font-weight: 600;
    }

    @keyframes marquee {
        0% { transform: translateX(0%); }
        100% { transform: translateX(-50%); }
    }

    .tape-item {
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }

    .tape-up { color: #10B981; }
    .tape-down { color: #EF4444; }
    .tape-warn { color: #F59E0B; }
    .tape-cyan { color: #06B6D4; }

    /* Glassmorphic Metric Cards */
    .hud-card {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 12px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }

    .hud-card:hover {
        border-color: rgba(56, 189, 248, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(56, 189, 248, 0.2);
    }

    .hud-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-family: 'JetBrains Mono', monospace;
    }

    .badge-green { background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid #10B981; }
    .badge-red { background: rgba(239, 68, 68, 0.15); color: #F87171; border: 1px solid #EF4444; }
    .badge-cyan { background: rgba(6, 182, 212, 0.15); color: #38BDF8; border: 1px solid #06B6D4; }
    .badge-amber { background: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid #F59E0B; }

    /* Pulsing radar dot */
    .pulse-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 8px #10B981;
        animation: pulse-ring 2s infinite;
        margin-right: 6px;
    }

    @keyframes pulse-ring {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Lifecycle Timeline Stepper */
    .stepper-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 20px 0;
        padding: 16px 20px;
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        position: relative;
    }

    .stepper-step {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        z-index: 2;
        width: 18%;
    }

    .step-circle {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 14px;
        margin-bottom: 8px;
        background: #1E293B;
        color: #94A3B8;
        border: 2px solid #334155;
        transition: all 0.3s ease;
    }

    .step-active .step-circle {
        background: #0284C7;
        color: #FFFFFF;
        border-color: #38BDF8;
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.6);
        animation: pulse-ring 2s infinite;
    }

    .step-completed .step-circle {
        background: #065F46;
        color: #34D399;
        border-color: #10B981;
    }

    .step-label {
        font-size: 11px;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .step-active .step-label {
        color: #38BDF8;
        font-weight: 700;
    }
    </style>
    """


def render_lifecycle_stepper_html(active_stage_idx: int) -> str:
    """Render an interconnected 5-node glowing lifecycle stepper component."""
    stages = [
        (1, "1. Whisper", "Early leak / low vol"),
        (2, "2. Hype Build", "Mentions accelerate"),
        (3, "3. Peak Attention", "Viral retail FOMO"),
        (4, "4. Confirmation", "Official PR released"),
        (5, "5. Reaction", "Sell news vs continue"),
    ]

    items_html = []
    for idx, name, sub in stages:
        if idx == active_stage_idx:
            status_cls = "step-active"
        elif idx < active_stage_idx:
            status_cls = "step-completed"
        else:
            status_cls = "step-pending"

        items_html.append(f"""
        <div class="stepper-step {status_cls}">
            <div class="step-circle">{idx}</div>
            <div class="step-label">{name}</div>
            <div style="font-size: 9px; color: #64748B; margin-top: 2px;">{sub}</div>
        </div>
        """)

    return f"""
    <div class="stepper-container">
        {"".join(items_html)}
    </div>
    """


def render_ticker_tape_html(tickers_summary: List[Dict[str, Any]]) -> str:
    """Render an animated continuous marquee ticker tape."""
    items = []
    for t in tickers_summary:
        ticker = t.get("ticker", "TICKER")
        att = t.get("att_mult", "1.0x")
        stage = t.get("stage", "Stage 1")
        color_cls = "tape-up" if "Pre-News" in t.get("signal", "") else ("tape-down" if "Risk" in t.get("signal", "") else "tape-cyan")
        items.append(
            f'<div class="tape-item"><span style="color:#F1F5F9; font-weight:800;">${ticker}</span> '
            f'<span class="{color_cls}">ATTENTION: {att}</span> • '
            f'<span style="color:#94A3B8;">[{stage}]</span></div>'
        )

    # Repeat twice for seamless loop
    tape_content = " ".join(items) + " " + " ".join(items)
    return f"""
    <div class="ticker-tape-container">
        <div class="ticker-tape">
            {tape_content}
        </div>
    </div>
    """


def format_overview_metrics(bundle: DatasetBundle) -> pd.DataFrame:
    """Format high-level hype and market overview table for all tickers."""
    tickers = sorted(list({b.ticker for b in bundle.market_bars} | {
        t for p in bundle.social_posts for t in p.tickers
    }))

    rows: List[Dict[str, Any]] = []
    for ticker in tickers:
        posts = bundle.get_posts_for_ticker(ticker)
        bars = bundle.get_market_for_ticker(ticker)
        news = bundle.get_news_for_ticker(ticker)

        # Influencer breakdown
        small_count = sum(1 for p in posts if p.author.tier == AccountTier.SMALL)
        med_count = sum(1 for p in posts if p.author.tier == AccountTier.MEDIUM)
        large_count = sum(1 for p in posts if p.author.tier == AccountTier.LARGE)

        att = calculate_attention_metrics(posts, ticker)
        latest_ts = bars[-1].timestamp if bars else (posts[-1].timestamp if posts else datetime.now())
        lifecycle = classify_lifecycle_stage(ticker, latest_ts, posts, news, attention_metrics=att)

        latest_price = bars[-1].close if bars else None
        ret_1d = None
        if len(bars) >= 2 and bars[-2].close > 0:
            ret_1d = round((bars[-1].close - bars[-2].close) / bars[-2].close * 100.0, 2)

        rows.append({
            "Ticker": ticker,
            "Mentions": att.current_mentions,
            "Attention Multiplier": f"{att.attention_multiplier:.1f}x",
            "Influencers (S/M/L)": f"{small_count} / {med_count} / {large_count}",
            "Lifecycle Stage": lifecycle.stage.value.replace("_", " ").title(),
            "Latest Price": f"${latest_price:.2f}" if latest_price is not None else "N/A",
            "1D Return": f"{ret_1d:+.2f}%" if ret_1d is not None else "N/A",
            "Manipulation Alert": "FLAGGED" if att.is_coordinated_pump else "Normal",
        })

    return pd.DataFrame(rows)


def format_signal_card(
    prediction: MovementPrediction,
) -> Dict[str, Any]:
    """Format opportunity prediction and explanations into structured card payload."""
    signal = prediction.signal_class
    if "Pre-News" in signal:
        color = "green"
    elif "Risk" in signal or "Avoid" in signal:
        color = "red"
    elif "Continuation" in signal:
        color = "blue"
    else:
        color = "gray"

    return {
        "ticker": prediction.ticker,
        "signal_class": signal,
        "badge_color": color,
        "probability_outperform": round(prediction.probability_outperform * 100.0, 1),
        "confidence": round(prediction.confidence * 100.0, 1),
        "stage_index": prediction.lifecycle_stage_index,
        "explanation": prediction.explanation,
    }


def format_timeline_events(
    posts: List[SocialPost],
    news: List[FinancialNewsItem],
) -> List[Dict[str, Any]]:
    """Assemble a unified chronological event stream of social buzz and news."""
    events: List[Dict[str, Any]] = []

    for p in posts:
        events.append({
            "timestamp": p.timestamp.isoformat(),
            "time_dt": p.timestamp,
            "stream": "Community / Social",
            "platform": p.platform.value.replace("_", " ").title(),
            "author": f"@{p.author.username} ({p.author.tier.value} tier)",
            "summary": p.content_text,
            "is_official": p.author.is_official,
            "has_media": bool(p.video_path or p.image_paths),
        })

    for n in news:
        events.append({
            "timestamp": n.timestamp.isoformat(),
            "time_dt": n.timestamp,
            "stream": "Official / News Wire",
            "platform": n.publisher,
            "author": n.publisher,
            "summary": f"HEADLINE: {n.headline}",
            "is_official": True,
            "has_media": False,
        })

    return sorted(events, key=lambda e: e["time_dt"])


def format_multimodal_card(
    post: SocialPost,
    claim: Optional[ExtractedClaim] = None,
    is_mock: bool = True,
) -> Dict[str, Any]:
    """Format post with OCR, transcript, and cognitive claim extraction."""
    return {
        "post_id": post.post_id,
        "platform": post.platform.value,
        "timestamp": post.timestamp.isoformat(),
        "author": f"@{post.author.username} ({post.author.tier.value} tier)",
        "content_text": post.content_text,
        "ocr_text": post.ocr_text,
        "transcript": post.transcript,
        "video_path": post.video_path,
        "image_paths": post.image_paths,
        "claim": claim.claim if claim else "No claim extracted",
        "event_category": claim.event_category.value if claim else "unknown",
        "status": claim.status.value if claim else "unknown",
        "sentiment": round(claim.sentiment, 2) if claim else 0.0,
        "intensity": round(claim.intensity, 2) if claim else 0.0,
        "is_mock": is_mock,
        "provenance_banner": (
            "[DETERMINISTIC MOCK EXTRACTOR]" if is_mock else "[LOCAL GEMMA 4 INFERENCE]"
        ),
    }
