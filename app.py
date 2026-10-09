"""Streamlit Web Application for HypeLens: Multimodal Community Market Intelligence.

Demonstrates sector-based community hype capture, Instagram Reel & X post inspection,
rumour-to-news lifecycle tracking, and leakage-safe short-term opportunity signals.
Supports Live Indian Exchange (NSE/SEBI) feed monitoring, Real Social-Post Ingestion
(via official X API or manual user uploads/import), and local Gemma 4 multimodal analysis.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from hypelens.analytics.analogues import find_historical_analogues
from hypelens.analytics.attention_engine import calculate_attention_metrics
from hypelens.analytics.lifecycle_classifier import classify_lifecycle_stage
from hypelens.analytics.sector_engine import (
    SECTOR_DESCRIPTIONS,
    STOCK_SECTORS,
    aggregate_live_items_by_sector,
    build_sector_lifecycle_table,
    categorize_stocks_by_sector,
    get_sector_for_ticker,
)
from hypelens.config import get_config
from hypelens.data.loader import DatasetBundle, load_benchmark_dataset
from hypelens.ingestion.nse_announcements import OFFICIAL_FEEDS, OfficialExchangeIngestor
from hypelens.ingestion.social_connector import SocialIngestor
from hypelens.ml.experiments import run_all_experiments
from hypelens.ml.feature_builder import build_feature_row, build_time_series_dataset
from hypelens.ml.movement_model import MovementPredictor
from hypelens.models.taxonomy import get_taxonomy
from hypelens.nlp.extractor import GemmaExtractor
from hypelens.ui.components import (
    RESEARCH_DISCLAIMER,
    format_multimodal_card,
    format_overview_metrics,
    format_signal_card,
    format_timeline_events,
    get_terminal_css,
    render_lifecycle_stepper_html,
    render_ticker_tape_html,
)
from hypelens.utils.datetime_utils import format_ist, now_utc, to_ist


st.set_page_config(
    page_title="HypeLens Terminal | Sector Lifecycle Radar",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def get_cached_bundle() -> DatasetBundle:
    """Load and cache benchmark fixtures."""
    return load_benchmark_dataset()


@st.cache_data
def get_cached_dataset(horizon: int = 1) -> pd.DataFrame:
    """Synthesize and cache leakage-safe tabular dataset."""
    bundle = get_cached_bundle()
    return build_time_series_dataset(bundle, horizon_steps=horizon)


@st.cache_data(ttl=300)
def fetch_live_indian_feeds() -> Dict[str, Any]:
    """Fetch live NSE corporate announcements and SEBI regulatory circulars without fixtures."""
    ingestor = OfficialExchangeIngestor(use_offline_fallback=False, timeout=5.0)
    results = {
        "NSE": {"items": [], "status": "PENDING", "error": None, "url": OFFICIAL_FEEDS["NSE"]},
        "SEBI": {"items": [], "status": "PENDING", "error": None, "url": OFFICIAL_FEEDS["SEBI"]},
        "fetched_at": now_utc(),
    }

    # Fetch SEBI
    try:
        sebi_items = ingestor.fetch_exchange_feed(source="SEBI")
        results["SEBI"]["items"] = sebi_items
        results["SEBI"]["status"] = "LIVE_SUCCESS" if sebi_items else "NO_RECORDS"
    except Exception as e:
        results["SEBI"]["status"] = "NETWORK_ERROR"
        results["SEBI"]["error"] = str(e)

    # Fetch NSE
    try:
        nse_items = ingestor.fetch_exchange_feed(source="NSE")
        results["NSE"]["items"] = nse_items
        results["NSE"]["status"] = "LIVE_SUCCESS" if nse_items else "NO_RECORDS"
    except Exception as e:
        results["NSE"]["status"] = "NETWORK_ERROR"
        results["NSE"]["error"] = str(e)

    return results


def main():
    # Inject Cyberpunk / High-Tech Terminal CSS
    st.markdown(get_terminal_css(), unsafe_allow_html=True)

    # Initialize session state for user-submitted or ingested posts and analysis results
    if "session_social_posts" not in st.session_state:
        st.session_state.session_social_posts = []
    if "session_gemma_analyses" not in st.session_state:
        st.session_state.session_gemma_analyses = {}

    bundle = get_cached_bundle()
    cfg = get_config()
    taxonomy = get_taxonomy()
    ollama_online = cfg.ollama.check_health()
    social_ingestor = SocialIngestor()

    # Dynamic Ticker Tape Summary
    tape_items = []
    for t in cfg.stock_universe:
        p = bundle.get_posts_for_ticker(t)
        att = calculate_attention_metrics(p, t)
        tape_items.append({
            "ticker": t,
            "att_mult": f"{att.attention_multiplier:.1f}x",
            "stage": get_sector_for_ticker(t),
            "signal": "Risk" if att.is_coordinated_pump else "Normal",
        })
    st.markdown(render_ticker_tape_html(tape_items), unsafe_allow_html=True)

    # Header Bar
    head_col1, head_col2 = st.columns([3, 1])
    with head_col1:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
            <span style="font-size:32px;">⚡</span>
            <div>
                <h1 style="margin:0; font-size:28px; font-weight:800; letter-spacing:-0.02em; background:linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                    HYPELENS : SECTOR LIFECYCLE RADAR
                </h1>
                <div style="font-size:12px; font-family:'JetBrains Mono',monospace; color:#94A3B8;">
                    INDIA CAPITAL MARKETS • SECTOR TAXONOMY • GEMMA 4 MULTIMODAL TIMELINE INTELLIGENCE
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with head_col2:
        model_status_badge = (
            '<span class="hud-badge badge-green"><span class="pulse-dot"></span>LOCAL GEMMA 4 ONLINE</span>'
            if ollama_online else
            '<span class="hud-badge badge-red"><span class="pulse-dot" style="background:#EF4444;"></span>GEMMA 4 UNAVAILABLE</span>'
        )
        st.markdown(f"""
        <div style="text-align:right; padding-top:8px;">
            {model_status_badge}
            <div style="font-size:11px; color:#64748B; font-family:'JetBrains Mono',monospace; margin-top:4px;">
                ENGINE: {cfg.ollama.model.upper()} (Ollama: {'ONLINE' if ollama_online else 'OFFLINE'})
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.caption(f"🔒 {RESEARCH_DISCLAIMER}")

    # Sidebar Controls
    st.sidebar.markdown("""
    <div style="padding:10px 0; border-bottom:1px solid rgba(255,255,255,0.1); margin-bottom:15px;">
        <span style="font-size:18px; font-weight:800; color:#38BDF8;">RADAR & DATA MODE</span>
        <div style="font-size:11px; color:#64748B;">HACKATHON INTELLIGENCE HUD</div>
    </div>
    """, unsafe_allow_html=True)

    data_mode = st.sidebar.radio(
        "Data Mode",
        ["🔴 Live Indian Feed (NSE / SEBI)", "🧪 Demo Benchmark Mode (Curated)"],
        index=0,
    )

    if data_mode.startswith("🔴"):
        if st.sidebar.button("🔄 Refresh Live Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        live_data = fetch_live_indian_feeds()
        fetched_at_str = format_ist(live_data["fetched_at"])
        st.sidebar.markdown(f"**Last Refreshed:** `{fetched_at_str}`")

        nse_stat = live_data["NSE"]["status"]
        sebi_stat = live_data["SEBI"]["status"]
        nse_badge = "🟢 LIVE_SUCCESS" if nse_stat == "LIVE_SUCCESS" else f"🔴 {nse_stat}"
        sebi_badge = "🟢 LIVE_SUCCESS" if sebi_stat == "LIVE_SUCCESS" else f"🔴 {sebi_stat}"

        st.sidebar.markdown(f"- **NSE Feed:** {nse_badge} ({len(live_data['NSE']['items'])} records)")
        st.sidebar.markdown(f"- **SEBI Feed:** {sebi_badge} ({len(live_data['SEBI']['items'])} records)")
    else:
        live_data = None

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏢 Sector & Company Scope")

    taxonomy_sectors = sorted(list({p.sector for p in taxonomy.list_all_profiles()}))
    all_available_sectors = ["All Sectors"] + taxonomy_sectors + ["Market-wide / Unclassified"]
    selected_sector = st.sidebar.selectbox("Filter by Industrial Sector", all_available_sectors, index=0)

    if selected_sector == "All Sectors":
        company_options = ["None (All Companies)"] + taxonomy.list_all_symbols()
    else:
        matched_companies = [p.symbol for p in taxonomy.list_all_profiles() if p.sector == selected_sector]
        company_options = ["None (All Companies)"] + (matched_companies if matched_companies else taxonomy.list_all_symbols())

    selected_company = st.sidebar.selectbox("Filter by Individual Company (Optional)", company_options, index=0)
    has_company_filter = selected_company != "None (All Companies)"

    # Navigation Tabs
    if data_mode.startswith("🔴"):
        tab_live_overview, tab_live_timeline, tab_social_ingest = st.tabs([
            "🌐 Live Indian Sector Overview",
            "📜 Live Disclosures Timeline",
            "💬 Social Chatter & Gemma 4 Analysis",
        ])

        all_live_items = live_data["NSE"]["items"] + live_data["SEBI"]["items"]

        # TAB 1: Live Indian Sector Overview
        with tab_live_overview:
            st.markdown("### 🏢 Live Indian Market Sector Intelligence")
            st.write("Aggregated live disclosures from **National Stock Exchange (NSE)** and **SEBI Regulatory Feeds** organized by industrial sector.")

            if not all_live_items:
                st.warning("⚠️ No live records currently available. Check internet connection or refresh live feeds.")
            else:
                sector_agg = aggregate_live_items_by_sector(all_live_items)
                if selected_sector != "All Sectors":
                    display_sectors = {selected_sector: sector_agg[selected_sector]} if selected_sector in sector_agg else {}
                else:
                    display_sectors = sector_agg

                if not display_sectors:
                    st.info(f"No live records found for sector: **{selected_sector}**.")
                else:
                    cols = st.columns(min(len(display_sectors), 3) or 1)
                    for idx, (sec_name, sec_info) in enumerate(display_sectors.items()):
                        with cols[idx % 3]:
                            latest_str = format_ist(sec_info["latest_event_ts"])
                            comps_str = ", ".join(f"${c}" for c in sec_info["companies"]) if sec_info["companies"] else "Unclassified Market Disclosures"
                            st.markdown(f"""
                            <div class="hud-card" style="border: 1px solid #38BDF8; min-height: 180px; margin-bottom: 12px;">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <span style="font-size:16px; font-weight:800; color:#FFFFFF;">{sec_name}</span>
                                    <span class="hud-badge badge-green">{sec_info['event_count']} EVENTS</span>
                                </div>
                                <div style="font-size:11px; font-family:'JetBrains Mono',monospace; color:#94A3B8; margin: 8px 0;">
                                    LATEST: <b style="color:#38BDF8;">{latest_str}</b>
                                </div>
                                <div style="font-size:12px; color:#CBD5E1; margin-bottom: 6px;">
                                    Active Companies: <b style="color:#F1F5F9;">{comps_str}</b>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("### 📊 Sector Aggregation Summary Table")
                summary_rows = []
                for s_name, s_data in sector_agg.items():
                    summary_rows.append({
                        "Sector": s_name,
                        "Live Events": s_data["event_count"],
                        "Latest Event (IST)": format_ist(s_data["latest_event_ts"]),
                        "Associated Symbols": ", ".join(s_data["companies"]) if s_data["companies"] else "None (General)",
                    })
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        # TAB 2: Live Disclosures Timeline
        with tab_live_timeline:
            scope_title = f"Company: **${selected_company}**" if has_company_filter else (f"Sector: **{selected_sector}**" if selected_sector != "All Sectors" else "All Sectors & Companies")
            st.markdown(f"### 📜 Live Disclosures & Filings Timeline ({scope_title})")

            filtered_items = all_live_items

            if has_company_filter:
                filtered_items = [it for it in filtered_items if selected_company in getattr(it, "tickers", [])]
            elif selected_sector != "All Sectors":
                filtered_items = [
                    it for it in filtered_items
                    if (selected_sector in [taxonomy.get_sector(t) for t in getattr(it, "tickers", [])])
                    or (selected_sector == "Market-wide / Unclassified" and not getattr(it, "tickers", []))
                ]

            if not filtered_items:
                st.info(f"No live records found for current scope ({scope_title}).")
            else:
                st.write(f"Displaying **{len(filtered_items)}** real records sorted by publication time.")
                sorted_items = sorted(
                    filtered_items,
                    key=lambda x: getattr(x, "published_at", None) or getattr(x, "timestamp", None) or datetime.min.replace(tzinfo=timezone.utc),
                    reverse=True,
                )

                for item in sorted_items[:50]:
                    pub_ist = format_ist(getattr(item, "published_at", None) or getattr(item, "timestamp", None))
                    provider = getattr(item, "source_provider", "exchange")
                    badge_color = "#10B981" if "nse" in provider else "#F59E0B"
                    prov_badge = "NSE OFFICIAL" if "nse" in provider else ("SEBI CIRCULAR" if "sebi" in provider else "OFFICIAL")
                    tickers_str = " ".join(f"`${t}`" for t in getattr(item, "tickers", [])) if getattr(item, "tickers", []) else "`[MARKET-WIDE]`"
                    url = getattr(item, "url", None)

                    with st.expander(f"📌 {pub_ist} | {prov_badge} | {item.headline[:75]}"):
                        st.markdown(f"**Headline:** {item.headline}")
                        st.markdown(f"**Identified Companies:** {tickers_str}")
                        st.markdown(f"**Source Provider:** `{provider}` | **Cognitive Provenance:** `{getattr(item, 'cognitive_provenance', '[RAW FEED]')}`")
                        if item.body:
                            st.markdown(f"**Summary / Description:** {item.body}")
                        if url:
                            st.markdown(f"🔗 **Original Document Link:** [{url}]({url})")

        # TAB 3: Social Chatter & Gemma 4 Analysis
        with tab_social_ingest:
            st.markdown("### 💬 Multimodal Social Ingestion & Gemma 4 Cognitive Analysis")
            st.write("Ingest community posts (via official X API or manual/file upload) and run structured extraction via local Gemma 4.")

            # Ingestion Sub-options
            ingest_col1, ingest_col2, ingest_col3 = st.columns([1, 1, 1])

            with ingest_col1:
                st.markdown("#### 1. Official X API Search")
                x_query = st.text_input("X Search Query", "Reliance OR Tata Motors OR Infosys rumor", help="Requires X_BEARER_TOKEN environment variable.")
                btn_fetch_x = st.button("📡 Fetch Recent Posts (X API)", use_container_width=True)

                if btn_fetch_x:
                    if not social_ingestor.is_x_api_configured():
                        st.warning("🔒 **X API Credentials Missing:** Set `X_BEARER_TOKEN` in your environment to fetch live X posts.")
                    else:
                        with st.spinner("Querying official X API v2 (max 20 posts)..."):
                            x_res = social_ingestor.fetch_x_recent_posts(query=x_query, max_results=20)
                            if x_res["success"]:
                                st.session_state.session_social_posts.extend(x_res["items"])
                                st.success(f"Successfully fetched {len(x_res['items'])} posts from X API v2!")
                                st.rerun()
                            else:
                                st.error(f"X API Error ({x_res['status']}): {x_res['error']}")

            with ingest_col2:
                st.markdown("#### 2. Live Reddit Stream (RSS)")
                st.caption("Live whispers from Indian equity subreddits without API keys.")
                selected_sub = st.selectbox("Subreddit", ["IndianStockMarket", "IndianStreetBets"], index=0)
                if st.button("💬 Fetch Live Reddit Posts", use_container_width=True):
                    with st.spinner(f"Ingesting live posts from r/{selected_sub}..."):
                        red_res = social_ingestor.fetch_reddit_posts(subreddit=selected_sub, max_posts=15)
                        if red_res["success"]:
                            st.session_state.session_social_posts.extend(red_res["items"])
                            st.success(f"✅ Ingested {len(red_res['items'])} live posts from r/{selected_sub}!")
                            st.rerun()
                        else:
                            st.error(f"Reddit Error: {red_res.get('error')}")

            with ingest_col3:
                st.markdown("#### 3. Manual Post Entry & Reel Upload")
                manual_text = st.text_area("Post Content / Whisper", placeholder="Rumor: Big merger between Reliance Retail and international partner...")
                manual_platform = st.selectbox("Platform", ["x_post", "instagram_reel", "reddit"], index=0)
                uploaded_media = st.file_uploader("Upload Image or Video (Optional)", type=["png", "jpg", "jpeg", "mp4"])

                if st.button("➕ Add Post to Timeline", use_container_width=True):
                    if not manual_text.strip():
                        st.warning("Please provide post text or whisper content.")
                    else:
                        with st.spinner("Processing post and extracting video keyframes..."):
                            saved_img_paths = []
                            saved_video_path = None
                            if uploaded_media:
                                save_dir = CACHE_DIR / "uploads"
                                save_dir.mkdir(parents=True, exist_ok=True)
                                target_p = save_dir / uploaded_media.name
                                with open(target_p, "wb") as f:
                                    f.write(uploaded_media.getbuffer())
                                if uploaded_media.name.lower().endswith(".mp4"):
                                    saved_video_path = str(target_p)
                                else:
                                    saved_img_paths.append(str(target_p))

                            try:
                                post = social_ingestor.create_user_post(
                                    text=manual_text,
                                    platform=manual_platform,
                                    image_paths=saved_img_paths,
                                    video_path=saved_video_path,
                                    source_label="USER_UPLOAD",
                                )
                                if post:
                                    st.session_state.session_social_posts.insert(0, post)
                                    st.success(f"✅ Added post ({post.post_id}) with {len(post.image_paths)} visual keyframes!")
                                    st.rerun()
                                else:
                                    st.error("Failed to construct validated post record.")
                            except Exception as e:
                                st.error(f"Error processing upload: {e}")

            st.markdown("---")
            st.markdown(f"#### 📜 Ingested Community Stream & Lifecycle Tracking ({len(st.session_state.session_social_posts)} posts)")

            if not st.session_state.session_social_posts:
                st.info("No social posts ingested yet. Click 'Fetch Live Reddit Posts', query X, or upload a Reel above.")
            else:
                for idx, p in enumerate(st.session_state.session_social_posts):
                    prov = getattr(p, "cognitive_provenance", "[USER_UPLOAD]")
                    pub_ist = format_ist(p.published_at or p.timestamp)
                    col_ist = format_ist(p.collected_at or p.timestamp)
                    tickers_tag = " ".join(f"`${t}`" for t in p.tickers) if p.tickers else "`[UNMATCHED / SPECULATIVE]`"

                    # Calculate Lifecycle Stage for this post's stock
                    primary_ticker = p.tickers[0] if p.tickers else None
                    if primary_ticker:
                        post_cutoff = p.published_at or p.timestamp
                        matching_news = [n for n in all_live_items if primary_ticker in getattr(n, "tickers", [])]
                        matching_posts = [sp for sp in st.session_state.session_social_posts if primary_ticker in sp.tickers]
                        life_res = classify_lifecycle_stage(
                            ticker=primary_ticker,
                            cutoff_time=post_cutoff,
                            social_posts=matching_posts,
                            news_items=matching_news,
                        )
                        stage_badge = f"STAGE {life_res.stage_index}: {life_res.stage.value.replace('_', ' ').upper()}"
                        stage_color = "#38BDF8" if life_res.stage_index in (1, 2) else ("#F59E0B" if life_res.stage_index == 3 else "#10B981")
                        lead_info = f" • Social Lead: {life_res.social_lead_hours:.1f}h" if life_res.social_lead_hours else " • Pre-News Whisper"
                    else:
                        stage_badge = "STAGE 1: EARLY WHISPER"
                        stage_color = "#94A3B8"
                        lead_info = " • Unverified Market Rumour"

                    st.markdown(f"""
                    <div class="hud-card" style="border-left: 4px solid {stage_color}; margin-bottom: 12px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <span class="hud-badge badge-cyan">{p.platform.value.upper()} • {prov}</span>
                                <span class="hud-badge" style="background:{stage_color}22; color:{stage_color}; border:1px solid {stage_color}; margin-left:8px;">{stage_badge}</span>
                            </div>
                            <span style="font-family:'JetBrains Mono',monospace; font-size:11px; color:#94A3B8;">Published: {pub_ist} {lead_info}</span>
                        </div>
                        <div style="font-size:14px; color:#F1F5F9; margin: 8px 0; font-weight:600;">{p.content_text}</div>
                        <div style="font-size:12px; color:#94A3B8;">
                            Identified Equities: {tickers_tag} | Author: <b>@{p.author.username}</b> | Provider: <code>{p.source_provider}</code>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if p.video_path or p.image_paths:
                        with st.expander(f"🎬 Media Details ({len(p.image_paths)} visual keyframes)"):
                            if p.video_path and os.path.exists(p.video_path):
                                st.video(p.video_path)
                            if p.image_paths:
                                img_cols = st.columns(min(len(p.image_paths), 3) or 1)
                                for c_idx, img_p in enumerate(p.image_paths):
                                    if os.path.exists(img_p):
                                        with img_cols[c_idx % 3]:
                                            st.image(img_p, caption=f"Keyframe {c_idx+1}", use_column_width=True)

                    # Action: Analyse with Gemma 4
                    btn_key = f"analyse_btn_{idx}_{p.post_id}"
                    if st.button(f"⚡ Analyse with Gemma 4", key=btn_key):
                        if not ollama_online:
                            st.error("🚨 **Gemma 4 Unavailable:** Local Ollama daemon is offline or model `gemma4` is not running. Launch Ollama (`ollama run gemma4`) to perform genuine inference.")
                        else:
                            with st.spinner(f"Running Gemma 4 inference on post {p.post_id}..."):
                                extractor = GemmaExtractor(force_mock=False)
                                res = extractor.extract_from_post(p)
                                if res.is_mock:
                                    st.warning(f"⚠️ Mock fallback used because Ollama failed: {res.error}")
                                else:
                                    st.session_state.session_gemma_analyses[p.post_id] = res.claim
                                    st.success("✅ Genuine Gemma 4 inference completed!")

                    # Display Analysis if present
                    if p.post_id in st.session_state.session_gemma_analyses:
                        claim = st.session_state.session_gemma_analyses[p.post_id]
                        st.markdown(f"""
                        <div style="background:rgba(2, 132, 199, 0.15); border:1px solid #38BDF8; border-radius:8px; padding:12px; margin-bottom:15px;">
                            <div style="font-size:11px; color:#38BDF8; font-family:'JetBrains Mono',monospace; font-weight:700;">[LOCAL GEMMA 4 INFERENCE]</div>
                            <div style="font-size:13px; font-weight:700; color:#FFFFFF; margin:4px 0;">Extracted Claim: {claim.claim}</div>
                            <div style="font-size:12px; color:#94A3B8;">Event: <b>{claim.event_category.value}</b> | Epistemic Status: <b>{claim.status.value}</b> (Uncertainty: {1.0 - claim.confidence:.2f})</div>
                            <div style="font-size:12px; color:#34D399; margin-top:2px;">Sentiment: <b>{claim.sentiment:+.2f}</b> | Intensity: <b>{claim.intensity:.2f}</b> | Spam Flag: <b>{claim.is_promotional_spam}</b></div>
                        </div>
                        """, unsafe_allow_html=True)

    else:
        # DEMO BENCHMARK MODE
        tab_sectors, tab_ticker, tab_multimodal, tab_signals = st.tabs([
            "🌐 Sector Lifecycle Matrix",
            "⚡ Ticker Narrative & Stepper",
            "🎬 Multimodal Reel & Post Inspector",
            "🎯 Signals & 6-Experiment Suite",
        ])

        active_ticker = selected_company if has_company_filter else (
            cfg.stock_universe[0] if cfg.stock_universe else "NVDA"
        )

        # TAB 1: Sector Lifecycle Matrix
        with tab_sectors:
            st.markdown("### 🏢 Benchmark Sector Taxonomy & Lifecycle Breakdown")
            st.write("Curated offline benchmark equities categorized by sector and their point-in-time phase in the **Rumour ➔ Hype ➔ Peak ➔ News ➔ Reaction** cycle.")

            sector_data = categorize_stocks_by_sector(bundle)
            display_sectors = sector_data if selected_sector == "All Sectors" else {
                selected_sector: sector_data.get(selected_sector, [])
            }

            for sec_name, stocks in display_sectors.items():
                sec_desc = SECTOR_DESCRIPTIONS.get(sec_name, "Industry sector equity group.")
                st.markdown(f"""
                <div style="background:rgba(30, 41, 59, 0.4); border-left: 4px solid #38BDF8; padding: 12px 16px; border-radius: 8px; margin: 15px 0 10px 0;">
                    <div style="font-size:16px; font-weight:800; color:#F1F5F9;">{sec_name}</div>
                    <div style="font-size:12px; color:#94A3B8;">{sec_desc}</div>
                </div>
                """, unsafe_allow_html=True)

                if stocks:
                    cols = st.columns(len(stocks))
                    for i, stock in enumerate(stocks):
                        with cols[i]:
                            stage_idx = stock["stage_index"]
                            badge_color = "badge-cyan" if stage_idx in (1, 2) else ("badge-amber" if stage_idx == 3 else ("badge-green" if stage_idx == 4 else "badge-red"))
                            border_color = "#38BDF8" if stage_idx in (1, 2) else ("#F59E0B" if stage_idx == 3 else ("#10B981" if stage_idx == 4 else "#EF4444"))

                            st.markdown(f"""
                            <div class="hud-card" style="border: 1px solid {border_color}; min-height: 220px;">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <span style="font-size:20px; font-weight:800; color:#FFFFFF;">${stock['ticker']}</span>
                                    <span class="hud-badge {badge_color}">STAGE {stage_idx}</span>
                                </div>
                                <div style="font-size:13px; font-weight:700; color:{border_color}; margin: 6px 0;">
                                    {stock['stage_name']}
                                </div>
                                <div style="font-size:11px; font-family:'JetBrains Mono',monospace; color:#94A3B8; margin-bottom:6px;">
                                    TYPE: <b style="color:#CBD5E1;">{stock['epistemic_badge']}</b>
                                </div>
                                <div style="font-size:12px; color:#CBD5E1; margin: 4px 0;">
                                    Attention: <b style="color:#38BDF8;">{stock['attention_multiplier']}x</b> ({stock['current_mentions']} posts)
                                </div>
                                <div style="font-size:11px; color:#10B981; margin-bottom:8px;">
                                    Lead: <b>{stock['social_lead_hours']}h ahead</b>
                                </div>
                                <div style="font-size:11px; padding:6px; background:rgba(0,0,0,0.3); border-radius:6px; color:#CBD5E1;">
                                    {stock['rationale']}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

            st.markdown("---")
            matrix_df = build_sector_lifecycle_table(bundle)
            if selected_sector != "All Sectors":
                matrix_df = matrix_df[matrix_df["Sector"] == selected_sector]
            st.dataframe(matrix_df, use_container_width=True)

        # TAB 2: Ticker Narrative & Stepper
        with tab_ticker:
            st.markdown(f"### 🧬 Narrative & Lifecycle Progression: **${active_ticker}** ({get_sector_for_ticker(active_ticker)})")
            posts = bundle.get_posts_for_ticker(active_ticker)
            news = bundle.get_news_for_ticker(active_ticker)
            bars = bundle.get_market_for_ticker(active_ticker)

            att = calculate_attention_metrics(posts, active_ticker)
            cutoff_ts = bars[-1].timestamp if bars else (posts[-1].timestamp if posts else datetime.now(timezone.utc))
            lifecycle = classify_lifecycle_stage(active_ticker, cutoff_ts, posts, news, attention_metrics=att)

            st.markdown(render_lifecycle_stepper_html(lifecycle.stage_index), unsafe_allow_html=True)

            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Abnormal Attention Spike", f"{att.attention_multiplier:.1f}x", delta=f"{att.current_mentions} current posts")
            lead_val = f"{lifecycle.social_lead_hours:.1f} hrs" if lifecycle.social_lead_hours else "Pending News"
            col_m2.metric("Social Rumour Lead Time", lead_val)
            col_m3.metric("Epistemic Certainty", "Confirmed PR" if lifecycle.has_official_confirmation else "Unverified Rumour")

            st.markdown("#### 📜 Interleaved Chronological Narrative Stream")
            events = format_timeline_events(posts, news)
            if events:
                for ev in events:
                    icon = "📰" if ev["is_official"] else "💬"
                    badge = "OFFICIAL NEWS WIRE" if ev["is_official"] else f"{ev['platform'].upper()} POST"
                    with st.expander(f"{icon} {ev['timestamp'][:16]} | {badge} | {ev['author']}"):
                        st.markdown(f"**Content:** {ev['summary']}")
            else:
                st.info(f"No narrative entries available for ${active_ticker}.")

        # TAB 3: Multimodal Inspector
        with tab_multimodal:
            st.markdown(f"### 🎬 Multimodal Community Inspector: **${active_ticker}**")
            posts_to_show = bundle.get_posts_for_ticker(active_ticker)
            for post in posts_to_show:
                card = format_multimodal_card(post, claim=None, is_mock=not ollama_online)
                st.markdown(f"""
                <div class="hud-card" style="border-left: 4px solid {'#10B981' if card['platform'] == 'instagram_reel' else '#38BDF8'}; margin-bottom:12px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="hud-badge {'badge-green' if card['platform'] == 'instagram_reel' else 'badge-cyan'}">{card['platform'].upper()}</span>
                        <span style="font-family:'JetBrains Mono',monospace; font-size:11px; color:#94A3B8;">{card['timestamp']}</span>
                    </div>
                    <div style="font-size:14px; font-weight:700; color:#F1F5F9; margin:8px 0;">{card['author']}</div>
                    <div style="font-size:13px; color:#CBD5E1;">{card['content_text']}</div>
                </div>
                """, unsafe_allow_html=True)

        # TAB 4: Signals & Experiments
        with tab_signals:
            st.markdown(f"### 🎯 Decision-Support Opportunity Signals: **${active_ticker}**")
            df = get_cached_dataset(horizon=1)
            predictor = MovementPredictor()
            predictor.fit(df)
            row = build_feature_row(active_ticker, cutoff_ts, bundle.social_posts, bundle.news_items, bundle.market_bars)
            pred = predictor.predict_row(row)
            card = format_signal_card(pred)
            badge_color = "#10B981" if card["badge_color"] == "green" else ("#EF4444" if card["badge_color"] == "red" else "#38BDF8")
            st.markdown(f"""
            <div class="hud-card" style="border: 2px solid {badge_color};">
                <span class="hud-badge" style="background:{badge_color}22; color:{badge_color};">SIGNAL: {card['signal_class'].upper()}</span>
                <span style="font-family:'JetBrains Mono',monospace; font-weight:800; color:{badge_color}; margin-left:12px;">PROBABILITY: {card['probability_outperform']}%</span>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
