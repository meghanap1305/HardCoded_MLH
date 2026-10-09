# HypeLens: Project Task Roadmap

This roadmap lists tasks in strict dependency order. Unimplemented work is marked as `Pending`. Verification commands specify the exact test target for each task without implying that the test has already been run.

---

## Part 1: Required MVP Work

### Phase 1: Project Foundation & Environment Setup

#### TASK-01: Environment Configuration, Video Dependency & Ollama Setup
- **Status:** Completed
- **Dependencies:** None
- **Acceptance Criteria:**
  - Python project environment configured (`requirements.txt`).
  - Core dependencies pinned: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `requests`, `pillow`, `pytest`, and `opencv-python-headless` (chosen as the single video frame extraction library).
  - Setup validation script / helper verifying that `cv2` can be imported without GUI dependency errors.
  - Ollama client configuration helper with host URL, timeout settings, and fallback flag.
- **Expected Files:**
  - `requirements.txt`
  - `hypelens/config.py`
  - `tests/test_config.py`
- **Verification Command:**
  - `pytest tests/test_config.py`

---

### Phase 2: Data Schemas & Multimodal Test Fixtures

#### TASK-02: Multi-Stream Data Models & Validation Schemas
- **Status:** Completed
- **Dependencies:** TASK-01
- **Acceptance Criteria:**
  - Data models for Social Post (Instagram Reel keyframe metadata/OCR, X post, author tier, verification status, timestamp).
  - Data models for Financial News (headline, publisher, timestamp, body).
  - Data models for Market Data (OHLCV, benchmark return, volume).
  - Strict validation policy: quarantine or flag records missing timestamps; enforce UTC normalization.
- **Expected Files:**
  - `hypelens/models/data_models.py`
  - `tests/test_data_models.py`
- **Verification Command:**
  - `pytest tests/test_data_models.py`

#### TASK-03: Multimodal Benchmark Fixtures & Video Frame Extraction Helper
- **Status:** Completed
- **Dependencies:** TASK-02
- **Acceptance Criteria:**
  - Video keyframe extraction module (`hypelens/data/video.py`) using `opencv-python-headless` to extract timestamped sample frames from video clips.
  - Representative multimodal fixtures for at least 3 stocks covering each rumour-to-news lifecycle stage.
  - Includes Instagram Reel sample keyframe images, X posts with charts, matched news feeds, and chronological market CSVs.
  - Edge cases represented: unverified whispers, bot spam repetitions, contradictory claims, and quarantined records.
- **Expected Files:**
  - `hypelens/data/video.py`
  - `hypelens/data/loader.py`
  - `hypelens/data/fixtures/sample_social.jsonl`
  - `hypelens/data/fixtures/sample_news.jsonl`
  - `hypelens/data/fixtures/sample_market.csv`
  - `hypelens/data/fixtures/sample_images/`
  - `tests/test_video.py`
  - `tests/test_data_loader.py`
- **Verification Command:**
  - `pytest tests/test_video.py tests/test_data_loader.py`

---

### Phase 3: Gemma 4 Multimodal Extraction Engine

#### TASK-04: Gemma 4 Prompt Engineering & Structured JSON Parsing
- **Status:** Completed
- **Dependencies:** TASK-03
- **Acceptance Criteria:**
  - Structured prompt templates for local Gemma 4 instructing extraction of:
    1. Ticker / company entity.
    2. Event taxonomy (M&A, earnings, product, regulatory, speculation).
    3. Epistemic status (`Rumour/Unverified`, `Credible Report`, `Confirmed`, `Denied`).
    4. Aspect sentiment (polarity `[-1, 1]` + intensity `[0, 1]`).
    5. Manipulation flags (copy-paste spam, promotional hype).
  - Resilient JSON parser that repairs markdown wrappers, handles trailing commas, and returns validated schema objects.
- **Expected Files:**
  - `hypelens/nlp/prompts.py`
  - `hypelens/nlp/parser.py`
  - `tests/test_parser.py`
- **Verification Command:**
  - `pytest tests/test_parser.py`

#### TASK-05: Multimodal Extractor Client with Mock Fallback
- **Status:** Completed
- **Dependencies:** TASK-04
- **Acceptance Criteria:**
  - Multimodal extractor client interfacing with local Gemma 4 via Ollama REST API.
  - Accepts image inputs (Reel frames, chart screenshots) + text.
  - Deterministic offline mock extractor activated automatically during automated test suites or when Ollama is unreachable.
  - Graceful timeout, context length, and network error handling.
- **Expected Files:**
  - `hypelens/nlp/extractor.py`
  - `tests/test_extractor.py`
- **Verification Command:**
  - `pytest tests/test_extractor.py`

---

### Phase 4: Intelligence Engines, Six-Experiment Suite & Prediction

#### TASK-06: Attention Engine & Baseline Multiplier
- **Status:** Completed
- **Dependencies:** TASK-05
- **Acceptance Criteria:**
  - Calculates rolling baseline mention volume per stock.
  - Computes abnormal attention score ($A_t / \bar{A}_{\text{baseline}}$), mention acceleration, and author diversity.
  - Flags coordinated pump signatures (author-to-post ratio, repeated text hash clusters).
- **Expected Files:**
  - `hypelens/analytics/attention_engine.py`
  - `tests/test_attention_engine.py`
- **Verification Command:**
  - `pytest tests/test_attention_engine.py`

#### TASK-07: Rumour-to-News Lifecycle Classifier
- **Status:** Completed
- **Dependencies:** TASK-06
- **Acceptance Criteria:**
  - Maps each stock into 5 discrete lifecycle stages:
    1. Stage 1: Early Whisper
    2. Stage 2: Hype Build-up
    3. Stage 3: Peak Attention
    4. Stage 4: Official Confirmation
    5. Stage 5: Reaction / Reversal
  - Evaluates cross-stream lead–lag (e.g., social hype spike preceding official news disclosure by $k$ hours).
- **Expected Files:**
  - `hypelens/analytics/lifecycle_classifier.py`
  - `tests/test_lifecycle_classifier.py`
- **Verification Command:**
  - `pytest tests/test_lifecycle_classifier.py`

#### TASK-08: Leakage-Safe Feature Builder, Six-Experiment Suite & Movement Model
- **Status:** Completed
- **Dependencies:** TASK-07
- **Acceptance Criteria:**
  - Synthesizes temporal feature matrix enforcing strict cutoff timestamp $t$ (zero look-ahead leakage).
  - Target variable: Short-term Cumulative Abnormal Return (CAR) at $T+1$ and $T+3$ days.
  - Implements the complete six-experiment framework from the research foundation:
    - **Experiment A:** Market only (price, volume, volatility).
    - **Experiment B:** Sentiment only (social + news sentiment).
    - **Experiment C:** Market + sentiment.
    - **Experiment D:** Market + sentiment + abnormal attention.
    - **Experiment E:** Full multi-source with rumour-lifecycle features.
    - **Experiment F:** Event study around official announcements (pre-news run-up vs. post-news reversal).
  - **Hack Day Demo Priority:** Prioritize working implementations of **Experiment A (Market-only)** and **Experiment C/D (Market + Social feature comparison)** for the initial demonstration, with clean interface contracts for Experiments B, E, and F.
  - **Historical-Analogue Matching (Stretch Feature):** Module (`hypelens/analytics/analogues.py`) providing nearest-neighbor historical setup retrieval, flagged as data-dependent stretch functionality rather than a demo blocker.
- **Expected Files:**
  - `hypelens/ml/feature_builder.py`
  - `hypelens/ml/movement_model.py`
  - `hypelens/ml/experiments.py`
  - `hypelens/analytics/analogues.py`
  - `tests/test_ml_pipeline.py`
  - `tests/test_experiments.py`
- **Verification Command:**
  - `pytest tests/test_ml_pipeline.py tests/test_experiments.py`

---

### Phase 5: Streamlit Web Application & Open-Source Release

#### TASK-09: Streamlit Dashboard UI & Signal Explorer
- **Status:** Completed
- **Dependencies:** TASK-08
- **Acceptance Criteria:**
  - Interactive multi-page Streamlit dashboard:
    1. **Community Hype Leaderboard:** Top trending tickers, attention multipliers, sentiment acceleration.
    2. **Multimodal Feed Explorer:** Feed of Instagram Reels (keyframes/OCR) and X posts with Gemma 4 extracted JSON metadata.
    3. **Rumour-to-News Timeline:** Unified chronological chart plotting social hype spikes, news arrival, and market price movements.
    4. **Signal & Risk Cards:** Actionable research cards displaying opportunity classification, lifecycle stage, CAR outlook, and manipulation warnings.
    5. **Experiment Comparison View:** Side-by-side performance metrics of Experiment A (Market-only) vs. Experiment C/D (Market + Social signals).
- **Expected Files:**
  - `app.py`
  - `hypelens/ui/components.py`
  - `tests/test_app_components.py`
- **Verification Command:**
  - `pytest tests/test_app_components.py`

#### TASK-10: Open-Source Release, Documentation & E2E Verification
- **Status:** Completed
- **Dependencies:** TASK-09
- **Acceptance Criteria:**
  - Valid open-source license file (`LICENSE`, e.g., Apache 2.0 or MIT) added to repository root for MLH compliance.
  - Comprehensive `README.md` detailing architecture, Gemma 4 setup via Ollama, local execution instructions, and MLH Hack Day submission notes.
  - End-to-end integration test passes from sample dataset ingestion to signal generation and UI component verification.
- **Expected Files:**
  - `LICENSE`
  - `README.md`
  - `tests/test_e2e_pipeline.py`
- **Verification Command:**
  - `pytest tests/test_e2e_pipeline.py`

---

## Part 2: Staged Indian Data Integrations & Intelligence Platform Roadmap

### Phase 6: India Market Adaptation & Company Intelligence Taxonomy

#### TASK-11: Configurable Indian Company Taxonomy & Alias Resolver
- **Status:** Completed
- **Dependencies:** TASK-10
- **Acceptance Criteria:**
  - Define external JSON configuration (`config/companies_india.json`) containing top NSE-listed companies across sectors (e.g., RELIANCE, TATAMOTORS, INFY, HDFCBANK, TCS, ITC).
  - Include canonical ticker symbols (NSE/BSE), full legal corporate names, subsidiary brand aliases (e.g., "Jio", "Jaguar Land Rover"), prominent leadership handles, and industry sector mappings.
  - Implement an alias resolution module that normalizes colloquial text mentions, hashtags, and Cashtags (`$RELIANCE`, `#TataMotors`) to canonical NSE symbols without hardcoded logic.
  - Preserve backward compatibility for benchmark symbols (`NVDA`, `TSLA`, `AAPL`).
- **Expected Files:**
  - `config/companies_india.json`
  - `hypelens/models/taxonomy.py`
  - `tests/test_taxonomy.py`
- **Verification Command:**
  - `pytest tests/test_taxonomy.py`

#### TASK-12: Dual Timestamp & Provenance Schema Update
- **Status:** Completed
- **Dependencies:** TASK-11
- **Acceptance Criteria:**
  - Update data models to enforce dual timestamp protocol across all streams: `published_at` (source release time) and `collected_at` (system ingestion time).
  - Guarantee timezone-aware storage (strict UTC) and provide formatting utilities for display in Indian Standard Time (`Asia/Kolkata`, UTC+05:30).
  - Enrich records with explicit provenance metadata: source URL, publisher/author attribution, access tier, licensing constraints, and processing provenance (`[LOCAL GEMMA 4 INFERENCE]` vs. `[DETERMINISTIC MOCK EXTRACTOR]`).
  - Quarantine records with missing or ambiguous publication timestamps from leakage-sensitive analytics.
- **Expected Files:**
  - `hypelens/models/data_models.py`
  - `hypelens/utils/datetime_utils.py`
  - `tests/test_datetime_utils.py`
  - `tests/test_data_models.py`
- **Verification Command:**
  - `pytest tests/test_datetime_utils.py tests/test_data_models.py`

#### TASK-13: NSE/SEBI Corporate Filings & Indian Financial News RSS Ingestion Adapter
- **Status:** Completed
- **Dependencies:** TASK-12
- **Acceptance Criteria:**
  - Public RSS reader adapter for Indian financial news outlets (Economic Times, Livemint, Moneycontrol) parsing ISO-8601 timestamps, headlines, summaries, and canonical URLs.
  - Exchange disclosures adapter for structured NSE corporate announcements and SEBI regulatory circulars.
  - Built-in offline fallback: seamless switch to cached local JSONL fixtures if feeds are unreachable or blocked.
  - Rate limiting, error backoff, and strict no-scraping compliance for protected exchange portals.
- **Expected Files:**
  - `hypelens/ingestion/rss_news.py`
  - `hypelens/ingestion/nse_announcements.py`
  - `tests/test_indian_ingestion.py`
- **Verification Command:**
  - `pytest tests/test_indian_ingestion.py`

#### TASK-14: Indian Market Data Adapter & Multimodal Social Input Connector
- **Status:** Pending
- **Dependencies:** TASK-13
- **Acceptance Criteria:**
  - Historical and EOD market data adapter for NSE tickers (`.NS` format) supporting OHLCV bars and Nifty 50 benchmark returns.
  - Multimodal input connector supporting:
    1. Authorized local Instagram Reel video/frame uploads with keyframe extraction.
    2. Optional authenticated X API v2 connector with rate-limit tracking.
    3. Offline fallback to local synthetic/fixture social feeds.
  - Flag missing market days (e.g., Indian exchange holidays) gracefully without breaking CAR or feature matrices.
- **Expected Files:**
  - `hypelens/data/market_india.py`
  - `hypelens/ingestion/social_connector.py`
  - `tests/test_market_india.py`
- **Verification Command:**
  - `pytest tests/test_market_india.py`

#### TASK-15: Company Search & Unified Source-Linked Timeline Streamlit View
- **Status:** Pending
- **Dependencies:** TASK-14
- **Acceptance Criteria:**
  - Reposition Streamlit application with **Company Search & Unified Timeline** as the primary landing workflow:
    - Search bar auto-completing NSE companies, sectors, and aliases from `config/companies_india.json`.
    - Unified, chronological timeline rendering corporate filings, news headlines, and social posts side-by-side with `Asia/Kolkata` timestamps and source links.
    - Multimodal evidence drawer inspecting extracted Reel keyframes, OCR snippets, and Gemma 4 structured event JSON.
  - Present short-term stock movement prediction and experimental evaluation (Experiments A–F) in a dedicated **Optional Research & Lifecycle Analytics** tab, accompanied by clear non-financial-advice disclaimers and missing-data warnings.
- **Expected Files:**
  - `app.py`
  - `hypelens/ui/timeline_view.py`
  - `tests/test_timeline_view.py`
- **Verification Command:**
  - `pytest tests/test_timeline_view.py`

