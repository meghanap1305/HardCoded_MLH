# HypeLens: Multimodal Community Market Intelligence

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Track](https://img.shields.io/badge/Track_01-Best_Use_of_Gemma_4-orange.svg)](#)
[![Problem Statement](https://img.shields.io/badge/PS01-Multimodal_Community_Intelligence-green.svg)](#)

> **Empirical AI market-intelligence and signal-detection platform decoding social hype, tracking the rumour-to-news lifecycle, and investigating the classic market dynamic: *"Buy the rumour, sell the news."***

---

## 1. Project Purpose & Challenge Alignment

**HypeLens** participates in **Track 01 (PS01: Multimodal Community Intelligence)** for the Hacktoberfest Hack Day Bengaluru '26 (IEEE CIS / MLH). 

Retail investor communities actively discuss stocks across diverse media—from short-form video (Instagram Reels with on-screen chart leaks and spoken transcripts) to microblogs (X/Twitter posts with meme charts, cashtags, and engagement swarms). However, retail traders routinely buy at peak excitement and get caught in classic "sell the news" reversals.

HypeLens bridges this gap by making **Gemma 4** central to an explainable multimodal community intelligence pipeline:
- **Multimodal Video & Image Understanding:** Extracts explicit claims, event categories, epistemic certainty, and aspect sentiment from Reel keyframes, OCR text, and chart screenshots.
- **Rumour Validity Principle:** Rumours are treated as first-class analysis inputs; external verification is never required to analyse a post.
- **Independent Validation Streams:** Financial news and OHLCV price action are tracked as separate, asynchronous streams to measure cross-stream lead–lag and confirmation timing.
- **Leakage-Safe Movement Signals:** Models forward Cumulative Abnormal Returns (CAR) without look-ahead bias.

### Explicit Guardrails
- **No Investor Personalization:** No user risk profiling, portfolio allocation, or individualized advice.
- **No Automated Execution:** Strictly a research decision-support tool; no brokerage execution.
- **No Unsupported Predictions:** Outputs are probabilistic signals accompanied by risk warnings and historical analogues.

---

## 2. Architecture & The Rumour-to-News Lifecycle

HypeLens models the progression of community discourse across five distinct phases:

```
Stage 1: EARLY WHISPER   ──►  Stage 2: HYPE BUILD-UP   ──►  Stage 3: PEAK ATTENTION
(Low volume, unverified)       (Spike in creators/views)    (Mainstream retail FOMO)
                                                                    │
                                                                    ▼
Stage 5: REACTION / REVERSAL ◄──  Stage 4: OFFICIAL CONFIRMATION
(Sell-the-news vs. Continuation)  (Company filing / Major press wire)
```

### System Directory Layout
```text
hypelens/
├── config.py                 # Central config, Ollama settings & OpenCV checks
├── data/
│   ├── video.py              # Video keyframe extraction (opencv-python-headless)
│   ├── loader.py             # Multi-stream dataset loader & quarantine handler
│   └── fixtures/             # Local benchmark fixtures (social, news, market)
├── models/
│   └── data_models.py        # Strict UTC-normalized data models & schemas
├── nlp/
│   ├── prompts.py            # Gemma 4 prompt engineering templates
│   ├── parser.py             # Resilient JSON cleaner, clampers & claim schemas
│   └── extractor.py          # Multimodal Ollama client with mock fallback
├── analytics/
│   ├── attention_engine.py   # Rolling baselines, multipliers & bot pump detector
│   ├── lifecycle_classifier.py # 5-stage point-in-time lifecycle staging
│   └── analogues.py          # Nearest-neighbor historical setup retrieval
├── ml/
│   ├── feature_builder.py    # Leakage-safe feature builder & forward CAR target
│   ├── movement_model.py     # Baseline classifiers & opportunity signal cards
│   └── experiments.py        # Complete six-experiment evaluation suite (A-F)
└── ui/
    └── components.py         # Streamlit formatting helpers & research disclaimer

app.py                        # Interactive Streamlit dashboard application
tests/                        # Automated unit & end-to-end test suite
```

---

## 3. Installation & Environment Setup

### Prerequisites
- Python 3.11 or higher
- Git

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/your-username/hypelens.git
cd hypelens

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 4. Local Gemma 4 & Ollama Setup

HypeLens runs **100% locally** using Ollama without paid LLM cloud APIs.

### 1. Install & Launch Ollama
Download and install Ollama from [ollama.com](https://ollama.com/). Start the daemon:
```bash
ollama serve
```

### 2. Pull Gemma 4 Model
```bash
ollama pull gemma4
```

### 3. Environment Overrides (Optional)
Configure custom endpoints or models via environment variables:
```bash
# Optional overrides:
set OLLAMA_HOST=http://localhost:11434
set OLLAMA_MODEL=gemma4
set OLLAMA_TIMEOUT=30.0
set OLLAMA_FALLBACK=true
```

> **Offline Mock Fallback:** When Ollama is offline or unavailable, HypeLens automatically engages its deterministic offline mock extractor (`mock-gemma4`). All mock outputs are explicitly badged (`[DETERMINISTIC MOCK EXTRACTOR]`) and never presented as genuine model inference.

---

## 5. Launching the Dashboard

Launch the interactive multi-page Streamlit application:
```bash
streamlit run app.py
```

### Dashboard Features
1. **Market & Hype Overview:** Ticker leaderboard showing attention multipliers, influencer tier distributions, latest 1D return, and manipulation warnings.
2. **Rumour-to-News Timeline:** Unified chronological event stream tracking when social hype erupted relative to official disclosures.
3. **Multimodal Feed Inspector:** Detailed viewer for Instagram Reels (keyframes, OCR, transcripts) and X posts with user-triggered Gemma 4 claim extraction.
4. **Signals & 6-Experiment Suite:** Opportunity cards (`Pre-News Run-Up`, `Sell-the-News Reversal Risk`, `Continuation`), nearest-neighbor historical analogues, and benchmark evaluation tables.

---

## 6. Running the Automated Test Suite

HypeLens includes a complete unit and integration test suite designed to run entirely offline without requiring a running Ollama daemon or internet access:

```bash
# Run the complete test suite:
pytest

# Or run specific task suites:
pytest tests/test_config.py
pytest tests/test_data_models.py
pytest tests/test_video.py tests/test_data_loader.py
pytest tests/test_parser.py
pytest tests/test_extractor.py
pytest tests/test_attention_engine.py
pytest tests/test_lifecycle_classifier.py
pytest tests/test_ml_pipeline.py tests/test_experiments.py
pytest tests/test_app_components.py
pytest tests/test_e2e_pipeline.py
```

---

## 7. Research Foundation & Six Comparative Experiments

Built upon the financial sentiment analysis survey by **Du et al. (ACM Computing Surveys, 2024)**, HypeLens implements the six comparative ablation experiments:

| Experiment | Feature Scope | Research Objective |
|---|---|---|
| **Experiment A** | Market Only | Pure technical baseline (returns, abnormal volume, volatility). |
| **Experiment B** | Sentiment Only | Textual sentiment polarity and intensity without price data. |
| **Experiment C** | Market + Sentiment | Tests whether adding sentiment improves over market data alone. |
| **Experiment D** | Market + Sentiment + Attention | Isolates the predictive contribution of abnormal attention spikes. |
| **Experiment E** | Full Multi-Source + Lifecycle | Incorporates rumour vs. confirmed status and lifecycle staging. |
| **Experiment F** | Event Study | Evaluates pre-news run-ups vs. post-news reversal magnitude. |

---

## 8. Limitations & Open Empirical Decisions

In accordance with responsible AI guidelines:
1. **DEC-01 (Model Sizing):** The exact local Gemma 4 parameter size (2B vs 9B vs 27B) depends on available host GPU memory.
2. **DEC-02 (Hardware Constraints):** Local multimodal frame inference on CPU hardware is slower than GPU inference; offline mock mode is recommended for fast unit testing.
3. **DEC-03 (Data Licensing):** Scraping live Instagram or X content requires API compliance. The MVP uses curated benchmark fixtures under `hypelens/data/fixtures/`.
4. **DEC-04 (Clock Skew):** Social post timestamps must be normalized to UTC. Posts missing timestamps are quarantined to eliminate look-ahead leakage.
5. **Sample Size Notice:** On small fixture datasets, statistical power is limited; classifiers fall back to prior baselines when class variance is insufficient.

---

## 9. License & Citations

- **License:** Distributed under the [Apache-2.0 License](LICENSE).
- **Literature Reference:** Kelvin Du, Frank Xing, Rui Mao, Erik Cambria, *"Financial Sentiment Analysis: Techniques and Applications,"* ACM Computing Surveys, Vol. 56, No. 9, Article 220, 2024. DOI: [10.1145/3649451](https://doi.org/10.1145/3649451).
- **Notice on Reference Guidelines:** The official Hack Day guidelines PDF is retained under `docs/reference/` for developer consultation and excluded from redistribution in `.gitignore`.
