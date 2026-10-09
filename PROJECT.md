# HypeLens: India-Focused Unified Company-Intelligence Timeline

**Track:** Track 01 — Best Use of Gemma 4  
**Problem Statement:** PS01 — Multimodal Community Intelligence  
**Hackathon:** Hacktoberfest Hack Day Bengaluru '26 (IEEE CIS / MLH)  
**Core Model:** Gemma 4 (via local Ollama inference)  
**Primary Stack:** Python 3.11+, Streamlit, Ollama, Pandas, Scikit-learn, OpenCV (`opencv-python-headless`)  
**Target Market:** Indian Capital Markets (NSE/BSE Listed Companies)  

---

## 1. Executive Summary & Repositioned Vision

**HypeLens** is an open-source, India-focused **unified company-intelligence timeline platform**. Built for Track 01 (PS01), HypeLens organizes fragmented public corporate information, financial news wires, and multimodal community discourse (Instagram Reels, X posts) into an explainable, source-attributed, chronological timeline for Indian equities.

### Primary Product Workflow
The primary user workflow is **Company Search & Unified Source-Linked Timeline Exploration**:
1. Search any NSE-listed company by name, symbol, or common alias (e.g., *Reliance*, *RELIANCE.NS*, *Tata Motors*, *TATAMOTORS*, *Infy*, *INFY*).
2. Explore an interactive, chronological timeline that unifies verified corporate disclosures (NSE/SEBI filings), reputable news headlines (Mint, Economic Times, Moneycontrol), and multimodal social chatter.
3. Inspect raw multimodal evidence (video keyframes, on-screen OCR, audio transcripts) with full provenance and source links.

### Secondary / Research Workflow (Optional Feature)
- **Short-term stock movement prediction is an optional research feature**, *not* the primary product promise.
- The platform retains the leakage-safe 5-stage rumour-to-news lifecycle classifier and the six comparative evaluation experiments (A–F from Du et al., 2024) to study whether abnormal community hype precedes news confirmation and price moves.
- **Explicit Guardrails:** No automated trade execution, no portfolio personalization, and no guarantees of future price performance.

---

## 2. Core Inputs, Timezone Discipline & Provenance Architecture

### A. Dual Timestamp Protocol (UTC Storage & Asia/Kolkata Display)
To prevent temporal ambiguity and look-ahead bias:
- **Storage:** All datetime fields are stored internally as strict timezone-aware UTC timestamps.
- **Presentation:** All UI components, timelines, and audit logs display timestamps in Indian Standard Time (**`Asia/Kolkata`**, UTC+05:30).
- **Dual Timing Fields:**
  - `published_at`: The exact point-in-time when the author, publisher, or exchange released the content.
  - `collected_at`: The system ingestion timestamp when HypeLens fetched or processed the record.
  - *Leakage guardrail:* Predictive feature synthesis strictly evaluates records by `published_at <= cutoff_time`.

### B. Configurable Indian Company Taxonomy
NSE-listed company symbols, ISINs, and linguistic/colloquial aliases are defined as **external configurable data** (e.g., `config/companies_india.json`), not hardcoded logic.
- Example:
  - Symbol: `RELIANCE` (NSE) / `RELIANCE.NS`
  - Canonical Name: *Reliance Industries Limited*
  - Aliases: `["RIL", "Jio", "Reliance Retail", "Mukesh Ambani", "$RELIANCE"]`
  - Sector: `Energy & Conglomerate`

### C. Ingestion Streams & Separation of Data Layers
HypeLens strictly segregates three tiers of data:
1. **Real Source Data:** Live or captured feeds with source URLs, author attribution, and platform metadata.
2. **Local Test Fixtures:** Offline deterministic JSONL/CSV datasets in `hypelens/data/fixtures/` allowing 100% offline development and test execution without internet access or live APIs.
3. **Mock vs. Real Model Outputs:** Extracted claims explicitly declare their cognitive provenance (`[LOCAL GEMMA 4 INFERENCE]` vs. `[DETERMINISTIC MOCK EXTRACTOR]`).

---

## 3. Central Role of Gemma 4 in Multimodal Timeline Intelligence

Gemma 4 provides local, zero-cloud-cost multimodal intelligence:
- **Video & Reel Keyframe OCR:** Reads financial charts, leaked documents, and tickers from Instagram Reels.
- **Transcript Reasoning:** Analyzes spoken audio transcripts for subtle rumor whispers vs. confirmed facts.
- **Aspect-Based Sentiment:** Decomposes narrative tone across revenue growth, margin pressures, and regulatory exposure `[-1.0, +1.0]`.
- **Claim Extraction:** Maps unstructured discourse into structured event taxonomies (Earnings, Regulatory/SEBI, M&A, Product Launch).

---

## 4. Staged Indian Data Integrations

HypeLens defines a progressive, staged integration path for Indian financial feeds:

| Stage | Integration Feed | Provider / Mechanism | Access & Licensing Policy |
|---|---|---|---|
| **Phase 1 (Active)** | Offline Curated Fixtures | Local JSONL/CSV/MP4 fixtures | Open-source Apache-2.0 benchmark files. |
| **Phase 2 (Staged)** | Indian Financial News Feeds | Public RSS (Livemint, Economic Times, Moneycontrol) | Public RSS syndication with full publisher attribution and canonical links. |
| **Phase 3 (Staged)** | Official Exchange Feeds | NSE Corporate Announcements / SEBI Filings feed | Public exchange disclosure portals; read-only headline and circular ingestion. |
| **Phase 4 (Staged)** | Indian Market Quotes | Yahoo Finance (`.NS`) / Local EOD CSV | Daily/hourly historical bars for abnormal volume and benchmark CAR. |
| **Phase 5 (Staged)** | Social Inputs | Optional X API v2 & Manual/Authorized Reel uploads | Compliant developer API or user-provided video files; no unauthorized scraping. |

*Offline Fallback Guarantee:* If external network feeds or APIs fail, the platform falls back seamlessly to cached local JSON/CSV files without interruptions.

---

## 5. Formal Unresolved Decisions & Risk Register

| ID | Category | Description | Status & Strategy |
|---|---|---|---|
| **DEC-01** | **Local Gemma 4 Sizing** | Selection of Gemma 4 model quantization (e.g. 2B vs 9B) based on host VRAM. | Configurable via `OLLAMA_MODEL`; offline mock fallback active in CI. |
| **DEC-02** | **NSE/SEBI Feed Access** | Official NSE website endpoints implement anti-scraping and IP blocks for programmatic scraping. | Rely on official public RSS feeds or authenticated financial data APIs rather than brittle web scraping. |
| **DEC-03** | **Social Media ToS & Scraping** | Live scraping of Instagram Reels and X without enterprise API keys violates Terms of Service. | Restrict live ingestion to authorized manual uploads or official API tokens; default to local benchmark fixtures. |
| **DEC-04** | **Timestamp Quality & Clock Skew** | News wire feeds and social posts may report imprecise publication times (e.g. "2 hours ago"). | Require ISO-8601 parsing; unresolvable timestamps are quarantined from leakage-sensitive evaluations. |
| **DEC-05** | **Company Alias Ambiguity** | Indian retail discourse frequently references prominent executives (e.g. "Ambani", "Adani", "Tata") or subsidiary brands (e.g. "Jio") rather than official ticker symbols. | Multi-tier alias matching with confidence weighting in entity extraction. |
| **DEC-06** | **Redistribution Restrictions** | Third-party PDF guidelines and proprietary corporate filings cannot be redistributed without permission. | Keep reference guidelines in `docs/reference/` and excluded from git release via `.gitignore`. |
