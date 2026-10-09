# Social Hype & Rumor-to-News Short-Term Stock Movement Intelligence System

## 1. Project Identity

This project is an AI-powered **market-intelligence and signal-detection platform** that captures social-media hype, financial-news flow, and company events around stocks, and tests whether these signals can identify **short-term stock movement opportunities**, especially around the classic **"buy the rumour, sell the news"** pattern.

The system has three connected goals:

1. **Hype capture:** detect which stocks are suddenly being talked about, by how much, and by whom, across social media and financial news.
2. **Rumour-to-news lifecycle tracking:** work out *where* a stock currently sits in the cycle of early rumour → rising hype → official confirmation → market reaction, and what has historically tended to happen at each stage.
3. **Short-term movement signals:** combine sentiment, attention, event, and market data to estimate the probability and direction of short-term price movement, and to flag possible entry/exit windows.

There is **no user profiling or personalization**. Every user sees the same market-level signals. The product is about the market, not the investor.

The project is a decision-support / research prototype. It must not claim to know or guarantee future stock prices.

---

# 2. Core Problem

Retail traders already follow social media, but they do it manually and badly:

- they see a stock trending only *after* the move has largely happened
- they cannot tell genuine information from bots, spam, and coordinated pumping
- they cannot tell a rumour from a confirmed announcement
- they buy at the peak of excitement and get caught by the "sell the news" reversal
- they have no historical evidence of how similar hype episodes ended

The gap this project targets:

> "People are talking about this stock" is not the same as
> "This stock is in an early, credible, not-yet-priced-in rumour phase with abnormal attention, and similar setups have historically moved X within Y days."

Price data alone does not capture this. Social discussion, news flow, event type, rumour/confirmation status, and abnormal attention may contain information about short-term movement, especially in the window *before* and *just after* official news.

---

# 3. Research Foundation

## Primary paper

**Title:** Financial Sentiment Analysis: Techniques and Applications  
**Authors:** Kelvin Du, Frank Xing, Rui Mao, Erik Cambria  
**Journal:** ACM Computing Surveys, Vol. 56, No. 9, Article 220, 2024  
**DOI:** 10.1145/3649451

### What we take from this paper

The paper is a survey of financial sentiment analysis (FSA). It strongly supports using financial text and social-media signals for forecasting.

Important ideas used by this project:

- financial sentiment is different from ordinary sentiment analysis
- financial sentiment can be derived from financial news, social media, company disclosures, and other financial text
- X/Twitter and StockTwits are established data sources in research
- financial sentiment can be used for stock-market movement prediction
- combining multiple sentiment sources is an emerging research direction
- sentiment intensity can be more informative than simple positive/negative classification
- social-media activity/attention can provide useful information in addition to sentiment
- time and sequence matter when connecting text with market reactions
- event-oriented and aspect-based sentiment analysis can improve interpretability
- market sentiment is not necessarily identical to the semantic sentiment of text
- combining textual sentiment with price, volume, volatility, technical, or other market features is important
- explainability is particularly important in financial applications

The paper also describes research using social-media text, news, event information, technical indicators, and temporal modeling for stock movement prediction.

### What the paper does NOT give us

- It does not establish "buy the rumour, sell the news" as a reliable rule. That is **our hypothesis to test**.
- It does not provide a ready-made rumour-vs-confirmed classifier or hype-manipulation detector. Those are **our proposed extensions**.
- Instagram and similar platforms are not core sources in the surveyed literature.

## Dropped from the previous version

The earlier version of this project was built around personalized investor recommendation (risk profiling, interest sectors, portfolio-aware ranking, and the ICESIC 2024 paper *Adaptable Individualized Investment Recommendation System*). That direction has been **removed**. User profiles, risk tiers, portfolio diversification, and personalized ranking are no longer part of the project.

---

# 4. Project Concept in One Sentence

> Build a system that detects abnormal social-media and news hype around stocks, classifies where each stock sits in the rumour-to-news lifecycle, and combines that with market data to estimate short-term movement and highlight potential pre-news run-up and post-news reversal opportunities.

---

# 5. High-Level Architecture

```text
 Social / Internet Text          Financial News              Company Events
         │                             │                           │
   X / Twitter                  Financial media             Announcements
   Reddit / StockTwits          Headlines / articles        Filings / reports
   Other public forums          Press releases              Earnings / deals
         │                             │                           │
         └─────────────────────────────┼───────────────────────────┘
                                       ▼
                         ┌───────────────────────────┐
                         │  Cleaning & Noise Filter  │
                         ├───────────────────────────┤
                         │ Dedup / spam / bot filter │
                         │ Ticker disambiguation     │
                         └─────────────┬─────────────┘
                                       ▼
                         ┌───────────────────────────┐
                         │   Financial NLP / FSA     │
                         ├───────────────────────────┤
                         │ Entity detection          │
                         │ Sentiment + intensity     │
                         │ Event extraction          │
                         │ Rumour vs confirmed       │
                         │ Aspect / topic            │
                         └─────────────┬─────────────┘
                                       ▼
                         ┌───────────────────────────┐
                         │   Hype / Attention Engine │
                         ├───────────────────────────┤
                         │ Mention volume & growth   │
                         │ Abnormal attention score  │
                         │ Unique authors / spread   │
                         │ Sentiment change          │
                         │ Manipulation-risk flags   │
                         └─────────────┬─────────────┘
                                       │
          ┌────────────────────────────┼───────────────────────────┐
          ▼                            ▼                           ▼
   Price / Returns                 Volume                  Volatility / Gaps
          └────────────────────────────┼───────────────────────────┘
                                       ▼
                       Rumour-to-News Lifecycle Stage Detector
                                       │
                                       ▼
                      Short-Term Movement Prediction Model
                                       │
                                       ▼
                    ┌─────────────────────────────────────┐
                    │ Opportunity Ranking & Alerts        │
                    │ (Pre-news run-up / Sell-the-news /  │
                    │  Continuation / Avoid)              │
                    └──────────────────┬──────────────────┘
                                       ▼
                           Explainable Signal Output
```

---

# 6. The Two Main Intelligence Layers

## A. Hype & Narrative Layer

This answers:

> **"What is the market talking about right now, and how unusual is it?"**

Possible inputs:

- X/Twitter posts
- Reddit / StockTwits / public stock forums
- financial news headlines and articles
- company announcements and filings
- market commentary
- other available public financial text

For every relevant text item, the system should attempt to identify:

- target company/stock
- event or topic
- sentiment polarity
- sentiment intensity
- rumour vs confirmed status
- source and source type
- timestamp
- confidence/relevance
- author information when available (account age, follower count, activity level)
- engagement/attention when available

## B. Market Reaction Layer

This answers:

> **"Is the market already moving on this, and how does the price/volume behaviour compare with the hype?"**

Inputs:

- price and returns (intraday and daily)
- volume and abnormal volume
- volatility
- price gaps
- technical indicators
- market and sector index returns

The most interesting situations are the **divergences**:

```text
High hype  + price hasn't moved yet   → possible early opportunity
High hype  + price already ran up      → possible late / crowded / reversal risk
Low hype   + price moving on volume    → possible informed or non-social move
Official news released after run-up    → classic sell-the-news setup
```

---

# 7. Financial Sentiment ≠ Normal Sentiment

Do not use a generic sentiment model blindly.

Financial language is domain-specific.

Examples:

- "debt" is not automatically negative in financial text
- "profit" can be positive or negative depending on direction/context
- sarcasm, memes, and slang ("to the moon", emojis) are common on social platforms and difficult
- multiple companies may be mentioned in one post
- the same text may be positive for one company/aspect and negative for another

Therefore:

> Prefer a finance-domain model such as FinBERT or another evaluated financial NLP model when practical, and consider slang/emoji handling for social text.

The exact model is an implementation decision and must be evaluated on the chosen dataset.

---

# 8. Sentiment & Hype Pipeline

```text
Raw post / news item
        ↓
Clean / normalize
        ↓
Spam, bot, duplicate filtering
        ↓
Detect company / ticker
        ↓
Check relevance
        ↓
Extract event / topic / aspect
        ↓
Classify: rumour / speculation / confirmed
        ↓
Financial sentiment model
        ↓
Polarity + intensity + confidence
        ↓
Timestamp + source metadata
        ↓
Aggregate by stock and time window
        ↓
Compare against stock's own historical baseline
```

Example aggregated result:

```text
Stock: COMPANY_X

1-hour window
-------------------------
Sentiment: +0.71
Intensity: 0.82
Mentions: 8,420
Change in mentions: +410%
Attention multiplier: 6.0x
Sentiment change: +0.29
News sentiment: +0.63
Social sentiment: +0.75
Rumour share of posts: 78%
Official confirmation: Not detected
```

---

# 9. Do NOT Use Only "Positive / Negative"

A stronger system should represent:

### Sentiment polarity

```text
-1 -------- 0 -------- +1
Negative   Neutral    Positive
```

### Sentiment intensity

How strongly the text expresses that sentiment.

### Sentiment change

```text
Previous window: +0.35
Current window:  +0.78
Change:          +0.43
```

### Attention / mention volume

```text
Normal:  1,000 mentions
Current: 9,000 mentions
```

This helps distinguish:

```text
High positive sentiment + normal attention
```

from:

```text
High positive sentiment + massive increase in attention
```

Those are not the same market situation. For this project, **attention spikes and sentiment acceleration are the primary signals**, not static sentiment level.

---

# 10. Multi-Source Aggregation

Do not simply concatenate all text and produce one sentiment number.

Preserve source-level signals.

```text
                    COMPANY_X
                        │
       ┌────────────────┼─────────────────┐
       ▼                ▼                 ▼
     Social            News           Company
    Sentiment        Sentiment         Events
       │                │                 │
     +0.76            +0.52        Positive/Neutral
       └────────────────┼─────────────────┘
                        ▼
               Aggregated Signal
```

A useful lead–lag question is **which source moves first**. Social hype that precedes news coverage is exactly the rumour-phase signature; news that arrives *after* social peak is a candidate sell-the-news trigger.

Possible aggregation approaches:

- weighted average
- source reliability weights
- time-decay weighting
- PCA/Kalman-style aggregation for different frequencies
- learned aggregation model

The first implementation should remain understandable and explainable.

---

# 11. Social Media Attention (Core Signal)

The system treats **attention** separately from sentiment, and as a first-class signal.

Useful features:

- number of mentions
- change in mention volume
- mention acceleration (rate of change of growth)
- engagement rate where available
- number of unique authors
- share of posts from high-activity or low-credibility accounts
- cross-platform spread (same stock trending on several platforms)
- sudden spikes relative to the stock's own historical baseline

Example:

```text
Mentions today: 15,000
7-day average:   2,500

Attention multiplier = 6.0x
```

Baselines should be **per stock** and ideally normalized for market-wide activity and time of day, since large, popular stocks always have more chatter than small ones.

---

# 12. "Buy the Rumour, Sell the News" — Central Hypothesis

This phrase is the central **market hypothesis the project investigates**. It is *not* a hard-coded trading rule.

## The lifecycle we model

```text
Stage 1  EARLY RUMOUR / WHISPER
         Low volume of discussion, few sources, unconfirmed
                ↓
Stage 2  HYPE BUILD-UP
         Mentions accelerate, sentiment shifts positive,
         more authors and platforms join, price begins to move
                ↓
Stage 3  PEAK ATTENTION
         Mainstream news picks it up, attention at maximum,
         price often already well above pre-rumour level
                ↓
Stage 4  OFFICIAL NEWS / CONFIRMATION
         Announcement, filing, or result is released
                ↓
Stage 5  MARKET REACTION
         Either: SELL THE NEWS  (reversal / profit-taking)
         Or:     CONTINUATION   (news better than expected)
         Or:     RUMOUR DENIED  (sharp drop / hype collapse)
```

## Research questions on this hypothesis

> Do early attention spikes and sentiment acceleration (Stage 1–2) predict positive short-term returns *before* official news?
>
> Does the probability of a reversal increase after official confirmation when the run-up was large and hype was already at peak?
>
> Can the system distinguish "priced-in" setups from "not-yet-priced-in" setups?

## Signal types we aim to detect

| Signal type | Description |
|---|---|
| **Pre-news run-up** | Rising credible hype and sentiment, no official confirmation yet, price has moved little |
| **Crowded / late hype** | Very high attention, price already extended, confirmation pending |
| **Sell-the-news** | Official news released after a large run-up, hype at or past peak |
| **Continuation** | Confirmation with strong surprise, attention still building |
| **Rumour collapse** | Denial or failure to confirm, hype decays rapidly |
| **Suspected manipulation** | Hype pattern consistent with coordinated promotion — avoid / flag |

Do not automatically implement:

```text
Positive sentiment → BUY
Official news      → SELL
```

The relationship must be **measured on historical data**, and results may show that the pattern is weak, inconsistent, or only present in particular event types, market caps, or regimes. That is a valid research outcome and must be reported honestly.

---

# 13. Event Detection & Rumour Classification

A key extension is detecting **what caused the hype** and **whether it is confirmed**.

Instead of:

```text
Stock X sentiment = +0.80
```

produce:

```text
Stock X

Detected event:
Partnership announcement

Status:
Rumour (unconfirmed)

Sentiment:
Positive

Attention:
+320% vs 7-day baseline

Market reaction so far:
+2.4%

Official confirmation:
Not detected
Expected catalyst date:
Unknown
```

Event categories:

- earnings/results
- product launch
- acquisition/merger
- partnership
- management change
- regulatory action / approval
- litigation
- dividend/buyback/stock split
- large order/deal
- index inclusion
- macroeconomic event
- sector event
- general rumour/speculation

Status categories:

- rumour / unverified
- credible report (reputable outlet, unconfirmed by company)
- officially confirmed
- denied / retracted

**Scheduled events** (earnings dates, regulatory decisions, product launches) are especially valuable because the "news" date is known in advance, making the rumour → news window measurable.

---

# 14. Aspect-Based Financial Sentiment

Example:

> "Company X's revenue growth is strong, but margins are under pressure."

Instead of classifying the whole sentence as positive or negative:

```text
Company X
├── Revenue growth → Positive
└── Margins        → Negative
```

This is more useful for explainability and for judging whether hype is about a real fundamental driver or purely speculative excitement. The survey identifies targeted and target-aspect financial sentiment analysis as important fine-grained tasks.

---

# 15. Short-Term Movement Prediction

The model should predict **movement / return / volatility**, not claim to know an exact future price.

Example targets:

```text
Next trading day:
UP / DOWN
```

or:

```text
Next 1-day return
Next 3-day return
Next 5-day return
```

or:

```text
Expected movement class:
Strong Up / Up / Neutral / Down / Strong Down
```

or, specific to this project:

```text
Return over the window around an official event:
  [pre-news window]   from hype detection → news release
  [post-news window]  news release → +1 / +3 / +5 days
```

Prefer **abnormal return** (stock return minus market/sector return) so that general market moves are not mistaken for hype-driven moves.

The target must be fixed before model training to avoid leakage.

---

# 16. Candidate Features

## Text / sentiment

- sentiment polarity
- sentiment intensity
- sentiment change / acceleration
- positive/negative/neutral ratios
- number of relevant texts
- source type
- emotion features if justified

## Attention / hype

- mention count
- mention growth rate and acceleration
- abnormal attention score (vs per-stock baseline)
- unique authors
- engagement
- cross-platform spread
- share of posts from new, low-follower, or highly repetitive accounts
- duplicate / copy-paste ratio

## Events / rumour lifecycle

- event category
- rumour vs confirmed status
- source credibility of first report
- event severity and novelty
- time since first mention
- time since peak attention
- time to / since official news
- days until scheduled catalyst

## Market

- recent returns and run-up since first rumour
- volume and abnormal volume
- volatility
- moving averages
- RSI or other technical features
- price gap
- market/index return
- sector/index return
- market cap and liquidity (small, illiquid stocks behave differently)

---

# 17. Model Design

Start simple.

### Baseline

```text
Market-only features
        ↓
Logistic Regression / Random Forest / XGBoost
        ↓
Movement prediction
```

### Sentiment model

```text
Market features
+
Sentiment features
        ↓
Prediction model
```

### Hype / multi-source model

```text
Market features
+
Social sentiment
+
News sentiment
+
Attention / hype features
+
Event & rumour-stage features
        ↓
Prediction model
```

Then compare the models.

The key research question becomes:

> **Does adding social hype, attention, and rumour-stage information improve short-term stock-movement prediction compared with market data alone?**

That is a much stronger experiment than simply demonstrating that a sentiment model can label text.

---

# 18. Evaluation

Evaluation must be time-aware.

Do NOT randomly shuffle a time series in a way that lets future information enter the training data.

Use chronological train/validation/test splits or rolling/expanding-window evaluation.

Example:

```text
2019-2022 → Training
2023      → Validation
2024      → Test
```

or rolling windows.

### Classification metrics

- Accuracy
- Precision
- Recall
- F1-score
- MCC (particularly useful when classes are imbalanced)

### Regression metrics

- MAE
- MSE
- RMSE
- R²

### Event-study metrics

- cumulative abnormal return (CAR) before and after official news
- average abnormal return by lifecycle stage
- reversal frequency after news following a large run-up

### Trading / strategy simulation

Only when the project actually evaluates a strategy. Must include realistic assumptions:

- transaction costs and slippage
- liquidity limits (especially for small caps)
- entry only at prices that would actually have been available after the signal timestamp

Metrics:

- cumulative return
- average return per trade
- Sharpe ratio
- maximum drawdown
- hit rate

The system should be compared against simple baselines (buy-and-hold, market-only model, "follow the most-mentioned stock" heuristic).

---

# 19. Critical Experimental Comparisons

At minimum, evaluate:

### Experiment A — Market only

```text
Price + volume + technical features
```

### Experiment B — Sentiment only

```text
Social/news sentiment
```

### Experiment C — Market + sentiment

```text
Market features + Sentiment features
```

### Experiment D — Market + sentiment + attention/hype

```text
Market + Sentiment + Attention features
```

### Experiment E — Full multi-source with rumour lifecycle

```text
Market + Social + News + Attention + Events + Rumour/confirmation stage
```

### Experiment F — "Buy the rumour, sell the news" event study

```text
Around official events:
  Does pre-news hype predict pre-news returns?
  Does post-news reversal depend on run-up size, hype level, and event type?
```

The goal is to show what each additional component contributes.

---

# 20. Opportunity Scoring Engine

After signals are generated, the system ranks stocks by short-term opportunity.

A candidate scoring framework:

```text
Opportunity Score =
    Abnormal Attention
  + Sentiment Acceleration
  + Rumour-Stage Signal
  + Not-Yet-Priced-In Estimate (run-up vs hype)
  + Liquidity / Tradability
  + Market Movement Model Probability
  − Manipulation / Noise Risk
```

The exact weights are implementation choices and must not be presented as universally correct.

The system should explain each contribution.

---

# 21. Example Final Signal UI

```text
--------------------------------------------------
                COMPANY X
--------------------------------------------------

Signal: PRE-NEWS RUN-UP CANDIDATE
Lifecycle stage: Hype build-up (Stage 2)

Market Intelligence
----------------------------
Social Sentiment:          +0.74 ↑ (accelerating)
News Sentiment:            +0.51
Attention:                 4.2x normal
Unique authors:            +280%
Event:                     Partnership rumour
Status:                    Unconfirmed
Official confirmation:     Not detected
First mention:             6 hours ago

Market Reaction
----------------------------
Price since first mention: +1.8%
Abnormal volume:           2.1x
Priced-in estimate:        Low-Moderate

Movement Model
----------------------------
Short-term outlook:        Positive
Confidence:                0.64

Why this appears
----------------------------
✓ Abnormal attention spike vs own baseline
✓ Sentiment accelerating across social and news
✓ Multiple independent sources
✓ Price has not yet fully reacted
✓ Similar past setups: avg +X% (n = N)

Risks / reasons for caution
----------------------------
! Rumour is not confirmed and may be denied
! Social-media signal can be noisy or promotional
! Sell-the-news reversal risk if confirmation arrives after a larger run-up
! Prediction is uncertain
```

This is a **decision-support explanation**, not a guarantee or promise of returns.

---

# 22. Explainability

Every signal should answer:

> Why is this stock being flagged, and what stage of the rumour-to-news cycle is it in?

Possible explanation dimensions:

- attention spike vs baseline
- sentiment trend and acceleration
- detected event and rumour/confirmed status
- source mix and credibility
- price/volume reaction so far
- lifecycle stage
- historical analogues (similar past setups and outcomes)
- manipulation-risk indicators
- model contribution

The survey emphasizes explainability as important in financial sentiment applications and discusses textual, visual, example-based, simplification, and feature-relevance explanations.

---

# 23. Data Requirements

The system needs synchronized data.

For each stock:

```text
timestamp
stock/company identifier
text source
text
author metadata
sentiment score
sentiment intensity
event/topic
rumour/confirmed status
engagement/attention
price
return
volume
volatility
market/sector data
official announcement timestamp (when applicable)
```

The most important design requirement is:

> **Every text signal must have a reliable timestamp that can be aligned with the market outcome being predicted. For the rumour-to-news analysis, the timestamp of the first mention and the timestamp of official confirmation must both be accurately known.**

---

# 24. Data Leakage Warning

This is one of the most important project rules, and it is even more critical for a timing-sensitive system.

Suppose we want to predict movement after a given signal time.

We may only use information that was available **before the prediction time**.

Bad:

```text
Use all news from the day
+
Tomorrow's closing price
```

if some of that "day's" news was published after the prediction point.

Also bad:

```text
Label a post as "rumour" using knowledge that the
news was later confirmed
```

The rumour/confirmed label at time *t* must only use information available at time *t*.

Good:

```text
Cutoff = prediction timestamp

Use only information available by the cutoff
        ↓
Predict future movement
```

This rule must be enforced in data preprocessing.

---

# 25. Social Media Noise & Manipulation Risk

Because this project deliberately chases hype, it is more exposed to noise and manipulation than a conventional sentiment system.

Social media can contain:

- bots
- spam
- duplicate posts
- promotional / paid content
- coordinated pump-and-dump activity
- sarcasm
- memes
- ambiguous ticker symbols
- low-quality opinions
- stale/reposted information

Therefore:

- preprocessing and relevance filtering are part of the model
- the system should include **manipulation-risk indicators** (e.g., many new accounts, near-identical text, narrow author set, thinly traded small-cap stock) and flag or down-weight such signals
- the project should avoid presenting flagged stocks as opportunities

Do not assume:

> "More tweets = more reliable signal."

Attention is a feature, not proof. A hype spike can be the *cause* of a price move, the *result* of one, or an artificial attempt to create one.

---

# 26. Platform Scope

Use only data sources that can be legally and technically accessed.

Possible sources:

- X/Twitter where API/data access permits
- Reddit / StockTwits / public forums where access permits
- financial news feeds/APIs
- company-released information and exchange announcements
- other public finance datasets

Instagram and similar platforms should be treated as an optional extension unless a suitable compliant data source is available.

The research survey strongly establishes X/Twitter, StockTwits, news, company reports, and similar sources.

---

# 27. Recommended Technical Stack

This is a project-level proposal; exact components can change.

### Backend

- Python
- FastAPI or Flask

### Data processing

- pandas
- NumPy
- scikit-learn

### NLP

- FinBERT or evaluated finance-domain transformer
- Hugging Face Transformers
- sentence embeddings for deduplication / near-duplicate detection and rumour clustering

### ML

- Logistic Regression
- Random Forest
- XGBoost
- gradient boosting
- anomaly / spike detection methods for attention
- neural/deep sequence models only if justified by results

### Streaming / scheduling (later stage)

- scheduled ingestion jobs first
- message queue / streaming only if live use requires it

### Database

- PostgreSQL / TimescaleDB or similar for time-series
- MongoDB for raw text if needed
- vector database only if semantic retrieval is needed

### Frontend

- React
- Vite
- Tailwind CSS

### Visualization

- Plotly / Recharts / Chart.js

---

# 28. Suggested System Modules

```text
backend/
├── market_data/
├── social_data/
├── news_data/
├── preprocessing/          (dedup, spam/bot filtering, ticker disambiguation)
├── sentiment/
├── entity_extraction/
├── event_detection/
├── rumour_classification/
├── attention_engine/       (baselines, spike detection, hype features)
├── lifecycle_detector/     (rumour → news stage logic)
├── manipulation_risk/
├── feature_engineering/
├── prediction/
├── opportunity_ranking/
├── explainability/
├── backtesting/
└── evaluation/

frontend/
├── dashboard/              (live hype leaderboard)
├── hype_monitor/
├── stock_details/          (timeline: rumour → news → reaction)
├── events_calendar/
├── signals/
└── history/                (past setups and outcomes)
```

---

# 29. Minimum Viable Product

Do not try to support every social platform and every ML method initially.

MVP:

```text
Stock universe
   ↓
X / Reddit / news historical dataset
   ↓
Cleaning + ticker matching + spam filtering
   ↓
Financial sentiment
   ↓
Attention / hype features (per-stock baseline)
   ↓
Event + rumour/confirmed tagging (limited categories)
   ↓
Market features
   ↓
Short-term up/down prediction
   ↓
Event study: pre-news vs post-news returns
   ↓
Opportunity ranking
   ↓
Explanation
```

Use historical/public datasets first.

Live APIs should be added only after the offline research pipeline works.

---

# 30. Strongest Research Question

> **Can social-media hype, attention spikes, and rumour-to-news lifecycle signals improve short-term stock movement prediction compared with market-only features, and can they identify when "buy the rumour, sell the news" patterns are likely to occur?**

---

# 31. Possible Sub-Research Questions

1. Does financial sentiment improve prediction over price/technical features alone?
2. Does combining social and news sentiment improve over a single source?
3. Does sentiment intensity or acceleration outperform simple positive/negative polarity?
4. Does abnormal attention contain additional predictive information beyond sentiment?
5. Does social hype lead news coverage, and does that lead time predict returns?
6. Do event-aware and rumour/confirmed features improve prediction?
7. Is there measurable run-up before official announcements, and a measurable reversal after?
8. Which event types, market caps, or liquidity levels show the strongest "sell the news" effect?
9. Can manipulation-risk features reduce false signals from coordinated or bot-driven hype?
10. Can explanations make model signals understandable?

---

# 32. What Counts as Innovation

Do NOT claim:

> "Nobody has ever used sentiment for stock prediction."

The supplied survey shows extensive prior research in this area.

Instead, position the project around **integration and system design**:

- real-time-style hype and abnormal-attention detection
- multi-source sentiment (social + news + events)
- rumour vs confirmed classification
- rumour-to-news lifecycle staging
- pre-news and post-news movement analysis
- manipulation-risk awareness
- temporal, leakage-safe evaluation
- explainable signals with historical analogues
- one integrated decision-support workflow

The strongest novelty claim should be based on what the actual implementation demonstrates.

---

# 33. What We Should NOT Claim

Avoid claims such as:

- "The AI can predict the stock market."
- "Our model knows which stock will rise."
- "Positive sentiment means the stock will rise."
- "Buy the rumour, sell the news always works."
- "High social attention means a stock is a good buy."
- "This system guarantees profit or lower risk."
- "Our score is the optimal trading strategy."
- "The system replaces a financial adviser."

Use language such as:

- "predictive signal"
- "estimated probability"
- "short-term movement classification"
- "decision-support"
- "hype / attention signal"
- "rumour-stage estimate"
- "historically associated"
- "empirically evaluated"

Short-term, hype-driven trading is high-risk. The product should include clear risk disclosures and should not be presented as financial advice.

---

# 34. Relationship to the Literature

## The survey contributes

```text
Financial text
 ↓
Financial sentiment
 ↓
Investor sentiment
 ↓
Market sentiment
 ↓
Stock-market forecasting
```

## Our project extends it

```text
Social + News + Events
        ↓
Cleaning, noise & manipulation filtering
        ↓
Sentiment + Attention + Rumour status
        ↓
Rumour-to-News Lifecycle Stage
        ↓
Short-Term Movement Prediction
        ↓
Opportunity Signals (run-up / sell-the-news / avoid)
        ↓
Explanation
```

---

# 35. Key Literature-Supported Insights

The survey is especially useful for these project choices:

### Financial text needs domain-aware NLP

Generic sentiment lexicons may misclassify financial terminology.

### Multi-source sentiment is important

Research increasingly combines different information sources for forecasting.

### Sentiment intensity matters

A continuous strength score can be richer than only positive/neutral/negative.

### Social media can contain predictive information

Research reviewed in the survey includes studies relating social-media sentiment and activity to future market movement.

### Event information matters

Event-driven stock movement prediction is an established research direction.

### Market data should be combined with text

The survey includes research combining sentiment with technical/market indicators.

### Explainability matters

Financial signals have significant consequences, so the system should make the reasons for its predictions understandable.

---

# 36. Final Project Positioning

### Project Name

**Social Hype & Rumour-to-News Stock Movement Intelligence System**

### One-line pitch

> An AI-powered market-intelligence platform that detects abnormal social-media and news hype, tracks each stock's rumour-to-news lifecycle, and combines sentiment, attention, event, and market signals to flag explainable short-term movement opportunities, including pre-news run-ups and sell-the-news risks.

### Core differentiator

> **It does not only ask "Is this stock trending?" — it asks "Is this hype early or late, credible or manipulated, confirmed or still a rumour, and what has historically happened next?"**

---

# 37. Source References

### Supplied Paper

Kelvin Du, Frank Xing, Rui Mao, Erik Cambria, "Financial Sentiment Analysis: Techniques and Applications," ACM Computing Surveys, Vol. 56, No. 9, Article 220, 2024.

DOI: https://doi.org/10.1145/3649451

Important sections from the supplied PDF:

- Section 2: FSA background and relationship between financial, investor, and market sentiment
- Section 4: FSA tasks, benchmark datasets, evaluation, and methods
- Section 5: FSA applications including financial forecasting and stock prediction
- Section 6: main findings and financial forecasting
- Section 7: challenges, future directions, and explainability

### Previously referenced (no longer a foundation)

N. Sabiyath Fatima et al., "Adaptable Individualized Investment Recommendation System," ICESIC 2024. DOI: https://doi.org/10.1109/ICESIC61777.2024.10846428

This paper supported the earlier personalization direction, which has been removed. It may be cited as related work only.

---

# 38. Development Rule

Before implementing a new feature, ask:

1. What user problem does it solve?
2. Which input data supports it?
3. What is the prediction target?
4. How is leakage avoided?
5. How is it evaluated?
6. How will the system explain the result?
7. Is the feature supported by the literature, or is it our proposed extension?
8. How does it behave under noisy or manipulated hype?

Do not add machine-learning complexity just to make the project look more advanced. Prefer a smaller, testable, explainable pipeline over a large collection of models.
