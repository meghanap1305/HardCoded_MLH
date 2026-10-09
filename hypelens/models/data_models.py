"""Multi-stream data models and validation schemas for HypeLens.

Enforces strict timestamp UTC normalization, quarantine policies for records
missing valid timestamps, author influencer tiering, and independent stream modeling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar


class PlatformType(str, Enum):
    """Supported community and social platforms."""
    INSTAGRAM_REEL = "instagram_reel"
    X_POST = "x_post"
    REDDIT = "reddit"
    OTHER = "other"


class AccountTier(str, Enum):
    """Influencer and creator scale tiers."""
    SMALL = "small"          # Retail creators (< 10k followers)
    MEDIUM = "medium"        # Niche creators (10k - 100k followers)
    LARGE = "large"          # Macro finfluencers (> 100k followers)
    UNKNOWN = "unknown"


@dataclass
class AuthorMetadata:
    """Metadata regarding post author credibility and reach."""
    author_id: str
    username: str
    tier: AccountTier = AccountTier.UNKNOWN
    is_official: bool = False       # Verified / corporate / regulatory account
    follower_count: Optional[int] = None
    account_created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "author_id": self.author_id,
            "username": self.username,
            "tier": self.tier.value,
            "is_official": self.is_official,
            "follower_count": self.follower_count,
            "account_created_at": (
                self.account_created_at.isoformat()
                if self.account_created_at
                else None
            ),
        }


@dataclass
class SocialPost:
    """Multimodal community post (Instagram Reel or X post)."""
    post_id: str
    platform: PlatformType
    timestamp: datetime            # Normalized UTC timestamp (published_at)
    author: AuthorMetadata
    content_text: str
    tickers: List[str] = field(default_factory=list)
    image_paths: List[str] = field(default_factory=list)
    video_path: Optional[str] = None
    ocr_text: Optional[str] = None
    transcript: Optional[str] = None
    engagement: Dict[str, int] = field(default_factory=dict)  # likes, reposts, comments, views
    raw_payload: Dict[str, Any] = field(default_factory=dict) # Provenance preservation
    published_at: Optional[datetime] = None                    # Point-in-time source release
    collected_at: Optional[datetime] = None                    # Point-in-time system ingestion
    source_provider: str = "community"                        # Provider identifier
    cognitive_provenance: str = "[RAW FEED]"                  # Inference status

    def __post_init__(self) -> None:
        if self.published_at is None:
            self.published_at = self.timestamp
        if self.collected_at is None:
            self.collected_at = self.timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "post_id": self.post_id,
            "platform": self.platform.value,
            "timestamp": self.timestamp.isoformat(),
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "collected_at": self.collected_at.isoformat() if self.collected_at else None,
            "source_provider": self.source_provider,
            "cognitive_provenance": self.cognitive_provenance,
            "author": self.author.to_dict(),
            "content_text": self.content_text,
            "tickers": self.tickers,
            "image_paths": self.image_paths,
            "video_path": self.video_path,
            "ocr_text": self.ocr_text,
            "transcript": self.transcript,
            "engagement": self.engagement,
        }


@dataclass
class FinancialNewsItem:
    """Verified financial news article or regulatory disclosure."""
    news_id: str
    headline: str
    publisher: str
    timestamp: datetime            # Normalized UTC timestamp (published_at)
    body: str = ""
    url: Optional[str] = None
    tickers: List[str] = field(default_factory=list)
    raw_payload: Dict[str, Any] = field(default_factory=dict)
    published_at: Optional[datetime] = None                    # Point-in-time release
    collected_at: Optional[datetime] = None                    # Point-in-time system ingestion
    source_provider: str = "news_wire"                        # e.g., nse, sebi, rss, gdelt
    cognitive_provenance: str = "[RAW FEED]"

    def __post_init__(self) -> None:
        if self.published_at is None:
            self.published_at = self.timestamp
        if self.collected_at is None:
            self.collected_at = self.timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "news_id": self.news_id,
            "headline": self.headline,
            "publisher": self.publisher,
            "timestamp": self.timestamp.isoformat(),
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "collected_at": self.collected_at.isoformat() if self.collected_at else None,
            "source_provider": self.source_provider,
            "cognitive_provenance": self.cognitive_provenance,
            "body": self.body,
            "url": self.url,
            "tickers": self.tickers,
        }



@dataclass
class MarketBar:
    """Point-in-time equity market price bar."""
    ticker: str
    timestamp: datetime            # Normalized UTC timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    benchmark_return: float = 0.0  # Associated market benchmark return for abnormal calculations
    abnormal_volume: float = 1.0   # Ratio against rolling baseline

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "benchmark_return": self.benchmark_return,
            "abnormal_volume": self.abnormal_volume,
        }


T = TypeVar("T")


@dataclass
class ValidationResult(Generic[T]):
    """Validation envelope indicating validity, parsed record, or quarantine status."""
    is_valid: bool
    record: Optional[T] = None
    errors: List[str] = field(default_factory=list)
    quarantined: bool = False


def parse_utc_timestamp(val: Any) -> Optional[datetime]:
    """Parse various timestamp representations into a timezone-aware UTC datetime."""
    from hypelens.utils.datetime_utils import to_utc
    return to_utc(val)


def validate_social_post(raw: Dict[str, Any]) -> ValidationResult[SocialPost]:
    """Validate and construct a SocialPost from raw input dictionary."""
    errors: List[str] = []

    post_id = str(raw.get("post_id", "")).strip()
    if not post_id:
        errors.append("Missing required field 'post_id'.")

    # Timestamp verification
    dt = parse_utc_timestamp(raw.get("timestamp"))
    if dt is None:
        return ValidationResult(
            is_valid=False,
            quarantined=True,
            errors=["Missing or unparseable timestamp; post quarantined to avoid temporal leakage."],
        )

    # Platform normalization
    raw_plat = str(raw.get("platform", "other")).lower()
    try:
        platform = PlatformType(raw_plat)
    except ValueError:
        platform = PlatformType.OTHER

    # Author metadata
    author_raw = raw.get("author", {})
    if not isinstance(author_raw, dict):
        author_raw = {"username": str(author_raw)}

    author_id = str(author_raw.get("author_id", author_raw.get("username", "unknown"))).strip()
    username = str(author_raw.get("username", author_id)).strip()
    raw_tier = str(author_raw.get("tier", "unknown")).lower()
    try:
        tier = AccountTier(raw_tier)
    except ValueError:
        tier = AccountTier.UNKNOWN

    author = AuthorMetadata(
        author_id=author_id,
        username=username,
        tier=tier,
        is_official=bool(author_raw.get("is_official", False)),
        follower_count=author_raw.get("follower_count"),
        account_created_at=parse_utc_timestamp(author_raw.get("account_created_at")),
    )

    content_text = str(raw.get("content_text", raw.get("text", "")))
    tickers = [str(t).upper().replace("$", "") for t in raw.get("tickers", [])]

    image_paths = [str(p) for p in raw.get("image_paths", []) if p]
    video_path = str(raw["video_path"]) if raw.get("video_path") else None
    ocr_text = str(raw["ocr_text"]) if raw.get("ocr_text") else None
    transcript = str(raw["transcript"]) if raw.get("transcript") else None

    engagement = raw.get("engagement", {})
    if not isinstance(engagement, dict):
        engagement = {}

    if errors:
        return ValidationResult(is_valid=False, quarantined=False, errors=errors)

    pub_dt = parse_utc_timestamp(raw.get("published_at")) or dt
    col_dt = parse_utc_timestamp(raw.get("collected_at")) or dt
    source_prov = str(raw.get("source_provider", "community"))
    cog_prov = str(raw.get("cognitive_provenance", "[RAW FEED]"))

    post = SocialPost(
        post_id=post_id,
        platform=platform,
        timestamp=dt,
        author=author,
        content_text=content_text,
        tickers=tickers,
        image_paths=image_paths,
        video_path=video_path,
        ocr_text=ocr_text,
        transcript=transcript,
        engagement={k: int(v) for k, v in engagement.items() if str(v).isdigit()},
        raw_payload=raw,
        published_at=pub_dt,
        collected_at=col_dt,
        source_provider=source_prov,
        cognitive_provenance=cog_prov,
    )
    return ValidationResult(is_valid=True, record=post)



def validate_financial_news(raw: Dict[str, Any]) -> ValidationResult[FinancialNewsItem]:
    """Validate and construct a FinancialNewsItem from raw input dictionary."""
    errors: List[str] = []

    news_id = str(raw.get("news_id", "")).strip()
    if not news_id:
        errors.append("Missing required field 'news_id'.")

    headline = str(raw.get("headline", "")).strip()
    if not headline:
        errors.append("Missing required field 'headline'.")

    publisher = str(raw.get("publisher", "")).strip()
    if not publisher:
        errors.append("Missing required field 'publisher'.")

    dt = parse_utc_timestamp(raw.get("timestamp"))
    if dt is None:
        return ValidationResult(
            is_valid=False,
            quarantined=True,
            errors=["Missing or unparseable timestamp; news item quarantined."],
        )

    if errors:
        return ValidationResult(is_valid=False, quarantined=False, errors=errors)

    pub_dt = parse_utc_timestamp(raw.get("published_at")) or dt
    col_dt = parse_utc_timestamp(raw.get("collected_at")) or dt
    source_prov = str(raw.get("source_provider", "news_wire"))
    cog_prov = str(raw.get("cognitive_provenance", "[RAW FEED]"))

    item = FinancialNewsItem(
        news_id=news_id,
        headline=headline,
        publisher=publisher,
        timestamp=dt,
        body=str(raw.get("body", "")),
        url=raw.get("url"),
        tickers=[str(t).upper().replace("$", "") for t in raw.get("tickers", [])],
        raw_payload=raw,
        published_at=pub_dt,
        collected_at=col_dt,
        source_provider=source_prov,
        cognitive_provenance=cog_prov,
    )
    return ValidationResult(is_valid=True, record=item)



def validate_market_bar(raw: Dict[str, Any]) -> ValidationResult[MarketBar]:
    """Validate and construct a MarketBar from raw input dictionary."""
    errors: List[str] = []

    ticker = str(raw.get("ticker", "")).upper().replace("$", "").strip()
    if not ticker:
        errors.append("Missing required field 'ticker'.")

    dt = parse_utc_timestamp(raw.get("timestamp"))
    if dt is None:
        return ValidationResult(
            is_valid=False,
            quarantined=True,
            errors=["Missing or unparseable timestamp; market bar quarantined."],
        )

    try:
        o = float(raw["open"])
        h = float(raw["high"])
        l = float(raw["low"])
        c = float(raw["close"])
        v = float(raw["volume"])
    except (KeyError, ValueError, TypeError) as e:
        return ValidationResult(
            is_valid=False,
            quarantined=False,
            errors=[f"Invalid or missing OHLCV numerical values: {e}"],
        )

    benchmark_ret = float(raw.get("benchmark_return", 0.0))
    abnormal_vol = float(raw.get("abnormal_volume", 1.0))

    bar = MarketBar(
        ticker=ticker,
        timestamp=dt,
        open=o,
        high=h,
        low=l,
        close=c,
        volume=v,
        benchmark_return=benchmark_ret,
        abnormal_volume=abnormal_vol,
    )
    return ValidationResult(is_valid=True, record=bar)
