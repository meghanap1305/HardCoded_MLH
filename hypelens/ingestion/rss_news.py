"""RSS and GDELT news ingestion adapters for Indian financial news.

Ingests public RSS feeds (Livemint, Economic Times, Moneycontrol) and GDELT DOC API
metadata into normalized FinancialNewsItem schema records with deduplication,
local caching, and offline fallback.
"""

from __future__ import annotations

import hashlib
import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

from hypelens.config import FIXTURES_DIR, PROJECT_ROOT
from hypelens.models.data_models import FinancialNewsItem, validate_financial_news
from hypelens.models.taxonomy import get_taxonomy
from hypelens.utils.datetime_utils import now_utc, to_utc


CACHE_DIR = PROJECT_ROOT / "hypelens" / "data" / "cache"


# Configurable public RSS feed endpoints
INDIAN_NEWS_FEEDS: Dict[str, str] = {
    "Livemint": os.getenv("MINT_RSS_URL", "https://www.livemint.com/rss/markets"),
    "Economic Times": os.getenv("ET_RSS_URL", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    "Moneycontrol": os.getenv("MONEYCONTROL_RSS_URL", "https://www.moneycontrol.com/rss/MCtopnews.xml"),
}


class RSSNewsIngestor:
    """Ingests Indian financial news from RSS feeds with caching and deduplication."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        timeout: float = 5.0,
        use_offline_fallback: bool = True,
    ) -> None:
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.use_offline_fallback = use_offline_fallback
        self.seen_hashes: set[str] = set()

    def _hash_record(self, headline: str, url: Optional[str]) -> str:
        """Create unique signature for deduplication."""
        base = f"{headline.strip().lower()}|{str(url).strip().lower()}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

    def parse_rss_xml(
        self,
        xml_content: str,
        publisher: str,
        source_provider: str = "news_rss",
    ) -> List[FinancialNewsItem]:
        """Parse RSS XML content into normalized FinancialNewsItem records."""
        taxonomy = get_taxonomy()
        items: List[FinancialNewsItem] = []
        collected_at = now_utc()

        try:
            root = ET.fromstring(xml_content)
        except Exception:
            return items

        # Look for <item> nodes in RSS channel
        for item_node in root.findall(".//item"):
            title = item_node.findtext("title", "").strip()
            link = item_node.findtext("link", "").strip()
            description = item_node.findtext("description", "").strip()
            pub_date = item_node.findtext("pubDate", "")
            guid = item_node.findtext("guid", "")

            if not title:
                continue

            # Deduplication
            sig = self._hash_record(title, link)
            if sig in self.seen_hashes:
                continue
            self.seen_hashes.add(sig)

            published_dt = to_utc(pub_date) if pub_date else collected_at
            news_id = f"news_rss_{guid or sig[:12]}"

            # Detect company mentions using taxonomy
            text_for_matching = f"{title} {description}"
            tickers = taxonomy.extract_symbols_from_text(text_for_matching)

            res = validate_financial_news({
                "news_id": news_id,
                "headline": title,
                "publisher": publisher,
                "timestamp": published_dt.isoformat() if published_dt else None,
                "body": description,
                "url": link or None,
                "tickers": tickers,
                "published_at": published_dt.isoformat() if published_dt else None,
                "collected_at": collected_at.isoformat(),
                "source_provider": source_provider,
                "cognitive_provenance": "[FEED RSS INGESTION]",
            })

            if res.is_valid and res.record:
                items.append(res.record)

        return items

    def fetch_feed(self, publisher: str, url: str) -> List[FinancialNewsItem]:
        """Fetch RSS feed over HTTP with offline fixture fallback."""
        try:
            resp = requests.get(
                url,
                timeout=self.timeout,
                headers={"User-Agent": "HypeLens-Research-Bot/1.0"},
            )
            if resp.status_code == 200:
                items = self.parse_rss_xml(resp.text, publisher=publisher)
                # Cache results locally
                cache_file = self.cache_dir / f"rss_{publisher.lower().replace(' ', '_')}.jsonl"
                self._save_cache(items, cache_file)
                return items
        except Exception:
            pass

        # Offline fallback
        if self.use_offline_fallback:
            return self.load_fixture_or_cache(publisher)

        return []

    def load_fixture_or_cache(self, publisher: str) -> List[FinancialNewsItem]:
        """Load from local cache or fallback fixtures."""
        cache_file = self.cache_dir / f"rss_{publisher.lower().replace(' ', '_')}.jsonl"
        if cache_file.exists():
            return self._load_cache(cache_file)

        # Fallback to local sample fixture
        fixture_file = FIXTURES_DIR / "sample_news_rss.xml"
        if fixture_file.exists():
            with open(fixture_file, "r", encoding="utf-8") as f:
                return self.parse_rss_xml(f.read(), publisher=publisher)

        return []

    def _save_cache(self, items: List[FinancialNewsItem], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item.to_dict()) + "\n")

    def _load_cache(self, path: Path) -> List[FinancialNewsItem]:
        items: List[FinancialNewsItem] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    res = validate_financial_news(data)
                    if res.is_valid and res.record:
                        items.append(res.record)
        return items


class GDELTDocIngestor:
    """Configurable ingestion adapter for the public GDELT DOC 2.0 API.

    Queries news metadata concerning Indian companies with conservative request intervals,
    local caching, and offline fallback.
    """

    GDELT_DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        timeout: float = 6.0,
        use_offline_fallback: bool = True,
    ) -> None:
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.use_offline_fallback = use_offline_fallback
        self.seen_hashes: set[str] = set()

    def parse_gdelt_json(self, data: Dict[str, Any]) -> List[FinancialNewsItem]:
        """Parse GDELT JSON articles array into normalized FinancialNewsItem records."""
        taxonomy = get_taxonomy()
        items: List[FinancialNewsItem] = []
        articles = data.get("articles", [])
        collected_at = now_utc()

        for art in articles:
            title = art.get("title", "").strip()
            url = art.get("url", "").strip()
            domain = art.get("domain", "GDELT News Wire")
            seendate = art.get("seendate", "")

            if not title:
                continue

            # Deduplication
            sig = hashlib.sha256(f"{title}|{url}".encode("utf-8")).hexdigest()
            if sig in self.seen_hashes:
                continue
            self.seen_hashes.add(sig)

            # GDELT date format is typically YYYYMMDDTHHMMSSZ or YYYYMMDDHHMMSS
            published_dt = to_utc(seendate) or collected_at
            tickers = taxonomy.extract_symbols_from_text(title)

            res = validate_financial_news({
                "news_id": f"gdelt_{sig[:12]}",
                "headline": title,
                "publisher": domain,
                "timestamp": published_dt.isoformat() if published_dt else None,
                "body": f"Article indexed from {domain} via GDELT DOC API.",
                "url": url,
                "tickers": tickers,
                "published_at": published_dt.isoformat() if published_dt else None,
                "collected_at": collected_at.isoformat(),
                "source_provider": "gdelt_doc_api",
                "cognitive_provenance": "[GDELT DOC API METADATA]",
            })

            if res.is_valid and res.record:
                items.append(res.record)

        return items

    def fetch_articles(self, query: str = "India business") -> List[FinancialNewsItem]:
        """Fetch GDELT metadata with query and offline fallback."""
        params = {
            "query": query,
            "mode": "artlist",
            "maxrecords": "25",
            "format": "json",
        }
        try:
            resp = requests.get(self.GDELT_DOC_API_URL, params=params, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                items = self.parse_gdelt_json(data)
                # Cache results locally
                cache_file = self.cache_dir / "gdelt_cache.jsonl"
                with open(cache_file, "w", encoding="utf-8") as f:
                    for item in items:
                        f.write(json.dumps(item.to_dict()) + "\n")
                return items
        except Exception:
            pass

        # Offline fallback
        if self.use_offline_fallback:
            fixture_file = FIXTURES_DIR / "sample_gdelt_feed.json"
            if fixture_file.exists():
                with open(fixture_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return self.parse_gdelt_json(data)

        return []
