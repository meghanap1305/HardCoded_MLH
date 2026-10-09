"""Official NSE announcements and SEBI regulatory circulars ingestion adapters.

Ingests public exchange feeds and regulatory notifications using official RSS/XML
directory structures without fragile HTML scraping, enforcing offline caching,
deduplication, and fallback fixtures.
"""

from __future__ import annotations

import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional
import requests

from hypelens.config import FIXTURES_DIR, PROJECT_ROOT
from hypelens.models.data_models import FinancialNewsItem, validate_financial_news
from hypelens.models.taxonomy import get_taxonomy
from hypelens.utils.datetime_utils import now_utc, to_utc


CACHE_DIR = PROJECT_ROOT / "hypelens" / "data" / "cache"

# Official public RSS directory endpoints
OFFICIAL_FEEDS: Dict[str, str] = {
    "NSE": os.getenv(
        "NSE_RSS_URL",
        "https://nsearchives.nseindia.com/content/RSS/Online_announcements.xml"
    ),
    "NSE_CIRCULARS": os.getenv(
        "NSE_CIRCULARS_URL",
        "https://nsearchives.nseindia.com/content/RSS/Circulars.xml"
    ),
    "SEBI": os.getenv("SEBI_RSS_URL", "https://www.sebi.gov.in/sebirss.xml"),
}



class OfficialExchangeIngestor:
    """Ingests official NSE corporate announcements and SEBI circulars."""

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

    def _hash_record(self, headline: str, link: Optional[str]) -> str:
        base = f"{headline.strip().lower()}|{str(link).strip().lower()}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

    def parse_announcements_xml(
        self,
        xml_content: str,
        source: str = "NSE",
    ) -> List[FinancialNewsItem]:
        """Parse exchange or regulatory XML announcements into FinancialNewsItem records."""
        taxonomy = get_taxonomy()
        items: List[FinancialNewsItem] = []
        collected_at = now_utc()

        try:
            root = ET.fromstring(xml_content)
        except Exception:
            return items

        for item_node in root.findall(".//item"):
            title = item_node.findtext("title", "").strip()
            link = item_node.findtext("link", "").strip()
            desc = item_node.findtext("description", "").strip()
            pub_date = item_node.findtext("pubDate", "")
            guid = item_node.findtext("guid", "")

            if not title:
                continue

            sig = self._hash_record(title, link)
            if sig in self.seen_hashes:
                continue
            self.seen_hashes.add(sig)

            published_dt = to_utc(pub_date) if pub_date else collected_at
            news_id = f"official_{source.lower()}_{guid or sig[:12]}"

            # Detect company mentions using taxonomy
            combined_text = f"{title} {desc}"
            tickers = taxonomy.extract_symbols_from_text(combined_text)

            res = validate_financial_news({
                "news_id": news_id,
                "headline": title,
                "publisher": f"{source} Official Disclosures",
                "timestamp": published_dt.isoformat() if published_dt else None,
                "body": desc,
                "url": link or None,
                "tickers": tickers,
                "published_at": published_dt.isoformat() if published_dt else None,
                "collected_at": collected_at.isoformat(),
                "source_provider": f"official_{source.lower()}",
                "cognitive_provenance": f"[{source} OFFICIAL DISCLOSURE]",
            })

            if res.is_valid and res.record:
                items.append(res.record)

        return items

    def fetch_exchange_feed(self, source: str = "NSE") -> List[FinancialNewsItem]:
        """Fetch official feed with local caching and offline fallback."""
        url = OFFICIAL_FEEDS.get(source, OFFICIAL_FEEDS["NSE"])
        try:
            resp = requests.get(
                url,
                timeout=self.timeout,
                headers={"User-Agent": "HypeLens-Official-Disclosures/1.0"},
            )
            if resp.status_code == 200:
                items = self.parse_announcements_xml(resp.text, source=source)
                cache_file = self.cache_dir / f"official_{source.lower()}.jsonl"
                with open(cache_file, "w", encoding="utf-8") as f:
                    for it in items:
                        f.write(json.dumps(it.to_dict()) + "\n")
                return items
        except Exception:
            pass

        # Offline fallback
        if self.use_offline_fallback:
            fixture_name = "sample_nse_feed.xml" if source == "NSE" else "sample_sebi_feed.xml"
            fixture_path = FIXTURES_DIR / fixture_name
            if fixture_path.exists():
                with open(fixture_path, "r", encoding="utf-8") as f:
                    return self.parse_announcements_xml(f.read(), source=source)

        return []
