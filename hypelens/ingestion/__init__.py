"""Ingestion module for HypeLens.

Provides adapters for official exchange announcements (NSE, SEBI),
financial news RSS feeds (Livemint, Economic Times, Moneycontrol),
and GDELT DOC API metadata.
"""

from hypelens.ingestion.nse_announcements import OfficialExchangeIngestor
from hypelens.ingestion.rss_news import GDELTDocIngestor, RSSNewsIngestor
from hypelens.ingestion.social_connector import SocialIngestor

__all__ = [
    "OfficialExchangeIngestor",
    "RSSNewsIngestor",
    "GDELTDocIngestor",
    "SocialIngestor",
]

