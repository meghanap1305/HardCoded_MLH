"""Unit tests for Indian news RSS, GDELT, and official exchange ingestion (TASK-13)."""

from pathlib import Path
import pytest

from hypelens.config import FIXTURES_DIR
from hypelens.ingestion.nse_announcements import OfficialExchangeIngestor
from hypelens.ingestion.rss_news import GDELTDocIngestor, RSSNewsIngestor


def test_rss_news_parser_with_sample_fixture(tmp_path: Path):
    """Verify RSS parser normalizes records to FinancialNewsItem schema."""
    ingestor = RSSNewsIngestor(cache_dir=tmp_path)
    fixture_path = FIXTURES_DIR / "sample_news_rss.xml"
    with open(fixture_path, "r", encoding="utf-8") as f:
        xml_content = f.read()

    items = ingestor.parse_rss_xml(xml_content, publisher="Livemint")
    assert len(items) == 2

    # Verify INFY item
    infy_item = next(it for it in items if "INFY" in it.tickers)
    assert "Infosys" in infy_item.headline
    assert infy_item.publisher == "Livemint"
    assert infy_item.source_provider == "news_rss"
    assert infy_item.published_at is not None
    assert infy_item.collected_at is not None
    assert infy_item.url is not None

    # Verify HDFCBANK item
    hdfc_item = next(it for it in items if "HDFCBANK" in it.tickers)
    assert "HDFC Bank" in hdfc_item.headline


def test_rss_deduplication(tmp_path: Path):
    """Verify that identical records are deduplicated."""
    ingestor = RSSNewsIngestor(cache_dir=tmp_path)
    fixture_path = FIXTURES_DIR / "sample_news_rss.xml"
    with open(fixture_path, "r", encoding="utf-8") as f:
        xml_content = f.read()

    items_first = ingestor.parse_rss_xml(xml_content, publisher="Livemint")
    items_second = ingestor.parse_rss_xml(xml_content, publisher="Livemint")
    assert len(items_first) == 2
    assert len(items_second) == 0  # All duplicates filtered


def test_official_exchange_ingestion_nse(tmp_path: Path):
    """Verify official NSE announcements ingestion and symbol mapping."""
    ingestor = OfficialExchangeIngestor(cache_dir=tmp_path)
    fixture_path = FIXTURES_DIR / "sample_nse_feed.xml"
    with open(fixture_path, "r", encoding="utf-8") as f:
        xml_content = f.read()

    items = ingestor.parse_announcements_xml(xml_content, source="NSE")
    assert len(items) == 2

    rel_item = next(it for it in items if "RELIANCE" in it.tickers)
    assert "RELIANCE" in rel_item.headline
    assert rel_item.source_provider == "official_nse"
    assert "[NSE OFFICIAL DISCLOSURE]" in rel_item.cognitive_provenance

    tamo_item = next(it for it in items if "TATAMOTORS" in it.tickers)
    assert "TATAMOTORS" in tamo_item.headline


def test_official_exchange_ingestion_sebi(tmp_path: Path):
    """Verify SEBI regulatory circulars parsing."""
    ingestor = OfficialExchangeIngestor(cache_dir=tmp_path)
    fixture_path = FIXTURES_DIR / "sample_sebi_feed.xml"
    with open(fixture_path, "r", encoding="utf-8") as f:
        xml_content = f.read()

    items = ingestor.parse_announcements_xml(xml_content, source="SEBI")
    assert len(items) == 1
    sebi_item = items[0]
    assert "Surveillance Framework" in sebi_item.headline
    assert sebi_item.source_provider == "official_sebi"
    assert "[SEBI OFFICIAL DISCLOSURE]" in sebi_item.cognitive_provenance


def test_gdelt_doc_ingestor(tmp_path: Path):
    """Verify GDELT DOC API JSON response parsing and metadata extraction."""
    import json
    ingestor = GDELTDocIngestor(cache_dir=tmp_path)
    fixture_path = FIXTURES_DIR / "sample_gdelt_feed.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = ingestor.parse_gdelt_json(data)
    assert len(items) == 2

    ril_item = next(it for it in items if "RELIANCE" in it.tickers)
    assert "Reliance Jio" in ril_item.headline
    assert ril_item.source_provider == "gdelt_doc_api"
    assert ril_item.publisher == "economictimes.indiatimes.com"


def test_offline_fallback_when_http_fails(tmp_path: Path):
    """Ensure fetch_feed falls back gracefully to offline fixture when URL is invalid."""
    ingestor = RSSNewsIngestor(cache_dir=tmp_path, use_offline_fallback=True)
    # Using unreachable mock address
    items = ingestor.fetch_feed(publisher="Livemint", url="http://127.0.0.1:9999/nonexistent")
    assert len(items) == 2
    assert any("INFY" in it.tickers for it in items)
