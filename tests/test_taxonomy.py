"""Unit tests for Indian company taxonomy and alias resolver."""

from pathlib import Path
import pytest

from hypelens.models.taxonomy import CompanyProfile, CompanyTaxonomy, get_taxonomy


def test_taxonomy_loading():
    """Verify loading from JSON configuration."""
    tax = get_taxonomy()
    symbols = tax.list_all_symbols()
    assert "RELIANCE" in symbols
    assert "TATAMOTORS" in symbols
    assert "INFY" in symbols
    assert "NVDA" in symbols
    assert "TSLA" in symbols
    assert "AAPL" in symbols


def test_symbol_resolution_exact_and_aliases():
    """Test resolving various aliases, cashtags and colloquial terms."""
    tax = get_taxonomy()

    # Exact symbols
    assert tax.resolve_symbol("RELIANCE") == "RELIANCE"
    assert tax.resolve_symbol("reliance") == "RELIANCE"
    assert tax.resolve_symbol("RELIANCE.NS") == "RELIANCE"

    # Aliases
    assert tax.resolve_symbol("RIL") == "RELIANCE"
    assert tax.resolve_symbol("Jio") == "RELIANCE"
    assert tax.resolve_symbol("Mukesh Ambani") == "RELIANCE"

    # Cashtags & Hashtags
    assert tax.resolve_symbol("$RELIANCE") == "RELIANCE"
    assert tax.resolve_symbol("#TataMotors") == "TATAMOTORS"
    assert tax.resolve_symbol("JLR") == "TATAMOTORS"
    assert tax.resolve_symbol("Infy") == "INFY"
    assert tax.resolve_symbol("$INFY") == "INFY"

    # Backward compatibility
    assert tax.resolve_symbol("NVDA") == "NVDA"
    assert tax.resolve_symbol("Elon Musk") == "TSLA"


def test_get_profile():
    """Verify company profile fields."""
    tax = get_taxonomy()
    profile = tax.get_profile("Jio")
    assert profile is not None
    assert profile.symbol == "RELIANCE"
    assert profile.canonical_name == "Reliance Industries Limited"
    assert profile.sector == "Energy & Conglomerate"
    assert profile.yahoo_symbol == "RELIANCE.NS"


def test_extract_symbols_from_text():
    """Verify extracting multiple tickers from social or news text."""
    tax = get_taxonomy()
    text = "Rumors that $RELIANCE and Jio are partnering with #TataMotors on EV charging network!"
    symbols = tax.extract_symbols_from_text(text)
    assert "RELIANCE" in symbols
    assert "TATAMOTORS" in symbols


def test_sector_retrieval():
    """Verify sector retrieval uses taxonomy."""
    tax = get_taxonomy()
    assert tax.get_sector("RELIANCE") == "Energy & Conglomerate"
    assert tax.get_sector("TATAMOTORS") == "EV & Automotive"
    assert tax.get_sector("INFY") == "Information Technology"
    assert tax.get_sector("NVDA") == "Semiconductors & AI"
