"""Company taxonomy and alias resolution module for HypeLens.

Enables resolving colloquial mentions, cashtags, hashtags, and executive names
to canonical stock symbols using configurable data (config/companies_india.json).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from hypelens.config import PROJECT_ROOT


CONFIG_PATH = PROJECT_ROOT / "config" / "companies_india.json"


@dataclass
class CompanyProfile:
    """Configurable company profile containing identifiers and alias aliases."""

    symbol: str
    nse_symbol: Optional[str]
    bse_symbol: Optional[str]
    yahoo_symbol: str
    canonical_name: str
    sector: str
    aliases: List[str] = field(default_factory=list)
    cashtags: List[str] = field(default_factory=list)
    hashtags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, any]:
        return {
            "symbol": self.symbol,
            "nse_symbol": self.nse_symbol,
            "bse_symbol": self.bse_symbol,
            "yahoo_symbol": self.yahoo_symbol,
            "canonical_name": self.canonical_name,
            "sector": self.sector,
            "aliases": self.aliases,
            "cashtags": self.cashtags,
            "hashtags": self.hashtags,
        }


class CompanyTaxonomy:
    """Registry and resolver for company symbols and aliases."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self.config_path = config_path or CONFIG_PATH
        self._profiles_by_symbol: Dict[str, CompanyProfile] = {}
        self._lookup_map: Dict[str, str] = {}
        self._sector_map: Dict[str, str] = {}
        self.load()

    def load(self) -> None:
        """Load company profiles from JSON configuration."""
        if not self.config_path.exists():
            return

        with open(self.config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._profiles_by_symbol.clear()
        self._lookup_map.clear()
        self._sector_map.clear()

        for item in data:
            profile = CompanyProfile(
                symbol=item["symbol"].upper(),
                nse_symbol=item.get("nse_symbol"),
                bse_symbol=item.get("bse_symbol"),
                yahoo_symbol=item.get("yahoo_symbol", item["symbol"]),
                canonical_name=item["canonical_name"],
                sector=item.get("sector", "Diversified / Other"),
                aliases=item.get("aliases", []),
                cashtags=item.get("cashtags", []),
                hashtags=item.get("hashtags", []),
            )
            sym = profile.symbol
            self._profiles_by_symbol[sym] = profile
            self._sector_map[sym] = profile.sector

            # Populate lookup map (case-insensitive keys)
            self._lookup_map[sym.lower()] = sym
            if profile.nse_symbol:
                self._lookup_map[profile.nse_symbol.lower()] = sym
                self._lookup_map[f"{profile.nse_symbol.lower()}.ns"] = sym
            if profile.bse_symbol:
                self._lookup_map[profile.bse_symbol.lower()] = sym
            self._lookup_map[profile.canonical_name.lower()] = sym

            for alias in profile.aliases:
                self._lookup_map[alias.lower().strip()] = sym

            for cashtag in profile.cashtags:
                cleaned = cashtag.lower().strip()
                self._lookup_map[cleaned] = sym
                self._lookup_map[cleaned.replace("$", "")] = sym

            for hashtag in profile.hashtags:
                cleaned = hashtag.lower().strip()
                self._lookup_map[cleaned] = sym
                self._lookup_map[cleaned.replace("#", "")] = sym

    def get_profile(self, symbol_or_alias: str) -> Optional[CompanyProfile]:
        """Get company profile by symbol or alias."""
        canonical = self.resolve_symbol(symbol_or_alias)
        if canonical:
            return self._profiles_by_symbol.get(canonical)
        return None

    def resolve_symbol(self, query: str) -> Optional[str]:
        """Resolve a company name, alias, cashtag or ticker to canonical symbol."""
        if not query:
            return None
        cleaned = query.strip().lower()
        if cleaned in self._lookup_map:
            return self._lookup_map[cleaned]

        # Try removing leading $ or #
        stripped = re.sub(r"^[$#]", "", cleaned)
        if stripped in self._lookup_map:
            return self._lookup_map[stripped]

        # Try stripping exchange suffixes like .ns, .bo
        no_suffix = re.sub(r"\.(ns|bo)$", "", stripped)
        if no_suffix in self._lookup_map:
            return self._lookup_map[no_suffix]

        return None

    def get_sector(self, symbol: str) -> str:
        """Retrieve sector for a symbol."""
        canonical = self.resolve_symbol(symbol) or symbol.upper()
        return self._sector_map.get(canonical, "Diversified Tech / Other")

    def extract_symbols_from_text(self, text: str) -> List[str]:
        """Find all mentioned company symbols in freeform text."""
        if not text:
            return []

        matched: Set[str] = set()
        lowered = text.lower()

        # Check cashtags and exact words
        tokens = set(re.findall(r"[\$#]?[A-Za-z0-9_.]+", text))
        for token in tokens:
            sym = self.resolve_symbol(token)
            if sym:
                matched.add(sym)

        # Check multi-word aliases (e.g., "Tata Motors", "Reliance Retail")
        for alias_key, sym in self._lookup_map.items():
            if " " in alias_key and alias_key in lowered:
                matched.add(sym)

        return sorted(list(matched))

    def list_all_symbols(self) -> List[str]:
        """Return list of all registered canonical symbols."""
        return sorted(list(self._profiles_by_symbol.keys()))

    def list_all_profiles(self) -> List[CompanyProfile]:
        """Return list of all registered company profiles."""
        return list(self._profiles_by_symbol.values())


# Global singleton instance
_GLOBAL_TAXONOMY: Optional[CompanyTaxonomy] = None


def get_taxonomy() -> CompanyTaxonomy:
    """Get or initialize the global taxonomy singleton."""
    global _GLOBAL_TAXONOMY
    if _GLOBAL_TAXONOMY is None:
        _GLOBAL_TAXONOMY = CompanyTaxonomy()
    return _GLOBAL_TAXONOMY
