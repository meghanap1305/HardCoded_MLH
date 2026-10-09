"""Resilient JSON parsing and claim validation for Gemma 4 outputs.

Handles markdown code blocks, trailing commas, partial outputs,
and normalizes extracted event taxonomies and aspect sentiments.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class EventCategory(str, Enum):
    """Event classification taxonomy."""
    EARNINGS = "earnings"
    MERGER_ACQUISITION = "merger_acquisition"
    PRODUCT_LAUNCH = "product_launch"
    REGULATORY = "regulatory"
    MANAGEMENT_CHANGE = "management_change"
    GENERAL_SPECULATION = "general_speculation"
    OTHER = "other"


class EpistemicStatus(str, Enum):
    """Rumour and confirmation status categories."""
    RUMOUR_UNVERIFIED = "rumour_unverified"
    CREDIBLE_REPORT = "credible_report"
    OFFICIALLY_CONFIRMED = "officially_confirmed"
    DENIED_RETRACTED = "denied_retracted"


@dataclass
class ExtractedClaim:
    """Structured information extracted by Gemma 4 from a post."""
    ticker: Optional[str]
    claim: str
    event_category: EventCategory
    status: EpistemicStatus
    sentiment: float              # [-1.0, 1.0]
    intensity: float              # [0.0, 1.0]
    aspects: Dict[str, float] = field(default_factory=dict)
    is_promotional_spam: bool = False
    confidence: float = 0.0       # [0.0, 1.0]
    raw_json: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "claim": self.claim,
            "event_category": self.event_category.value,
            "status": self.status.value,
            "sentiment": self.sentiment,
            "intensity": self.intensity,
            "aspects": self.aspects,
            "is_promotional_spam": self.is_promotional_spam,
            "confidence": self.confidence,
        }


def clean_and_extract_json(text: str) -> Dict[str, Any]:
    """Extract and parse a JSON dictionary from arbitrary LLM output text.

    Repairs common LLM formatting artifacts:
    - Markdown code fences (```json ... ```)
    - Trailing commas in objects and arrays
    - Python-style booleans (True/False)
    """
    if not text or not text.strip():
        return {}

    cleaned = text.strip()

    # Remove markdown code block fences if present
    if "```" in cleaned:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if match:
            cleaned = match.group(1).strip()
        else:
            cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()

    # Locate outermost curly braces
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        cleaned = cleaned[first_brace : last_brace + 1]

    # Attempt direct parse first
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Apply heuristics to repair malformed JSON
    # 1. Strip trailing commas before closing braces/brackets
    repaired = re.sub(r",\s*([\]}])", r"\1", cleaned)

    # 2. Convert Python True/False/None to JSON true/false/null
    repaired = re.sub(r"\bTrue\b", "true", repaired)
    repaired = re.sub(r"\bFalse\b", "false", repaired)
    repaired = re.sub(r"\bNone\b", "null", repaired)

    try:
        data = json.loads(repaired)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    return {}


def parse_gemma_extraction(
    raw_text: str,
    fallback_ticker: Optional[str] = None,
) -> ExtractedClaim:
    """Parse Gemma 4 output text into a validated ExtractedClaim object.

    Applies sensible default fallbacks when fields are omitted or malformed.
    """
    data = clean_and_extract_json(raw_text)

    # Parse and normalize ticker
    ticker_val = data.get("ticker")
    if ticker_val and isinstance(ticker_val, str) and ticker_val.lower() != "null":
        ticker = ticker_val.upper().replace("$", "").strip()
    else:
        ticker = fallback_ticker.upper().replace("$", "").strip() if fallback_ticker else None

    claim = str(data.get("claim", "")).strip() or "No explicit claim extracted"

    # Normalize event category
    raw_cat = str(data.get("event_category", "other")).lower().replace(" ", "_")
    try:
        category = EventCategory(raw_cat)
    except ValueError:
        category = EventCategory.OTHER

    # Normalize epistemic status
    raw_status = str(data.get("status", "rumour_unverified")).lower().replace(" ", "_")
    try:
        status = EpistemicStatus(raw_status)
    except ValueError:
        status = EpistemicStatus.RUMOUR_UNVERIFIED

    # Clamp numeric values
    try:
        sentiment = float(data.get("sentiment", 0.0))
        sentiment = max(-1.0, min(1.0, sentiment))
    except (ValueError, TypeError):
        sentiment = 0.0

    try:
        intensity = float(data.get("intensity", 0.5))
        intensity = max(0.0, min(1.0, intensity))
    except (ValueError, TypeError):
        intensity = 0.5

    try:
        confidence = float(data.get("confidence", 0.7 if data else 0.0))
        confidence = max(0.0, min(1.0, confidence))
    except (ValueError, TypeError):
        confidence = 0.0

    aspects_raw = data.get("aspects", {})
    aspects: Dict[str, float] = {}
    if isinstance(aspects_raw, dict):
        for k, v in aspects_raw.items():
            try:
                aspects[str(k)] = max(-1.0, min(1.0, float(v)))
            except (ValueError, TypeError):
                continue

    is_spam = bool(data.get("is_promotional_spam", False))

    return ExtractedClaim(
        ticker=ticker,
        claim=claim,
        event_category=category,
        status=status,
        sentiment=sentiment,
        intensity=intensity,
        aspects=aspects,
        is_promotional_spam=is_spam,
        confidence=confidence,
        raw_json=data,
    )
