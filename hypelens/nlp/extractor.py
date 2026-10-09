"""Multimodal Gemma 4 cognitive extractor client with deterministic mock fallback.

Interfaces with local Ollama daemon for text + image extraction. Gracefully falls back
to an explicit mock extractor when Ollama is unreachable, timing out, or disabled in tests.
"""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests

from hypelens.config import OllamaConfig, get_config
from hypelens.models.data_models import SocialPost
from hypelens.nlp.parser import (
    EpistemicStatus,
    EventCategory,
    ExtractedClaim,
    parse_gemma_extraction,
)
from hypelens.nlp.prompts import (
    GEMMA_EXTRACTION_SYSTEM_PROMPT,
    build_extraction_prompt,
)


@dataclass
class ExtractionResult:
    """Outcome of multimodal cognitive extraction."""
    claim: ExtractedClaim
    is_mock: bool
    model_used: str
    latency_sec: float
    error: Optional[str] = None


def encode_image_to_base64(image_path: Union[str, Path]) -> Optional[str]:
    """Read an image file and convert it to a base64 encoded string."""
    p = Path(image_path)
    if not p.exists() or not p.is_file():
        return None
    try:
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


class GemmaExtractor:
    """Multimodal extractor using local Gemma 4 via Ollama REST API."""

    def __init__(self, config: Optional[OllamaConfig] = None, force_mock: bool = False):
        self.config = config or get_config().ollama
        self.force_mock = force_mock

    def _generate_mock_claim(
        self,
        content_text: str,
        tickers: List[str],
        ocr_text: Optional[str] = None,
    ) -> ExtractedClaim:
        """Deterministic mock extraction for testing and offline environments.
        
        Explicitly flagged as mock output; never presented as actual Gemma 4 inference.
        """
        combined = f"{content_text} {ocr_text or ''}".lower()

        # Ticker resolution
        ticker = tickers[0] if tickers else None
        if not ticker:
            for word in combined.split():
                if word.startswith("$") and len(word) > 1:
                    ticker = word[1:].upper().rstrip(".,!?")
                    break

        # Event category heuristics
        if any(w in combined for w in ["deal", "partner", "acquire", "merger", "buying", "buyout"]):
            category = EventCategory.MERGER_ACQUISITION
        elif any(w in combined for w in ["earnings", "revenue", "quarter", "margin"]):
            category = EventCategory.EARNINGS
        elif any(w in combined for w in ["permit", "regulatory", "sec", "approval"]):
            category = EventCategory.REGULATORY
        elif any(w in combined for w in ["launch", "product", "release", "chip"]):
            category = EventCategory.PRODUCT_LAUNCH
        else:
            category = EventCategory.GENERAL_SPECULATION

        # Status heuristics
        if any(w in combined for w in ["rumor", "rumour", "leak", "whisper", "unconfirmed"]):
            status = EpistemicStatus.RUMOUR_UNVERIFIED
        elif any(w in combined for w in ["official", "announced", "confirmed", "press release"]):
            status = EpistemicStatus.OFFICIALLY_CONFIRMED
        elif any(w in combined for w in ["denied", "retracted", "fake"]):
            status = EpistemicStatus.DENIED_RETRACTED
        elif any(w in combined for w in ["sources", "reported", "insider"]):
            status = EpistemicStatus.CREDIBLE_REPORT
        else:
            status = EpistemicStatus.RUMOUR_UNVERIFIED

        # Sentiment heuristics
        is_spam = any(w in combined for w in ["100x", "to the moon", "guaranteed", "buy now"])
        if is_spam or any(w in combined for w in ["surge", "massive", "momentum", "bull"]):
            sentiment = 0.8
            intensity = 0.9 if is_spam else 0.7
        elif any(w in combined for w in ["sell the news", "reversal", "delayed", "bear"]):
            sentiment = -0.6
            intensity = 0.7
        else:
            sentiment = 0.2
            intensity = 0.4

        return ExtractedClaim(
            ticker=ticker,
            claim=f"[MOCK] Detected {category.value} ({status.value}) for {ticker or 'unknown'}",
            event_category=category,
            status=status,
            sentiment=sentiment,
            intensity=intensity,
            aspects={"growth": sentiment},
            is_promotional_spam=is_spam,
            confidence=0.75,
            raw_json={"is_mock": True, "source": "deterministic_mock_extractor"},
        )

    def extract(
        self,
        content_text: str,
        tickers: Optional[List[str]] = None,
        image_paths: Optional[List[Union[str, Path]]] = None,
        ocr_text: Optional[str] = None,
        transcript: Optional[str] = None,
    ) -> ExtractionResult:
        """Extract structured claim from text and visual inputs."""
        start_time = time.time()
        tickers_list = tickers or []
        images_list = image_paths or []

        if self.force_mock:
            claim = self._generate_mock_claim(content_text, tickers_list, ocr_text)
            return ExtractionResult(
                claim=claim,
                is_mock=True,
                model_used="mock-gemma4",
                latency_sec=round(time.time() - start_time, 4),
            )

        # Encode images for Ollama
        base64_images: List[str] = []
        for img in images_list:
            b64 = encode_image_to_base64(img)
            if b64:
                base64_images.append(b64)

        prompt = build_extraction_prompt(
            content_text=content_text,
            ocr_text=ocr_text,
            transcript=transcript,
            has_images=bool(base64_images),
        )

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "prompt": prompt,
            "system": GEMMA_EXTRACTION_SYSTEM_PROMPT,
            "stream": False,
        }
        if base64_images:
            payload["images"] = base64_images

        try:
            resp = requests.post(
                self.config.generate_url,
                json=payload,
                timeout=self.config.timeout_seconds,
            )
            if resp.status_code == 200:
                data = resp.json()
                raw_response = data.get("response", "")
                fallback_t = tickers_list[0] if tickers_list else None
                claim = parse_gemma_extraction(raw_response, fallback_ticker=fallback_t)
                return ExtractionResult(
                    claim=claim,
                    is_mock=False,
                    model_used=self.config.model,
                    latency_sec=round(time.time() - start_time, 4),
                )
            error_msg = f"Ollama returned HTTP status {resp.status_code}: {resp.text}"
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            error_msg = f"Ollama communication failure: {e}"
        except Exception as e:
            error_msg = f"Unexpected error during extraction: {e}"

        # Handle failure: Fall back to mock if permitted
        if self.config.allow_mock_fallback:
            claim = self._generate_mock_claim(content_text, tickers_list, ocr_text)
            return ExtractionResult(
                claim=claim,
                is_mock=True,
                model_used="mock-gemma4",
                latency_sec=round(time.time() - start_time, 4),
                error=error_msg,
            )

        # Fallback disabled, return empty low-confidence claim
        empty_claim = parse_gemma_extraction("", fallback_ticker=tickers_list[0] if tickers_list else None)
        return ExtractionResult(
            claim=empty_claim,
            is_mock=False,
            model_used=self.config.model,
            latency_sec=round(time.time() - start_time, 4),
            error=error_msg,
        )

    def extract_from_post(self, post: SocialPost) -> ExtractionResult:
        """Convenience helper extracting directly from a validated SocialPost model."""
        return self.extract(
            content_text=post.content_text,
            tickers=post.tickers,
            image_paths=post.image_paths,
            ocr_text=post.ocr_text,
            transcript=post.transcript,
        )
