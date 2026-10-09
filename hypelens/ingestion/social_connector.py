"""Social media ingestion adapter for X and manual/authorized Reel inputs.

Supports:
1. Official X API v2 recent search (strictly configured via X_BEARER_TOKEN or X_API_KEY).
   Limited to maximum 20 posts per manual fetch with timeouts, deduplication, and error handling.
2. Manual post text entry and user-uploaded media (images/videos).
3. CSV/JSON import of social posts.
4. Preserves input provenance: 'LIVE_API', 'USER_UPLOAD', 'IMPORTED_DATA', or 'DEMO_FIXTURE'.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

from hypelens.config import FIXTURES_DIR, PROJECT_ROOT
from hypelens.data.video import extract_keyframes
from hypelens.models.data_models import (
    AccountTier,
    AuthorMetadata,
    PlatformType,
    SocialPost,
    validate_social_post,
)
from hypelens.models.taxonomy import get_taxonomy
from hypelens.utils.datetime_utils import now_utc, to_utc


CACHE_DIR = PROJECT_ROOT / "hypelens" / "data" / "cache"


class SocialIngestor:
    """Ingestor for social media posts from X API v2, user uploads, or file imports."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        timeout: float = 6.0,
    ) -> None:
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.seen_hashes: set[str] = set()

    @property
    def x_bearer_token(self) -> Optional[str]:
        """Read X API Bearer token from environment variables."""
        return os.getenv("X_BEARER_TOKEN") or os.getenv("TWITTER_BEARER_TOKEN")

    def is_x_api_configured(self) -> bool:
        """Return True only if an authorized X Bearer token is provided."""
        return bool(self.x_bearer_token and self.x_bearer_token.strip())

    def _hash_record(self, text: str, author_or_id: str) -> str:
        base = f"{text.strip().lower()}|{author_or_id.strip().lower()}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

    def fetch_x_recent_posts(
        self,
        query: str = "Reliance OR Tata Motors OR Infosys rumor",
        max_results: int = 20,
    ) -> Dict[str, Any]:
        """Fetch up to 20 posts from official X API v2 search endpoint.

        Never runs automatically; requires explicit user action and configured credentials.
        """
        if not self.is_x_api_configured():
            return {
                "success": False,
                "status": "AUTH_REQUIRED",
                "error": "X API credentials not configured. Provide X_BEARER_TOKEN to enable live search.",
                "items": [],
            }

        max_results = min(max(10, max_results), 20)
        url = "https://api.twitter.com/2/tweets/search/recent"
        headers = {
            "Authorization": f"Bearer {self.x_bearer_token}",
            "User-Agent": "HypeLens-Research-Bot/1.0",
        }
        params = {
            "query": f"{query} -is:retweet lang:en",
            "max_results": max_results,
            "tweet.fields": "created_at,author_id,public_metrics,entities",
        }

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                items = self.parse_x_api_json(data)
                return {
                    "success": True,
                    "status": "LIVE_SUCCESS",
                    "error": None,
                    "items": items,
                }
            elif resp.status_code == 429:
                return {
                    "success": False,
                    "status": "THROTTLED",
                    "error": "X API rate limit exceeded.",
                    "items": [],
                }
            elif resp.status_code in (401, 403):
                return {
                    "success": False,
                    "status": "AUTH_FAILED",
                    "error": f"X API authentication failed (HTTP {resp.status_code}).",
                    "items": [],
                }
            else:
                return {
                    "success": False,
                    "status": "NETWORK_ERROR",
                    "error": f"X API error HTTP {resp.status_code}: {resp.text[:200]}",
                    "items": [],
                }
        except Exception as e:
            return {
                "success": False,
                "status": "NETWORK_ERROR",
                "error": str(e),
                "items": [],
            }

    def parse_x_api_json(self, data: Dict[str, Any]) -> List[SocialPost]:
        """Parse official X API v2 JSON response into validated SocialPost records."""
        taxonomy = get_taxonomy()
        items: List[SocialPost] = []
        collected_at = now_utc()

        tweets = data.get("data", [])
        for tw in tweets:
            text = tw.get("text", "").strip()
            tw_id = str(tw.get("id", ""))
            created_at_str = tw.get("created_at")

            if not text:
                continue

            sig = self._hash_record(text, tw_id)
            if sig in self.seen_hashes:
                continue
            self.seen_hashes.add(sig)

            pub_dt = to_utc(created_at_str) or collected_at
            metrics = tw.get("public_metrics", {})
            tickers = taxonomy.extract_symbols_from_text(text)

            res = validate_social_post({
                "post_id": f"x_{tw_id}",
                "platform": "x_post",
                "timestamp": pub_dt.isoformat(),
                "author": {
                    "author_id": str(tw.get("author_id", "x_user")),
                    "username": f"user_{tw.get('author_id', 'anon')}",
                    "tier": "medium",
                },
                "content_text": text,
                "tickers": tickers,
                "engagement": {
                    "likes": metrics.get("like_count", 0),
                    "reposts": metrics.get("retweet_count", 0),
                    "comments": metrics.get("reply_count", 0),
                },
                "published_at": pub_dt.isoformat(),
                "collected_at": collected_at.isoformat(),
                "source_provider": "x_api_v2",
                "cognitive_provenance": "[LIVE_API]",
            })

            if res.is_valid and res.record:
                items.append(res.record)

        return items

    def create_user_post(
        self,
        text: str,
        author: str = "analyst_manual",
        platform: str = "x_post",
        image_paths: Optional[List[str]] = None,
        video_path: Optional[str] = None,
        ocr_text: Optional[str] = None,
        transcript: Optional[str] = None,
        source_label: str = "USER_UPLOAD",
    ) -> Optional[SocialPost]:
        """Construct a validated SocialPost from manual entry or uploaded media."""
        if not text.strip() and not ocr_text:
            return None

        taxonomy = get_taxonomy()
        now = now_utc()
        full_text = f"{text} {ocr_text or ''}"
        tickers = taxonomy.extract_symbols_from_text(full_text)
        post_id = f"manual_{int(now.timestamp())}_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]}"

        # If a video file is supplied, extract sample keyframes if not provided
        imgs = list(image_paths or [])
        if video_path and not imgs and Path(video_path).exists():
            extracted = extract_keyframes(video_path, max_frames=2)
            imgs.extend(extracted)

        res = validate_social_post({
            "post_id": post_id,
            "platform": platform,
            "timestamp": now.isoformat(),
            "author": {
                "author_id": author,
                "username": author,
                "tier": "small",
            },
            "content_text": text,
            "tickers": tickers,
            "image_paths": imgs,
            "video_path": video_path,
            "ocr_text": ocr_text,
            "transcript": transcript,
            "engagement": {"likes": 1},
            "published_at": now.isoformat(),
            "collected_at": now.isoformat(),
            "source_provider": "manual_entry",
            "cognitive_provenance": f"[{source_label}]",
        })

        if res.is_valid and res.record:
            return res.record
        return None

    def import_from_jsonl(self, jsonl_content: str, source_label: str = "IMPORTED_DATA") -> List[SocialPost]:
        """Import social posts from JSONL text."""
        posts: List[SocialPost] = []
        for line in jsonl_content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                data["source_provider"] = data.get("source_provider", "jsonl_import")
                data["cognitive_provenance"] = f"[{source_label}]"
                res = validate_social_post(data)
                if res.is_valid and res.record:
                    posts.append(res.record)
            except Exception:
                continue
        return posts
