"""Configuration module for HypeLens.

Provides application configuration, Ollama client configuration helper,
environment variable overrides, and video extraction dependency validation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# Base directory for the HypeLens project
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "hypelens" / "data"
FIXTURES_DIR = DATA_DIR / "fixtures"

# Default benchmark equity universe
DEFAULT_STOCK_UNIVERSE = ["NVDA", "TSLA", "AAPL"]


@dataclass
class OllamaConfig:
    """Configuration helper for local Ollama and Gemma 4 inference."""

    host: str = field(
        default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434")
    )
    model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "gemma4")
    )
    timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("OLLAMA_TIMEOUT", "30.0"))
    )
    allow_mock_fallback: bool = field(
        default_factory=lambda: os.getenv("OLLAMA_FALLBACK", "true").lower()
        in ("1", "true", "yes")
    )

    @property
    def base_url(self) -> str:
        """Normalized base URL without trailing slash."""
        return self.host.rstrip("/")

    @property
    def tags_url(self) -> str:
        """Endpoint to query available local models."""
        return f"{self.base_url}/api/tags"

    @property
    def generate_url(self) -> str:
        """Endpoint for completion generation."""
        return f"{self.base_url}/api/generate"

    @property
    def chat_url(self) -> str:
        """Endpoint for multimodal chat interactions."""
        return f"{self.base_url}/api/chat"

    def check_health(self) -> bool:
        """Check if local Ollama daemon is reachable.
        
        Returns False gracefully if server is unreachable or offline.
        """
        try:
            import requests

            resp = requests.get(self.tags_url, timeout=min(self.timeout_seconds, 2.0))
            return resp.status_code == 200
        except Exception:
            return False


@dataclass
class AppConfig:
    """Central application configuration."""

    stock_universe: List[str] = field(
        default_factory=lambda: os.getenv(
            "HYPELENS_STOCK_UNIVERSE", ",".join(DEFAULT_STOCK_UNIVERSE)
        ).split(",")
    )
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    data_dir: Path = field(default_factory=lambda: DATA_DIR)
    fixtures_dir: Path = field(default_factory=lambda: FIXTURES_DIR)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a dictionary."""
        return {
            "stock_universe": self.stock_universe,
            "ollama_host": self.ollama.host,
            "ollama_model": self.ollama.model,
            "ollama_timeout": self.ollama.timeout_seconds,
            "ollama_fallback": self.ollama.allow_mock_fallback,
            "data_dir": str(self.data_dir),
            "fixtures_dir": str(self.fixtures_dir),
        }


def get_config() -> AppConfig:
    """Retrieve the application configuration instance."""
    return AppConfig()


def check_video_dependency() -> Dict[str, Any]:
    """Validate that the OpenCV video extraction backend is available and operational.

    Returns a status dictionary detailing module availability, version,
    and capability to decode video frames.
    """
    result: Dict[str, Any] = {
        "available": False,
        "library": "opencv",
        "version": None,
        "error": None,
    }
    try:
        import cv2

        result["available"] = True
        result["version"] = getattr(cv2, "__version__", "unknown")
    except ImportError as e:
        result["error"] = f"OpenCV import error: {e}"
    except Exception as e:
        result["error"] = f"Unexpected OpenCV error: {e}"

    return result
