"""Tests for HypeLens configuration and environment setup (TASK-01)."""

import os
from unittest import mock

import pytest

from hypelens.config import (
    AppConfig,
    DEFAULT_STOCK_UNIVERSE,
    OllamaConfig,
    check_video_dependency,
    get_config,
)


def test_ollama_config_defaults():
    """Verify default Ollama parameters and URL generation."""
    config = OllamaConfig()
    assert config.base_url == "http://localhost:11434"
    assert config.model == "gemma4"
    assert config.timeout_seconds == 30.0
    assert config.allow_mock_fallback is True
    assert config.tags_url == "http://localhost:11434/api/tags"
    assert config.generate_url == "http://localhost:11434/api/generate"
    assert config.chat_url == "http://localhost:11434/api/chat"


def test_ollama_config_env_overrides(monkeypatch):
    """Verify that environment variables correctly override Ollama settings."""
    monkeypatch.setenv("OLLAMA_HOST", "http://192.168.1.100:11434/")
    monkeypatch.setenv("OLLAMA_MODEL", "gemma4:9b-instruct-q4_K_M")
    monkeypatch.setenv("OLLAMA_TIMEOUT", "45.5")
    monkeypatch.setenv("OLLAMA_FALLBACK", "false")

    config = OllamaConfig()
    assert config.base_url == "http://192.168.1.100:11434"
    assert config.model == "gemma4:9b-instruct-q4_K_M"
    assert config.timeout_seconds == 45.5
    assert config.allow_mock_fallback is False
    assert config.chat_url == "http://192.168.1.100:11434/api/chat"


def test_ollama_health_check_graceful_offline():
    """Ensure check_health returns False gracefully when server is offline."""
    config = OllamaConfig(host="http://localhost:59999", timeout_seconds=0.1)
    # Target non-existent port should return False without raising an exception
    assert config.check_health() is False


def test_ollama_health_check_mocked_success():
    """Ensure check_health returns True when Ollama responds with 200 OK."""
    config = OllamaConfig()
    with mock.patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        assert config.check_health() is True


def test_app_config_defaults():
    """Verify top-level application configuration."""
    cfg = get_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.stock_universe == DEFAULT_STOCK_UNIVERSE
    assert "NVDA" in cfg.stock_universe
    assert cfg.ollama.model == "gemma4"

    d = cfg.to_dict()
    assert "stock_universe" in d
    assert "ollama_host" in d
    assert "fixtures_dir" in d


def test_app_config_universe_override(monkeypatch):
    """Verify stock universe customization through environment variable."""
    monkeypatch.setenv("HYPELENS_STOCK_UNIVERSE", "RELIANCE,TCS,INFY")
    cfg = AppConfig()
    assert cfg.stock_universe == ["RELIANCE", "TCS", "INFY"]


def test_check_video_dependency():
    """Verify OpenCV dependency check reports availability and version."""
    status = check_video_dependency()
    assert status["available"] is True
    assert status["library"] == "opencv"
    assert status["version"] is not None
    assert status["error"] is None
