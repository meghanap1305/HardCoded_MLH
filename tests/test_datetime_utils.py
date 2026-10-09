"""Unit tests for timezone utilities and IST/UTC conversions."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import pytest

from hypelens.utils.datetime_utils import (
    IST_ZONE,
    UTC_ZONE,
    format_ist,
    now_utc,
    to_ist,
    to_utc,
)


def test_now_utc():
    """Verify now_utc returns timezone-aware UTC datetime."""
    now = now_utc()
    assert now.tzinfo is not None
    assert now.tzinfo == timezone.utc


def test_to_utc_normalization():
    """Verify conversion of various formats to UTC."""
    # From naive
    dt_naive = datetime(2026, 10, 9, 12, 0, 0)
    dt_utc = to_utc(dt_naive)
    assert dt_utc.tzinfo == timezone.utc
    assert dt_utc.hour == 12

    # From IST (+05:30)
    ist_str = "2026-10-09T15:30:00+05:30"
    dt_from_ist = to_utc(ist_str)
    assert dt_from_ist.tzinfo == timezone.utc
    assert dt_from_ist.hour == 10  # 15:30 - 5:30 = 10:00 UTC

    # From epoch
    assert to_utc(1700000000).tzinfo == timezone.utc

    # From RSS RFC-822 date
    rfc_date = "Fri, 09 Oct 2026 15:30:00 +0530"
    dt_rfc = to_utc(rfc_date)
    assert dt_rfc is not None
    assert dt_rfc.tzinfo == timezone.utc
    assert dt_rfc.hour == 10

    # Null / Invalid
    assert to_utc(None) is None
    assert to_utc("invalid-date") is None


def test_to_ist_conversion():
    """Verify converting UTC to Asia/Kolkata."""
    dt_utc = datetime(2026, 10, 9, 10, 0, 0, tzinfo=timezone.utc)
    dt_ist = to_ist(dt_utc)
    assert dt_ist.tzinfo == IST_ZONE
    assert dt_ist.hour == 15
    assert dt_ist.minute == 30


def test_format_ist_display():
    """Verify user-facing formatting in IST."""
    dt_utc = datetime(2026, 10, 9, 10, 0, 0, tzinfo=timezone.utc)
    formatted = format_ist(dt_utc)
    assert "09 Oct 2026" in formatted
    assert "03:30 PM IST" in formatted

    assert format_ist(None) == "N/A"
