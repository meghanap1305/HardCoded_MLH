"""Timezone and datetime utilities for HypeLens.

Enforces strict timezone-aware UTC datetime storage and formatting for Indian Standard Time (IST / Asia/Kolkata).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo


IST_ZONE = ZoneInfo("Asia/Kolkata")
UTC_ZONE = timezone.utc


def now_utc() -> datetime:
    """Return the current time in UTC."""
    return datetime.now(UTC_ZONE)


def to_utc(val: Any) -> Optional[datetime]:
    """Normalize any datetime, ISO string, or timestamp into timezone-aware UTC datetime."""
    if val is None:
        return None

    if isinstance(val, (int, float)):
        ts = float(val)
        if ts > 1e11:  # epoch milliseconds
            ts /= 1000.0
        return datetime.fromtimestamp(ts, tz=UTC_ZONE)

    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=UTC_ZONE)
        return val.astimezone(UTC_ZONE)

    if isinstance(val, str):
        val_clean = val.strip()
        if not val_clean:
            return None
        if val_clean.endswith("Z"):
            val_clean = val_clean[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(val_clean)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC_ZONE)
            return dt.astimezone(UTC_ZONE)
        except ValueError:
            pass

        # Try parsing standard email/RFC-822 date format (common in RSS feeds)
        from email.utils import parsedate_to_datetime
        try:
            dt = parsedate_to_datetime(val_clean)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC_ZONE)
            return dt.astimezone(UTC_ZONE)
        except Exception:
            pass

        # Try custom Indian exchange formats
        # 1. SEBI format: "08 Oct, 2026 +0530" or "08 Oct, 2026"
        for fmt in ("%d %b, %Y %z", "%d %b, %Y", "%d %B, %Y %z", "%d %B, %Y"):
            try:
                dt = datetime.strptime(val_clean, fmt)
                if dt.tzinfo is None:
                    # Assume IST if no offset is present
                    dt = dt.replace(tzinfo=IST_ZONE)
                return dt.astimezone(UTC_ZONE)
            except ValueError:
                pass

        # 2. NSE format: "09-Oct-2026 15:32:52" (local IST time)
        for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M", "%d-%b-%Y", "%d-%B-%Y %H:%M:%S"):
            try:
                dt = datetime.strptime(val_clean, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=IST_ZONE)
                return dt.astimezone(UTC_ZONE)
            except ValueError:
                pass

    return None


def to_ist(dt: Optional[datetime]) -> Optional[datetime]:
    """Convert a UTC or timezone-aware datetime to Asia/Kolkata (IST)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC_ZONE)
    return dt.astimezone(IST_ZONE)


def format_ist(dt: Optional[datetime], fmt: str = "%d %b %Y, %I:%M %p IST") -> str:
    """Format a datetime for user-facing display in Asia/Kolkata."""
    if dt is None:
        return "N/A"
    ist_dt = to_ist(dt)
    if ist_dt is None:
        return "N/A"
    return ist_dt.strftime(fmt)
