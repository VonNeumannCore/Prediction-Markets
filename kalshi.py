"""Fetch + filter Kalshi markets via the public read-only API (no auth)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterator

import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"


def fetch_open_markets(page_limit: int = 200) -> list[dict]:
    """Return all currently-open markets (handles cursor pagination)."""
    out: list[dict] = []
    cursor: str | None = None
    while True:
        params: dict = {"limit": page_limit, "status": "open"}
        if cursor:
            params["cursor"] = cursor
        r = requests.get(f"{BASE}/markets", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        out.extend(data.get("markets", []))
        cursor = data.get("cursor") or None
        if not cursor:
            break
    return out


def filter_short_dated_midprice(
    markets: Iterator[dict],
    *,
    max_days_to_close: int = 3,
    min_volume_24h: int = 1000,
    price_band: tuple[int, int] = (25, 75),
) -> list[dict]:
    """Markets that are: closing within N days, liquid, and priced in the
    uncertainty band (where mispricings are most actionable)."""
    now = datetime.now(timezone.utc)
    horizon = now + timedelta(days=max_days_to_close)
    lo, hi = price_band

    keep: list[dict] = []
    for m in markets:
        close_str = m.get("close_time")
        if not close_str:
            continue
        try:
            close = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
        except ValueError:
            continue
        if not (now < close <= horizon):
            continue
        if (m.get("volume_24h") or 0) < min_volume_24h:
            continue
        last = m.get("last_price")
        if last is None or not (lo <= last <= hi):
            continue
        keep.append(m)
    return keep
