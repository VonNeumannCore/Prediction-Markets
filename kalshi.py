"""Fetch + filter Kalshi markets via the public read-only API (no auth)."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Iterable

import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"
HEADERS = {"User-Agent": "prediction-markets-scanner/0.1"}

# Multivariate sports parlays: noisy, opaque, useless to an LLM analyst.
EXCLUDE_TICKER_PREFIXES = ("KXMVE",)


def _get_with_retry(url: str, params: dict, *, max_retries: int = 6) -> dict:
    """GET with exponential backoff on 429 / 5xx."""
    delay = 2.0
    last: requests.Response | None = None
    for _ in range(max_retries):
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        last = r
        if r.status_code == 429 or r.status_code >= 500:
            retry_after = float(r.headers.get("Retry-After") or delay)
            print(f"  ! {r.status_code} from kalshi, sleeping {retry_after:.1f}s")
            time.sleep(retry_after)
            delay = min(delay * 2, 60.0)
            continue
        r.raise_for_status()
        return r.json()
    assert last is not None
    last.raise_for_status()
    return {}


def _to_float(x, default=0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _normalize(m: dict) -> dict:
    """Add canonical numeric fields (cents int + contract count int)."""
    m["last_price"] = round(_to_float(m.get("last_price_dollars")) * 100)
    m["volume_24h"] = int(_to_float(m.get("volume_24h_fp")))
    m["open_interest"] = int(_to_float(m.get("open_interest_fp")))
    m["liquidity_dollars_f"] = _to_float(m.get("liquidity_dollars"))
    return m


def fetch_event_meta(max_pages: int = 50) -> dict[str, dict]:
    """Map event_ticker -> {category, series_ticker} for open events."""
    out: dict[str, dict] = {}
    cursor: str | None = None
    for _ in range(max_pages):
        params: dict = {"limit": 200, "status": "open"}
        if cursor:
            params["cursor"] = cursor
        data = _get_with_retry(f"{BASE}/events", params)
        for e in data.get("events", []):
            t = e.get("event_ticker")
            if t:
                out[t] = {
                    "category": e.get("category", ""),
                    "series_ticker": e.get("series_ticker", ""),
                }
        cursor = data.get("cursor") or None
        if not cursor:
            break
        time.sleep(0.3)
    print(f"  cached meta for {len(out)} events")
    return out


def fetch_markets_closing_within(
    days: int,
    page_limit: int = 1000,
    max_pages: int = 60,
) -> list[dict]:
    """Open markets closing within the next `days` days. Drops parlays
    and provisional (just-listed, no-trading-yet) markets. Walks all
    pages within the date window up to `max_pages` (safety cap)."""
    now = datetime.now(timezone.utc)
    horizon = now + timedelta(days=days)
    out: list[dict] = []
    cursor: str | None = None
    for page in range(1, max_pages + 1):
        params: dict = {
            "limit": page_limit,
            "status": "open",
            "min_close_ts": int(now.timestamp()),
            "max_close_ts": int(horizon.timestamp()),
        }
        if cursor:
            params["cursor"] = cursor
        data = _get_with_retry(f"{BASE}/markets", params)
        before = len(out)
        for m in data.get("markets", []):
            if m.get("is_provisional"):
                continue
            if any(m.get("ticker", "").startswith(p) for p in EXCLUDE_TICKER_PREFIXES):
                continue
            out.append(_normalize(m))
        cursor = data.get("cursor") or None
        added = len(out) - before
        if added or page % 10 == 0:
            print(f"  page {page}: +{added} (total {len(out)})")
        if not cursor:
            break
        time.sleep(0.3)
    return out


def filter_by_category(
    markets: Iterable[dict],
    event_meta: dict[str, dict],
    allowed: set[str],
) -> list[dict]:
    """Keep only markets whose parent event is in an allowed category.
    Also enriches each market with its category and series_ticker."""
    out = []
    for m in markets:
        meta = event_meta.get(m.get("event_ticker", ""), {})
        cat = meta.get("category", "")
        if cat in allowed:
            m["category"] = cat
            m["series_ticker"] = meta.get("series_ticker", "")
            out.append(m)
    return out


def filter_tradeable(
    markets: Iterable[dict],
    *,
    volume_band: tuple[int, int] = (100, 5000),
    min_open_interest: int = 100,
    price_band: tuple[int, int] = (25, 75),
) -> list[dict]:
    """Liquidity/price filter tuned for finding mispricings the crowd
    hasn't priced in yet (medium-thin volume, mid-priced)."""
    vlo, vhi = volume_band
    plo, phi = price_band
    keep: list[dict] = []
    for m in markets:
        v = m.get("volume_24h") or 0
        if not (vlo <= v <= vhi):
            continue
        if (m.get("open_interest") or 0) < min_open_interest:
            continue
        last = m.get("last_price")
        if last is None or not (plo <= last <= phi):
            continue
        keep.append(m)
    return keep
