"""Fetch + filter Kalshi 'Mentions' events via the public read-only API."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"
HEADERS = {"User-Agent": "prediction-markets-scanner/0.2"}

# Sports announcer / play-by-play mentions: an LLM has ~zero edge predicting
# which words a broadcast crew will use, so we skip these series outright.
SPORTS_MENTION_SERIES = {
    "KXNBAMENTION",
    "KXMLBMENTION",
    "KXNFLMENTION",
    "KXNHLMENTION",
    "KXFIGHTMENTION",
    "KXSOCCERMENTION",
    "KXTENNISMENTION",
    "KXGOLFMENTION",
}

# Per-word markets early-close to ~99c the moment the subject says the word.
# Anything above this is effectively resolved -- nothing left to bet.
MAX_LIVE_LAST_PRICE = 95

# Need at least this many still-bettable words to bother analyzing the event.
MIN_LIVE_SIBLINGS = 3


def _to_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


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


def _normalize_market(m: dict) -> dict:
    """Add canonical numeric fields. All prices in cents (int 0-100)."""
    m["last_price"] = round(_to_float(m.get("last_price_dollars")) * 100)
    m["yes_bid"] = round(_to_float(m.get("yes_bid_dollars")) * 100)
    m["yes_ask"] = round(_to_float(m.get("yes_ask_dollars")) * 100)
    m["no_bid"] = round(_to_float(m.get("no_bid_dollars")) * 100)
    m["no_ask"] = round(_to_float(m.get("no_ask_dollars")) * 100)
    m["volume_24h"] = int(_to_float(m.get("volume_24h_fp")))
    m["open_interest"] = int(_to_float(m.get("open_interest_fp")))
    # custom_strike.Word is the structured candidate-word field on mentions
    # markets; fall back to yes_sub_title for any oddly-shaped event.
    m["word"] = (
        ((m.get("custom_strike") or {}).get("Word"))
        or m.get("yes_sub_title")
        or ""
    )
    return m


def list_mentions_events(*, max_pages: int = 50) -> list[dict]:
    """List all open Kalshi events in the 'Mentions' category, excluding
    sports announcer series. Returns event dicts WITHOUT nested markets."""
    out: list[dict] = []
    cursor: str | None = None
    for page in range(1, max_pages + 1):
        params: dict = {"limit": 200, "status": "open"}
        if cursor:
            params["cursor"] = cursor
        data = _get_with_retry(f"{BASE}/events", params)
        for e in data.get("events", []):
            if e.get("category") != "Mentions":
                continue
            if e.get("series_ticker") in SPORTS_MENTION_SERIES:
                continue
            out.append(e)
        cursor = data.get("cursor") or None
        if not cursor:
            break
        time.sleep(0.3)
    print(f"  found {len(out)} non-sports mentions events")
    return out


def fetch_event_with_markets(event_ticker: str) -> dict:
    """Pull a single event with all sibling markets (normalized in place)."""
    data = _get_with_retry(
        f"{BASE}/events/{event_ticker}",
        {"with_nested_markets": "true"},
    )
    evt = data.get("event") or {}
    mkts = [_normalize_market(m) for m in (evt.get("markets") or [])]
    evt["markets"] = mkts
    return evt


def live_siblings(event: dict) -> list[dict]:
    """Sibling markets that are still bettable. Drops the already-resolved
    early-close contracts (last_price > MAX_LIVE_LAST_PRICE) and any market
    not in an active/open status."""
    out = []
    for m in event.get("markets", []):
        last = m.get("last_price")
        if last is None or last > MAX_LIVE_LAST_PRICE:
            continue
        status = (m.get("status") or "").lower()
        if status and status not in ("active", "open"):
            continue
        out.append(m)
    return out


def soonest_close(markets: list[dict]) -> datetime | None:
    """Earliest close_time across markets, or None if none parseable."""
    best: datetime | None = None
    for m in markets:
        ct = m.get("close_time")
        if not ct:
            continue
        try:
            t = datetime.fromisoformat(ct.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        if best is None or t < best:
            best = t
    return best


def event_within_horizon(markets: list[dict], days: int) -> bool:
    """True if at least one market closes in the future AND within `days`
    days from now. The lower-bound check matters because Kalshi rewrites
    `close_time` to the actual moment a per-word market early-closes when
    the subject says the word, so already-resolved siblings carry past
    timestamps that would otherwise spuriously pass the upper bound."""
    now = datetime.now(timezone.utc)
    horizon = now + timedelta(days=days)
    for m in markets:
        ct = m.get("close_time")
        if not ct:
            continue
        try:
            t = datetime.fromisoformat(ct.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        if now < t <= horizon:
            return True
    return False


def event_volume_24h(markets: list[dict]) -> int:
    return sum(int(m.get("volume_24h") or 0) for m in markets)
