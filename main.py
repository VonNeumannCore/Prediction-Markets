"""Entry point. Run every 6h via GitHub Actions.

Pipeline:
  1. Fetch all open events -> {event_ticker: category} cache.
  2. Fetch markets closing within MAX_DAYS_TO_CLOSE.
  3. Keep only markets in whitelisted categories (where LLM has edge).
  4. Apply tradeability filter (medium-thin volume, OI floor, mid-priced).
  5. Ask Perplexity per candidate with strict prompt.
  6. Send Telegram alert when mispriced AND confidence >= threshold.
"""
from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv

from analyzer import analyze
from kalshi import (
    fetch_event_categories,
    fetch_markets_closing_within,
    filter_by_category,
    filter_tradeable,
)
from notifier import format_alert, send

MAX_DAYS_TO_CLOSE = 3
CONFIDENCE_THRESHOLD = 75
ANALYZE_LIMIT = 25

ALLOWED_CATEGORIES = {
    "Politics",
    "Entertainment",
    "Companies",
    "Health",
    "Science and Technology",
    "World",
}


def main() -> int:
    load_dotenv()
    pplx_key = os.environ["PERPLEXITY_API_KEY"]
    tg_token = os.environ["TELEGRAM_BOT_TOKEN"]
    tg_chat = os.environ["TELEGRAM_CHAT_ID"]

    print("Caching event categories...")
    cat_by_event = fetch_event_categories()

    print(f"Fetching markets closing within {MAX_DAYS_TO_CLOSE} days...")
    markets = fetch_markets_closing_within(MAX_DAYS_TO_CLOSE)
    print(f"  got {len(markets)} short-dated open markets")

    in_scope = filter_by_category(markets, cat_by_event, ALLOWED_CATEGORIES)
    print(f"  {len(in_scope)} are in allowed categories: {sorted(ALLOWED_CATEGORIES)}")

    candidates = filter_tradeable(in_scope)
    candidates.sort(key=lambda m: m.get("volume_24h", 0), reverse=True)
    candidates = candidates[:ANALYZE_LIMIT]
    print(f"  {len(candidates)} pass tradeability filter (analyzing top by volume)")

    sent = 0
    for m in candidates:
        print(f"- [{m.get('category')}] {m['ticker']} @ {m.get('last_price')}c "
              f"vol24h={m.get('volume_24h')} oi={m.get('open_interest')}")
        a = analyze(m, pplx_key)
        if not a:
            continue
        if not a.get("mispriced"):
            print(f"    -> not mispriced (conf {a.get('confidence_pct')}%)")
            continue
        if a.get("confidence_pct", 0) < CONFIDENCE_THRESHOLD:
            print(f"    -> below threshold (conf {a.get('confidence_pct')}%)")
            continue
        print(f"    -> ALERT: bet {a.get('verdict')} @ {a.get('confidence_pct')}%")
        send(tg_token, tg_chat, format_alert(m, a))
        sent += 1
        time.sleep(1)

    print(f"Done. Sent {sent} alert(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
