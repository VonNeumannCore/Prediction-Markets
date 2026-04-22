"""Entry point. Run every 6h via GitHub Actions.

Pipeline: fetch open Kalshi markets -> filter (<=3d, vol>=1000, 25-75c)
-> rank by 24h volume -> ask Perplexity per candidate -> send Telegram alert
when mispriced AND confidence >= CONFIDENCE_THRESHOLD.
"""
from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv

from analyzer import analyze
from kalshi import fetch_open_markets, filter_short_dated_midprice
from notifier import format_alert, send

CONFIDENCE_THRESHOLD = 70
ANALYZE_LIMIT = 25  # cap LLM calls per run to control cost / runtime


def main() -> int:
    load_dotenv()
    pplx_key = os.environ["PERPLEXITY_API_KEY"]
    tg_token = os.environ["TELEGRAM_BOT_TOKEN"]
    tg_chat = os.environ["TELEGRAM_CHAT_ID"]

    print("Fetching Kalshi open markets...")
    markets = fetch_open_markets()
    print(f"  got {len(markets)} open markets")

    candidates = filter_short_dated_midprice(markets)
    candidates.sort(key=lambda m: m.get("volume_24h", 0), reverse=True)
    candidates = candidates[:ANALYZE_LIMIT]
    print(f"  {len(candidates)} candidates pass filter (analyzing top by volume)")

    sent = 0
    for m in candidates:
        print(f"- {m['ticker']} @ {m.get('last_price')}c  vol24h={m.get('volume_24h')}")
        a = analyze(m, pplx_key)
        if not a:
            continue
        if not a.get("mispriced"):
            continue
        if a.get("confidence_pct", 0) < CONFIDENCE_THRESHOLD:
            continue
        send(tg_token, tg_chat, format_alert(m, a))
        sent += 1
        time.sleep(1)

    print(f"Done. Sent {sent} alert(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
