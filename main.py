"""Entry point. Run every 6h via GitHub Actions.

Pipeline (mentions-only, post v0.2):
  1. List all open Kalshi events with category=='Mentions' (excluding
     sports announcer series).
  2. For each event, fetch nested markets, drop already-resolved (>95c)
     siblings, and require at least MIN_LIVE_SIBLINGS live words and a
     close time within MAX_DAYS_MENTIONS days.
  3. Sort by 24h event volume; analyze top MAX_EVENTS_PER_RUN.
  4. Per event: one Perplexity call returns ranked mispriced words after
     server-side gate (edge vs executable side >= 10pp, confidence >= 70).
  5. For each event with >= 1 surviving edge, send one Telegram alert.
  6. On manual /sup runs (or cron with zero alerts on manual only), send
     a short scan-summary recap.
"""
from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv

from analyzer import analyze_event
from kalshi import (
    MIN_LIVE_SIBLINGS,
    event_volume_24h,
    event_within_horizon,
    fetch_event_with_markets,
    list_mentions_events,
    live_siblings,
)
from notifier import format_mentions_alert, format_run_summary, send

MAX_DAYS_MENTIONS = 7
MAX_EVENTS_PER_RUN = 30


def main() -> int:
    load_dotenv()
    manual = "--manual" in sys.argv
    pplx_key = os.environ["PERPLEXITY_API_KEY"]
    tg_token = os.environ["TELEGRAM_BOT_TOKEN"]
    tg_chat = os.environ["TELEGRAM_CHAT_ID"]

    print(f"Run mode: {'MANUAL (/sup)' if manual else 'cron'}")
    print("Listing mentions events...")
    events = list_mentions_events()

    # Hydrate each event with its sibling markets and apply tradeability
    # filters at the event level (need enough live siblings, closing soon).
    print("Hydrating events with nested markets...")
    candidates: list[tuple[dict, list[dict]]] = []
    for e in events:
        try:
            full = fetch_event_with_markets(e["event_ticker"])
        except Exception as ex:
            print(f"  ! failed to fetch {e.get('event_ticker')}: {ex}")
            continue
        sibs = live_siblings(full)
        if len(sibs) < MIN_LIVE_SIBLINGS:
            continue
        # Only consider close-times of LIVE siblings; ignore the early-
        # closed ones whose close_time is now in the past.
        if not event_within_horizon(sibs, MAX_DAYS_MENTIONS):
            continue
        candidates.append((full, sibs))
        time.sleep(0.2)

    candidates.sort(
        key=lambda pair: -event_volume_24h(pair[0].get("markets", [])),
    )
    candidates = candidates[:MAX_EVENTS_PER_RUN]
    print(
        f"  {len(candidates)} events tradeable "
        f"(>= {MIN_LIVE_SIBLINGS} live words, closing in "
        f"{MAX_DAYS_MENTIONS}d), analyzing top {MAX_EVENTS_PER_RUN}"
    )

    sent = 0
    total_edges = 0
    cost_usd = 0.0
    analyzed = 0

    for event, sibs in candidates:
        ev_t = event.get("event_ticker")
        title = event.get("title", "?")
        print(f"- {ev_t} | {title} | {len(sibs)} live words")
        result = analyze_event(event, sibs, pplx_key)
        if result is None:
            continue
        analyzed += 1
        cost_usd += float(result.get("_cost_usd", 0.0))
        edges = result.get("edges") or []
        if not edges:
            print(
                f"    -> no edges "
                f"(raw {result.get('raw_edge_count', 0)}, post-filter 0)"
            )
            continue
        print(
            f"    -> {len(edges)} edge(s); top: "
            f"{edges[0].get('word')!r} +{edges[0].get('edge_pp')}pp "
            f"@ conf {edges[0].get('confidence_pct')}%"
        )
        send(tg_token, tg_chat, format_mentions_alert(event, result))
        sent += 1
        total_edges += len(edges)
        time.sleep(1)

    print(
        f"Done. Sent {sent} alert(s) covering {total_edges} mispriced "
        f"word(s). Spend ${cost_usd:.4f}"
    )

    # On manual /sup runs always reply with a recap so the user gets
    # acknowledgement. On cron stay silent unless there were no alerts at
    # all and the user asked us to (we don't, today: cron with zero hits
    # is a no-op).
    if manual:
        send(
            tg_token,
            tg_chat,
            format_run_summary(
                events_found=len(events),
                events_analyzed=analyzed,
                events_with_edges=sent,
                total_edges=total_edges,
                cost_usd=cost_usd,
            ),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
