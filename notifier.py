"""Render + send Telegram alerts for Kalshi 'mentions' edges.

Uses Telegram HTML parse mode so the body is essentially plain text -- only
&, <, > need escaping -- with a single bold header line. This avoids the
MarkdownV2 escape gymnastics on `.`, `(`, `)`, `$`, `,` etc., which appear
all over mentions content (URLs, prices, word lists).
"""
from __future__ import annotations

import requests

from analyzer import WIDE_SPREAD_CENTS


def _esc(s) -> str:
    """HTML-escape user/model text for Telegram parse_mode=HTML."""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _fmt_close(close_iso: str | None) -> str:
    if not close_iso:
        return "?"
    return close_iso[:16].replace("T", " ") + " UTC"


def _fmt_implied(last: int, executable: int, spread: int) -> str:
    """'42%' when spread is tight, '34% (executable: 49%)' when wide."""
    if spread > WIDE_SPREAD_CENTS and executable != last:
        return f"{last}% (executable: {executable}%)"
    return f"{last}%"


def format_mentions_alert(event: dict, analysis: dict) -> str:
    """One Telegram message for one event with >= 1 surviving edge."""
    title = event.get("title", "?")
    ev_ticker = event.get("event_ticker", "?")
    siblings = event.get("markets", [])
    sample = siblings[0] if siblings else {}
    expiry = _fmt_close(sample.get("close_time"))
    brief = analysis.get("event_brief") or ""
    edges = analysis.get("edges") or []
    cost = float(analysis.get("_cost_usd") or 0.0)
    model = analysis.get("_model") or "?"

    lines: list[str] = []
    # Single bold header line, body otherwise plain text.
    lines.append(f"<b>MARKET: {_esc(title)}</b>")
    lines.append(f"TICKER: {_esc(ev_ticker)}")
    lines.append(f"EVENT EXPIRY: {_esc(expiry)}")
    lines.append("")
    lines.append("EVENT BRIEF:")
    lines.append(_esc(brief))
    lines.append("")

    if not edges:
        lines.append("NO_EDGE_FOUND")
    else:
        lines.append("TOP MISPRICED WORDS:")
        for i, e in enumerate(edges, 1):
            m = e.get("market") or {}
            implied = _fmt_implied(
                e.get("last_pct", 0),
                e.get("executable_pct", 0),
                e.get("spread", 0),
            )
            ev_url = (e.get("evidence_url") or "").strip()
            lines.append(f"{i}. Word: {_esc(e.get('word', '?'))}")
            lines.append(f"   Ticker: {_esc(e.get('ticker', '?'))}")
            lines.append(f"   Expires: {_esc(_fmt_close(m.get('close_time')))}")
            lines.append(f"   Market implied probability: {_esc(implied)}")
            lines.append(
                f"   Perplexity estimated probability: "
                f"{int(e.get('model_pct_side', 0))}%"
            )
            lines.append(
                f"   Difference: {int(e.get('edge_pp', 0))} percentage "
                f"points (vs executable)"
            )
            lines.append(f"   Suggested side: {_esc(e.get('side', '?'))}")
            lines.append(f"   Confidence: {int(e.get('confidence_pct', 0))}%")
            lines.append(f"   Reason: {_esc(e.get('reason', ''))}")
            if ev_url:
                lines.append(f"   Evidence: {_esc(ev_url)}")
            lines.append("")

    lines.append(f"MODEL COST: ${cost:.4f}")
    lines.append(f"MODEL: {_esc(model)}")
    return "\n".join(lines)


def format_run_summary(
    *,
    events_found: int,
    events_analyzed: int,
    events_with_edges: int,
    total_edges: int,
    cost_usd: float,
) -> str:
    """One short message for manual /sup runs (or any time we want a recap)."""
    lines = [
        "<b>Kalshi mentions scan</b>",
        (
            f"Events found: {events_found}  |  analyzed: {events_analyzed}  "
            f"|  with edges: {events_with_edges}"
        ),
        f"Total mispriced words alerted: {total_edges}",
        f"Perplexity spend this run: ${cost_usd:.4f}",
    ]
    if events_with_edges == 0:
        lines.append("")
        lines.append(
            f"NO_EDGE_FOUND across {events_analyzed} mentions events."
        )
    return "\n".join(lines)


def send(token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(
        url,
        data={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        },
        timeout=30,
    )
    if not r.ok:
        print(f"  ! telegram error: {r.status_code} {r.text}")
        r.raise_for_status()
