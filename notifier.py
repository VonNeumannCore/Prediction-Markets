"""Format + send a clean Telegram alert."""
from __future__ import annotations

import requests

_MD2_SPECIAL = r"_*[]()~`>#+-=|{}.!\\"


def _esc(s) -> str:
    """Escape text for Telegram MarkdownV2."""
    out = []
    for c in str(s):
        if c in _MD2_SPECIAL:
            out.append("\\" + c)
        else:
            out.append(c)
    return "".join(out)


def format_alert(market: dict, analysis: dict) -> str:
    title = market.get("title") or market.get("ticker") or "?"
    ticker = market.get("ticker", "?")
    last = market.get("last_price")
    your_p = analysis["your_probability_pct"]
    edge = abs(your_p - last) if isinstance(last, int) else "?"
    verdict = analysis["verdict"]
    conf = analysis["confidence_pct"]
    reason = analysis["reasoning"]
    close = (market.get("close_time") or "")[:16].replace("T", " ")

    lines = [
        f"*{_esc(title)}*",
        f"`{_esc(ticker)}` \u00b7 closes {_esc(close)} UTC",
        "",
        f"Market *{last}c* \u2022 Model *{your_p}%* \u2022 Edge *{edge}pp*",
        f"\u27a4 Bet *{verdict}* \u00b7 confidence *{conf}%*",
        "",
        f"_{_esc(reason)}_",
    ]
    sources = analysis.get("sources") or []
    if sources:
        lines.append("")
        for i, s in enumerate(sources[:3], 1):
            t = _esc(s.get("title", "source"))
            u = s.get("url", "")
            lines.append(f"{i}\\. [{t}]({_esc(u)})")
    return "\n".join(lines)


def send(token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(
        url,
        data={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "MarkdownV2",
            "disable_web_page_preview": True,
        },
        timeout=30,
    )
    if not r.ok:
        print(f"  ! telegram error: {r.status_code} {r.text}")
        r.raise_for_status()
