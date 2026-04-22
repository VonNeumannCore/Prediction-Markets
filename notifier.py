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


def _kalshi_url(market: dict) -> str | None:
    """Best-effort link to the Kalshi page. Series-level URL is reliable;
    deep-linking to the exact event would require scraping their slugs."""
    series = (market.get("series_ticker") or "").lower()
    if series:
        return f"https://kalshi.com/markets/{series}"
    return None


def format_alert(market: dict, analysis: dict) -> str:
    title = market.get("title") or market.get("ticker") or "?"
    ticker = market.get("ticker", "?")
    last = market.get("last_price")
    verdict = analysis["verdict"]
    conf = analysis["confidence_pct"]
    reason = analysis["reasoning"]
    close = (market.get("close_time") or "")[:16].replace("T", " ")
    cat = market.get("category", "")

    header = f"*{_esc(title)}*"
    kurl = _kalshi_url(market)
    if kurl:
        header = f"*[{_esc(title)}]({_esc(kurl)})*"

    lines = [
        header,
        f"`{_esc(ticker)}` \u00b7 {_esc(cat)} \u00b7 closes {_esc(close)} UTC",
        "",
        f"Market *{last}c*",
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


def format_summary(
    *,
    fetched: int,
    in_scope: int,
    tradeable: int,
    analyzed: int,
    alerts_sent: int,
    cost_usd: float,
    best: dict | None,
) -> str:
    """Summary message for /sup runs (or any time we want a recap).

    `best` is a dict like:
      {
        "market": <kalshi market dict>,
        "analysis": <analyzer result dict>,
        "score": <float>,
      }
    """
    lines = [
        "*Prediction Markets scan*",
        (
            f"Fetched *{fetched}* \u00b7 in scope *{in_scope}* \u00b7 "
            f"tradeable *{tradeable}* \u00b7 analyzed *{analyzed}*"
        ),
        f"Alerts sent: *{alerts_sent}*",
        f"Perplexity spend this run: *${_esc(f'{cost_usd:.4f}')}*",
    ]
    if alerts_sent == 0:
        if best:
            m = best["market"]
            a = best["analysis"]
            title = m.get("title") or m.get("ticker") or "?"
            ticker = m.get("ticker", "?")
            last = m.get("last_price")
            verdict = a.get("verdict", "?")
            conf = a.get("confidence_pct", 0)
            your_p = a.get("your_probability_pct", 0)
            edge = abs(int(your_p) - int(last)) if last is not None else 0
            reason = a.get("reasoning", "")
            kurl = _kalshi_url(m)
            head = f"*{_esc(title)}*"
            if kurl:
                head = f"*[{_esc(title)}]({_esc(kurl)})*"
            lines += [
                "",
                "_No bets crossed threshold\\. Best candidate from this lot:_",
                head,
                f"`{_esc(ticker)}` \u00b7 market *{last}c* \u00b7 model *{your_p}%* "
                f"\u00b7 edge *{edge}pp* \u00b7 conf *{conf}%*",
                f"\u27a4 Lean *{verdict}*",
                "",
                f"_{_esc(reason)}_",
            ]
            sources = a.get("sources") or []
            if sources:
                lines.append("")
                for i, s in enumerate(sources[:3], 1):
                    t = _esc(s.get("title", "source"))
                    u = s.get("url", "")
                    lines.append(f"{i}\\. [{t}]({_esc(u)})")
        else:
            lines += ["", "_No analyzable candidates this run\\._"]
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
