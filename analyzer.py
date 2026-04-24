"""Per-event Kalshi 'mentions' analysis via Perplexity sonar-pro.

Sends ONE event (with its full live word table) per Perplexity call and
returns a JSON object {event_brief, edges} where each edge is a single
mispriced word the model believes is over- or under-priced by the crowd.
"""
from __future__ import annotations

import json
from typing import Optional

import requests

API_URL = "https://api.perplexity.ai/chat/completions"
MODEL = "sonar-pro"

EDGE_THRESHOLD_PP = 10
CONFIDENCE_THRESHOLD = 70
MAX_EDGES_PER_EVENT = 5
WIDE_SPREAD_CENTS = 5  # only show "executable" side when spread > this


SYSTEM_PROMPT = """You are a careful prediction-market analyst working on
Kalshi "mentions" markets. Each market in this category resolves YES if a
specific person (or entity) says a specific word/phrase during a defined
upcoming appearance, interview, or address. You are given EXACTLY ONE event
in the user message and must rank which of its candidate words are mispriced
relative to publicly available information.

# What you receive

For one event:
- Event title, ticker, expiry, settlement rules text.
- A description of the upcoming appearance / interview / address.
- A table of LIVE candidate words/phrases (already-resolved early-close
  contracts have been filtered out). Each row has:
    - per-word ticker
    - YES last_price (cents = market-implied probability %)
    - YES bid / ask (cents) and NO ask (cents)
    - 24h volume and open interest

# Process

1. Identify who the subject is, what they do, and in what capacity they
   are speaking.
2. Identify what the upcoming event/interview/address is about. Use the
   settlement rules to determine which sources/channels count for
   settlement.
3. Search the web for similar PUBLIC appearances by the subject in the
   LAST 7 DAYS, prioritizing appearances that match the same event type
   or speaking context. For each appearance gather: date, source/outlet,
   event type/purpose, and direct quote/transcript evidence of whether
   each candidate word was said.
4. For each candidate word, estimate the true probability the subject will
   say that word during the qualifying event. Weigh:
     - subject's recent language patterns and topical fixations
     - the purpose of the upcoming event
     - current news context driving the subject's attention
     - the named eligible sources for settlement
     - exactness of the word/phrase requirement (Kalshi treats plurals and
       possessives as equivalent; OTHER inflections, tense changes,
       hyphenated compounds, and synonyms do NOT count)
5. Pick a side for each word: YES if true is underpriced, NO if overpriced.
6. Compute edge against the EXECUTABLE side, not the last price:
     - If side == YES: executable_pct = YES ask
     - If side == NO:  executable_pct = 100 - YES bid (== NO ask)
   Edge_pp = abs(model_pct_for_chosen_side - executable_pct).
   For NO bets, model_pct_for_chosen_side = 100 - model_pct_yes.
7. Return only words where:
     edge_pp >= 10 percentage points  AND  confidence_pct >= 70
8. Rank by absolute mispricing (largest edge first). At most 5 edges.

# Hard rules (CRITICAL)

- Do NOT predict YES on a word merely because it is topical. You must
  cite a recent dated source where the subject said that EXACT word/phrase
  recently, or show a clear pattern of doing so in this exact event type.
- ACCEPTABLE evidence sources for prior usage:
    a) the subject's own Twitter/X or Truth Social posts
    b) major wires and papers quoting the subject directly: Reuters, AP,
       Bloomberg, WSJ, FT, NYT, Washington Post, Axios, The Economist
    c) primary transcripts (C-SPAN, Federal Register, court filings,
       official press releases, earnings call transcripts, Hansard)
    d) topic-specialist outlets when they directly quote the subject
       (Variety/Hollywood Reporter for entertainment; STAT/Endpoints for
       biotech; ESPN/The Athletic for breaking sports news)
- ABSOLUTELY FORBIDDEN as evidence -- never cite, never use:
    - other prediction markets: Polymarket, PredictIt, Manifold, Kalshi
      itself, Robinhood prediction markets, Smarkets, Insight Prediction
    - betting aggregators: VegasInsider, OddsShark, Action Network,
      Pinnacle odds pages
    - anonymous social posts, opinion blogs, Reddit threads (unless they
      directly quote a primary source you ALSO cite separately)
- If you cannot cite a dated source within the last 7 days bearing on
  the subject's likely use of a word, set that word's confidence_pct < 60
  (which will exclude it from output).
- Never invent URLs. If you didn't actually find a source, omit the word
  rather than fabricating a citation.
- A false alert is much worse than a missed one. When in doubt, exclude.

# Output format

Return ONE JSON object EXACTLY matching the provided schema. No prose
outside the JSON. Each edge object MUST include:
  ticker, word, model_pct (int 0..100, the YES-side probability you
  estimate), side ("YES"|"NO"), confidence_pct (int 0..100),
  reason (one short sentence naming the dated source), evidence_url
  (the single best supporting URL, or "" if none).

If no words clear the threshold, return:
  {"event_brief": "<2-3 sentences>", "edges": []}
"""


JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "event_brief": {"type": "string"},
        "edges": {
            "type": "array",
            "maxItems": MAX_EDGES_PER_EVENT,
            "items": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "word": {"type": "string"},
                    "model_pct": {
                        "type": "integer", "minimum": 0, "maximum": 100,
                    },
                    "side": {"type": "string", "enum": ["YES", "NO"]},
                    "confidence_pct": {
                        "type": "integer", "minimum": 0, "maximum": 100,
                    },
                    "reason": {"type": "string"},
                    "evidence_url": {"type": "string"},
                },
                "required": [
                    "ticker", "word", "model_pct", "side",
                    "confidence_pct", "reason", "evidence_url",
                ],
            },
        },
    },
    "required": ["event_brief", "edges"],
}


def _build_word_table(siblings: list[dict]) -> str:
    """Fixed-width table the LLM can read at a glance."""
    header = (
        f"{'TICKER':40s}  {'WORD':40s}  "
        f"{'LAST':>4s}  {'YBID':>4s}  {'YASK':>4s}  {'NASK':>4s}  "
        f"{'VOL24':>7s}  {'OI':>7s}"
    )
    lines = [header, "-" * len(header)]
    for m in sorted(siblings, key=lambda x: -int(x.get("volume_24h") or 0)):
        word = (m.get("word") or "")[:40]
        lines.append(
            f"{m['ticker']:40s}  {word:40s}  "
            f"{int(m.get('last_price') or 0):>4d}  "
            f"{int(m.get('yes_bid') or 0):>4d}  "
            f"{int(m.get('yes_ask') or 0):>4d}  "
            f"{int(m.get('no_ask') or 0):>4d}  "
            f"{int(m.get('volume_24h') or 0):>7d}  "
            f"{int(m.get('open_interest') or 0):>7d}"
        )
    return "\n".join(lines)


def _build_user_prompt(event: dict, siblings: list[dict]) -> str:
    title = event.get("title", "?")
    ev_ticker = event.get("event_ticker", "?")
    series = event.get("series_ticker", "")
    sub_title = event.get("sub_title", "")
    # rules_secondary is identical across siblings -- one canonical copy.
    sample = siblings[0] if siblings else {}
    rules_secondary = (sample.get("rules_secondary") or "").strip()
    early_close = (sample.get("early_close_condition") or "").strip()
    expiry = (sample.get("close_time") or "")[:16].replace("T", " ")
    # Replace the literal candidate word in rules_primary with <WORD> so
    # the model sees an explicit per-word template, not a rule that
    # mentions only whichever sibling happened to sort first.
    rp = (sample.get("rules_primary") or "").strip()
    sample_word = sample.get("word") or ""
    if sample_word and sample_word in rp:
        rules_primary_template = rp.replace(sample_word, "<WORD>")
    else:
        rules_primary_template = rp

    return (
        "EVENT\n=====\n"
        f"Title:        {title}\n"
        f"Event ticker: {ev_ticker}\n"
        f"Series:       {series}\n"
        f"Sub-title:    {sub_title}\n"
        f"Expiry (UTC): {expiry}\n\n"
        "Resolution rule (per-word template, applies to every candidate "
        "in the table; substitute <WORD> with the candidate word):\n"
        f"  {rules_primary_template}\n\n"
        "Settlement source rules (apply to all candidate words):\n"
        f"  {rules_secondary}\n\n"
        f"Early-close: {early_close}\n\n"
        "CANDIDATE WORDS  (already-resolved >95c contracts removed)\n"
        "==========================================================\n"
        f"{_build_word_table(siblings)}\n\n"
        "Identify the subject from the event title above. Search for the "
        "subject's recent public appearances and apply the system-prompt "
        "rules. Return JSON only."
    )


def _executable_pct(market: dict, side: str) -> int:
    """Probability the executable side of the trade implies."""
    if side == "YES":
        return int(market.get("yes_ask") or market.get("last_price") or 0)
    bid = int(market.get("yes_bid") or market.get("last_price") or 0)
    return 100 - bid


def _last_pct(market: dict) -> int:
    return int(market.get("last_price") or 0)


def _spread(market: dict) -> int:
    bid = int(market.get("yes_bid") or 0)
    ask = int(market.get("yes_ask") or 0)
    return max(0, ask - bid)


def post_filter_edges(edges: list[dict], siblings: list[dict]) -> list[dict]:
    """Server-side enforcement of the same rules the prompt states.

    The LLM is asked to gate on edge >= 10pp vs executable side AND
    confidence >= 70. We re-check here so a sloppy or fabricating model
    can't sneak a weak bet through. Each surviving edge is decorated with
    last_pct, executable_pct, spread, edge_pp, and the linked sibling
    market dict so the renderer has everything it needs.
    """
    by_ticker = {m["ticker"]: m for m in siblings}
    out: list[dict] = []
    for e in edges:
        m = by_ticker.get(e.get("ticker"))
        if m is None:
            continue
        side = e.get("side")
        if side not in ("YES", "NO"):
            continue
        try:
            model_yes = int(e.get("model_pct"))
            conf = int(e.get("confidence_pct"))
        except (TypeError, ValueError):
            continue
        if conf < CONFIDENCE_THRESHOLD:
            continue
        last = _last_pct(m)
        execp = _executable_pct(m, side)
        # model probability for the SIDE we're betting
        model_side = model_yes if side == "YES" else 100 - model_yes
        edge_pp = abs(model_side - execp)
        if edge_pp < EDGE_THRESHOLD_PP:
            continue
        e2 = dict(e)
        e2["market"] = m
        e2["last_pct"] = last
        e2["executable_pct"] = execp
        e2["spread"] = _spread(m)
        e2["edge_pp"] = edge_pp
        e2["model_pct_side"] = model_side
        out.append(e2)
    out.sort(key=lambda x: -x["edge_pp"])
    return out[:MAX_EDGES_PER_EVENT]


def analyze_event(
    event: dict,
    siblings: list[dict],
    api_key: str,
    *,
    timeout: int = 120,
) -> Optional[dict]:
    """Run one Perplexity call for the whole event. Returns
    {event_brief, edges (post-filtered + decorated), _cost_usd, _model}
    or None on hard failure."""
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(event, siblings)},
        ],
        "temperature": 0.1,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"schema": JSON_SCHEMA},
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        r = requests.post(API_URL, json=body, headers=headers, timeout=timeout)
        r.raise_for_status()
        payload = r.json()
        content = payload["choices"][0]["message"]["content"]
        result = json.loads(content)
    except Exception as e:
        print(f"  ! perplexity error for {event.get('event_ticker')}: {e}")
        return None

    raw_edges = result.get("edges") or []
    filtered = post_filter_edges(raw_edges, siblings)

    usage = payload.get("usage") or {}
    cost = ((usage.get("cost") or {}).get("total_cost")) or 0.0
    return {
        "event_brief": (result.get("event_brief") or "").strip(),
        "edges": filtered,
        "raw_edge_count": len(raw_edges),
        "_cost_usd": float(cost),
        "_model": payload.get("model") or MODEL,
    }
