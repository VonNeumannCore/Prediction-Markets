"""Ask Perplexity sonar-pro whether a Kalshi market is mispriced."""
from __future__ import annotations

import json
from typing import Optional

import requests

API_URL = "https://api.perplexity.ai/chat/completions"
MODEL = "sonar-pro"

SYSTEM_PROMPT = """You are a sharp prediction-market analyst. Be skeptical
and default to "not mispriced" unless evidence is strong.

For a given Kalshi market and its current YES price (in cents = implied
probability %), do this:

1. Search the web for recent, authoritative evidence about the question.
2. Form your own probability that YES resolves true.
3. Declare MISPRICED only if ALL of these are true:
   a) Your probability differs from the market by >= 20 percentage points.
   b) You can cite at least one specific dated article from the last 7 days
      that meaningfully shifts the question.
   c) The source is a credible outlet (major news, official filings,
      government data, well-known specialist publications).
   d) You can articulate WHY the market is wrong in one concrete sentence.
4. Pick the side to bet (YES if true is underpriced, NO if overpriced).
5. Return ONLY a single JSON object, no prose, matching this schema:

{
  "mispriced": bool,
  "verdict": "YES" | "NO",
  "your_probability_pct": int 0..100,
  "confidence_pct": int 0..100,
  "reasoning": "max 2 short sentences, must reference the evidence",
  "sources": [ {"title": str, "url": str}, ... up to 3, ranked best-first ]
}

Hard rules:
- NEVER invent URLs. If you didn't actually find a source, set mispriced=false.
- If your evidence is older than 7 days, set mispriced=false.
- If the question is a sports outcome, set mispriced=false (sportsbooks beat us).
- Confidence above 80 requires a smoking-gun source, not just "seems likely".
- When in doubt, set mispriced=false. The cost of a false alert is high."""


JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "mispriced": {"type": "boolean"},
        "verdict": {"type": "string", "enum": ["YES", "NO"]},
        "your_probability_pct": {"type": "integer", "minimum": 0, "maximum": 100},
        "confidence_pct": {"type": "integer", "minimum": 0, "maximum": 100},
        "reasoning": {"type": "string"},
        "sources": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "url": {"type": "string"},
                },
                "required": ["title", "url"],
            },
        },
    },
    "required": [
        "mispriced", "verdict", "your_probability_pct",
        "confidence_pct", "reasoning", "sources",
    ],
}


def analyze(market: dict, api_key: str, timeout: int = 60) -> Optional[dict]:
    title = market.get("title") or market.get("ticker")
    user_prompt = (
        f"Kalshi market: {title}\n"
        f"Category: {market.get('category', '?')}\n"
        f"Detail: {market.get('subtitle') or ''}\n"
        f"YES means: {market.get('yes_sub_title') or ''}\n"
        f"Current YES price: {market.get('last_price')}c "
        f"(= {market.get('last_price')}% implied)\n"
        f"24h volume: {market.get('volume_24h')} contracts | "
        f"open interest: {market.get('open_interest')}\n"
        f"Closes (UTC): {market.get('close_time')}\n"
        f"Ticker: {market.get('ticker')}\n\n"
        f"Decide if mispriced. Be skeptical."
    )

    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
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
        content = r.json()["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        print(f"  ! perplexity error for {market.get('ticker')}: {e}")
        return None
