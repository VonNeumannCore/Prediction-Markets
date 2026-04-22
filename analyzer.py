"""Ask Perplexity sonar-pro whether a Kalshi market is mispriced."""
from __future__ import annotations

import json
from typing import Optional

import requests

API_URL = "https://api.perplexity.ai/chat/completions"
MODEL = "sonar-pro"

SYSTEM_PROMPT = """You are a sharp prediction-market analyst.

For a given Kalshi market and its current YES price (in cents = implied
probability %), do the following:
1. Search the web for the most recent, authoritative evidence.
2. Form your own probability that YES resolves true.
3. The market is MISPRICED only if your probability differs from the market
   by >= 15 percentage points AND the evidence is solid.
4. Pick the side to bet (YES if you think true is underpriced, NO if overpriced).
5. Return ONLY a single JSON object, no prose, matching this schema:

{
  "mispriced": bool,
  "verdict": "YES" | "NO",
  "your_probability_pct": int 0..100,
  "confidence_pct": int 0..100,
  "reasoning": "max 2 short sentences, concrete, no fluff",
  "sources": [ {"title": str, "url": str}, ... up to 3 ranked best-first ]
}

Never invent URLs. If evidence is thin, set mispriced=false."""


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
        f"Detail: {market.get('subtitle') or ''}\n"
        f"YES means: {market.get('yes_sub_title') or ''}\n"
        f"Current YES price: {market.get('last_price')}c "
        f"(= {market.get('last_price')}% implied)\n"
        f"Closes (UTC): {market.get('close_time')}\n"
        f"Ticker: {market.get('ticker')}\n\n"
        f"Decide if this is mispriced and which side to bet."
    )

    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
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
