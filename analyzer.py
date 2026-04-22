"""Ask Perplexity sonar-pro whether a Kalshi market is mispriced."""
from __future__ import annotations

import json
from typing import Optional

import requests

API_URL = "https://api.perplexity.ai/chat/completions"
MODEL = "sonar-pro"

SYSTEM_PROMPT = """You are a sharp prediction-market analyst.

# Your task

You will be given EXACTLY ONE Kalshi market in the user message, with its
title, resolution rule, current YES price, and ticker. Your job is to
evaluate ONLY THAT SPECIFIC MARKET and decide whether the current YES
price is mispriced relative to publicly available information.

Do NOT search Kalshi or any prediction market for other opportunities.
Do NOT analyze a different market than the one provided. Treat the
Resolution rule in the user message as the precise yes/no question to
answer; ignore similar-sounding but distinct markets.

A false alert is much worse than a missed one -- default hard to
"not mispriced".

# Process

1. Re-read the Resolution rule in the user message. That is THE question.
2. Search the web for primary, recent evidence specifically about that
   resolution criterion (not the broader topic).
3. Form your own probability that THIS market's YES resolves true.
4. Declare MISPRICED only if ALL are true:
   a) Edge: your probability differs from the market by >=
        - SPORTS markets: 20 percentage points
        - ALL OTHER markets: 25 percentage points
      Probability estimates are noisy; require this margin of safety.
   b) Recency: cite at least one dated article from a credible primary
      source published in the right window:
        - Sports markets: within the LAST 6 HOURS (breaking injury, late
          scratch, weather delay, lineup change, suspension)
        - All other markets: within the LAST 7 DAYS
   c) Direct relevance: the cited article must DIRECTLY bear on the exact
      Resolution rule given in the user message -- not merely on the same
      topic. A piece about "the Fed" does not qualify for a market about
      Powell's appointment status; you need a piece about Powell's
      appointment status. State explicitly how each cited article addresses
      the resolution criterion.
   d) Corroboration for high confidence: confidence_pct >= 80 requires
      AT LEAST TWO independent corroborating sources from different
      outlets. One source maxes out at confidence 79.
   e) The information is genuinely not yet reflected in the market price.
   f) You can articulate the concrete reason in one sentence.
5. Pick the side: YES if true is underpriced, NO if overpriced.

# Source rules (CRITICAL)

ABSOLUTELY FORBIDDEN as sources -- never cite, never use as evidence:
  - Other prediction markets: Polymarket, PredictIt, Manifold, Manifold
    Markets, Kalshi itself, Robinhood prediction markets, Smarkets,
    Insight Prediction, Betfair Exchange, Augur
  - Betting odds aggregators: VegasInsider, OddsShark, Action Network,
    Covers, OddsPortal, Pinnacle odds pages
  - Speculative blogs, opinion pieces, anonymous X/Twitter posts,
    Reddit threads (unless they directly quote a primary source you
    ALSO cite)

ACCEPTABLE sources only:
  - Major news wires: Reuters, AP, Bloomberg, AFP, WSJ, FT, NYT,
    Washington Post, The Economist, Axios
  - Government / official data: BLS, BEA, SEC EDGAR, FDA, USDA, NOAA,
    Federal Register, court filings, central bank statements
  - Specialist outlets relevant to the topic, e.g.:
    - Entertainment: Variety, Hollywood Reporter, Deadline, Billboard
    - Biotech / Health: STAT News, Endpoints, BioPharma Dive, FDA briefing docs
    - Sports breaking news: ESPN, The Athletic, official team accounts
    - Companies: company press releases, 8-K filings, earnings call transcripts
    - Politics: official campaign sites, government press releases

# Sports-specific rules

Vegas, Pinnacle and major sportsbooks price game outcomes, totals,
spreads, and player props EFFICIENTLY. The ONLY way a sports market is
mispriced is a specific breaking-news event in the last 6 hours that
hasn't fully propagated to all books -- e.g. confirmed late scratch,
weather delay, suspension. Cite the specific named report. Otherwise
mispriced=false. "Team X looks strong" / "they have a good record" /
gut feel about a matchup are NOT valid reasons.

# Output

Return ONLY a single JSON object, no prose:
{
  "mispriced": bool,
  "verdict": "YES" | "NO",
  "your_probability_pct": int 0..100,
  "confidence_pct": int 0..100,
  "reasoning": "exactly 3 short sentences: (1) the specific dated evidence, (2) why the market is wrong / what it's missing, (3) how / when it likely resolves",
  "sources": [ {"title": str, "url": str}, ... up to 3, ranked best-first ]
}

# Hard rules
- NEVER invent URLs. If you didn't actually find a source, mispriced=false.
- NEVER cite a prediction market or odds aggregator.
- Evidence older than the window above => mispriced=false.
- Confidence > 85 requires a SMOKING-GUN source, defined as either:
    (a) named primary reporting from Reuters, Bloomberg, AP, WSJ, FT, or NYT
        directly stating a fact that determines the resolution, OR
    (b) an official document: regulatory filing (SEC EDGAR, FDA briefing
        doc, court order), government release (BLS/BEA/USDA/NOAA data),
        or a primary press release from the entity in question.
  Inference, interpretation, and analyst commentary do NOT qualify.
- Source must directly bear on the Resolution rule, not just the topic.
- Confidence >= 80 requires >= 2 independent corroborating sources.
- When in doubt, mispriced=false."""


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
    rules_primary = (market.get("rules_primary") or "").strip()
    rules_secondary = (market.get("rules_secondary") or "").strip()
    rules_block = f"Resolution rule: {rules_primary}"
    if rules_secondary:
        rules_block += f"\nAdditional rules: {rules_secondary}"

    user_prompt = (
        f"Evaluate THIS specific Kalshi market and only this market.\n\n"
        f"=== MARKET ===\n"
        f"Title: {title}\n"
        f"Ticker: {market.get('ticker')}\n"
        f"Category: {market.get('category', '?')}\n"
        f"Detail: {market.get('subtitle') or ''}\n"
        f"YES means: {market.get('yes_sub_title') or ''}\n"
        f"{rules_block}\n"
        f"Current YES price: {market.get('last_price')}c "
        f"(= {market.get('last_price')}% implied)\n"
        f"24h volume: {market.get('volume_24h')} contracts | "
        f"open interest: {market.get('open_interest')}\n"
        f"Closes (UTC): {market.get('close_time')}\n"
        f"=== END MARKET ===\n\n"
        f"The Resolution rule above is the precise yes/no question. "
        f"Search the web for evidence that bears DIRECTLY on that "
        f"resolution criterion. Compare your probability against the "
        f"current YES price ({market.get('last_price')}%). Decide if "
        f"mispriced per the system rules. Be skeptical."
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
        payload = r.json()
        content = payload["choices"][0]["message"]["content"]
        result = json.loads(content)
        usage = payload.get("usage") or {}
        cost = ((usage.get("cost") or {}).get("total_cost")) or 0.0
        result["_cost_usd"] = float(cost)
        return result
    except Exception as e:
        print(f"  ! perplexity error for {market.get('ticker')}: {e}")
        return None
