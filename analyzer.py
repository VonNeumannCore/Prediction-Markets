"""Per-event Kalshi 'mentions' analysis via Perplexity sonar-pro.

Sends ONE event (with its full live word table) per Perplexity call and
returns a JSON object {event_brief, edges} where each edge is a single
mispriced word the model believes is over- or under-priced by the crowd.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

import requests

API_URL = "https://api.perplexity.ai/chat/completions"
MODEL = "sonar-pro"

EDGE_THRESHOLD_PP = 10
CONFIDENCE_THRESHOLD = 70
MAX_EDGES_PER_EVENT = 5
WIDE_SPREAD_CENTS = 5  # only show "executable" side when spread > this


SYSTEM_PROMPT = """You are running a SHORT-HORIZON LINGUISTIC ARBITRAGE
engine on Kalshi "mentions" markets, NOT a generic forecasting engine.
You are NOT trying to out-predict the future better than the crowd. Your
edge comes from THREE specific arbitrages that the crowd consistently
misprices:

  1. SETTLEMENT-RULE arbitrage -- the crowd misreads what counts as a
     "say" (which channels qualify, exact phrase vs synonyms, time window
     bounds). Read the rules text precisely; small-print wins money.
  2. RECENCY-BIAS arbitrage -- the crowd overweights last week's hot
     phrase and underweights phrases that have cooled. A nickname (e.g.
     "Sleepy Joe") tied to a foil who is no longer in the news cycle
     should be priced near zero, even if the subject "always" used it.
  3. EVENT-SURFACE arbitrage -- the crowd doesn't know the subject's
     actual calendar in the window. A word that requires a scheduled
     speaking opportunity which doesn't exist in the window is overpriced;
     a word that maps cleanly to a known scheduled appearance (debate,
     hearing, FOMC, rally) is underpriced.

Frame every reasoning step around which of these three arbitrages, if
any, is in play. If none of the three applies to a given word, you have
no edge -- omit the word.

Each market resolves YES if a specific person (or entity) says a specific
word/phrase across qualifying sources during a defined upcoming WINDOW.
You are given EXACTLY ONE event in the user message and must rank which
of its candidate words are mispriced relative to publicly available
information.

# What you receive

For one event:
- Event title, ticker, expiry, settlement rules text.
- An explicit WINDOW block (today, expiry, hours remaining) and the
  qualifying SURFACE (channels that count for settlement).
- A table of LIVE candidate words/phrases (already-resolved early-close
  contracts have been filtered out). Each row has:
    - per-word ticker
    - YES last_price (cents = market-implied probability %)
    - YES bid / ask (cents) and NO ask (cents)
    - 24h volume and open interest

# How to think (CRITICAL -- the most common mistake here is to treat
"did the subject say this word in their last appearance?" as a substitute
for "will the subject say this word at least once across the qualifying
surface during the remaining window?". DO NOT do that.)

For EACH candidate word, walk this 5-stage chain explicitly:

(a) WINDOW. State the window in days/hours remaining. The probability you
    output is the chance the word is uttered AT LEAST ONCE across the
    qualifying surface between now and expiry -- not the chance it appears
    in any one specific event.

(b) PAST REGIME (last 7 days). Identify the 1-3 dominant TOPICS the
    subject has actually been speaking/posting about. Cite dated examples.
    For each topic, decide: still hot and likely to carry forward, or
    already cooled?

(c) CURRENT / FORWARD REGIME (now -> expiry). Identify the 1-3 topics
    that will most likely drive the subject's speech in the remaining
    window. Sources of forward signal:
      - breaking news of the day,
      - the subject's own pending announcements/legal actions/scheduled
        meetings,
      - holdovers from the past regime that are still active.

(d) MARKET STRUCTURE + CALENDAR. First classify the market:

    EVENT-ANCHORED: tied to ONE specific scheduled appearance (FOMC
    presser, Senate hearing, debate, single named interview, earnings
    call, Inaugural address, etc.). For these:
      - Identify the event by name, scheduled time, host/moderator,
        format (prepared remarks vs Q&A vs adversarial).
      - Note who controls the topic: prepared script (subject) vs Q&A
        (reporters/analysts/senators) vs hostile interview (host).
      - For RECURRING instances (weekly press briefing, monthly FOMC,
        Sunday show appearance), pull the MOST RECENT prior instance of
        the same event type as a TEMPLATE -- what topics actually came
        up there is your best forecast for what comes up here.

    WINDOW-ANCHORED: covers the subject's full output across N days
    with NO single anchor event ("What will Trump say this week?",
    "How will X reference themselves before July?"). For these:
      - Enumerate the 2-3 most likely qualifying speaking surfaces in
        the window (Truth/X cadence, scheduled press confs, rally
        calendar, recurring weekly briefings, scheduled interviews).
      - Pull each surface's recent template: "Trump's last 3 Truth
        posts were about X, Y, Z" or "his last rally hit themes A, B".
      - The model probability is the chance the word surfaces AT LEAST
        ONCE across ALL of these surfaces combined, not at any one.

    For each surface, note WHO controls the topic:
      - Truth/X posts        -> 100% subject-driven (whatever angered or
                                excited him that hour)
      - Press conf / gaggle  -> reporters drive; they ask about CURRENT
                                news cycle and his recent announcements
      - Rally / stump speech -> subject-driven, recurring stump themes
      - Earnings call        -> CEO/CFO script + analyst Q&A on the quarter
      - Interview            -> host steers; depends on the host
    This determines which regime topics actually surface where.

(e) MATCH. For the candidate word, evaluate the intersection
    (current regime) AND (event-type surface). Decide:
      - matches current regime AND a window event will surface it
        -> high probability (often 70-95%)
      - matches current regime BUT no qualifying event will surface it
        -> low / moderate
      - matches PAST regime that has cooled -> low
      - stump-speech / pet-phrase word (e.g. "Sleepy Joe", "Drill Baby
        Drill" for Trump): a word the subject HISTORICALLY uses
        constantly does NOT count as currently active. To assign
        probability > 50% you MUST cite a dated source from the LAST 7
        DAYS containing the EXACT word/phrase. If no such source exists,
        treat the word as cooled regardless of long-run habit -- nicknames
        especially go stale when the foil leaves the news cycle (e.g.
        "Sleepy Joe" goes cold once Biden stops being the main target).
      - exactness mismatch (Pope vs Pontiff; "Sleepy Joe" vs "Joe")
        -> low; Kalshi treats plurals/possessives as equivalent but NOT
        other inflections, hyphenations, or synonyms

# Source rules

ACCEPTABLE evidence:
  a) the subject's own Twitter/X or Truth Social posts (dated)
  b) major wires/papers quoting the subject directly: Reuters, AP,
     Bloomberg, WSJ, FT, NYT, Washington Post, Axios, The Economist
  c) primary transcripts (C-SPAN, Federal Register, court filings,
     official press releases, earnings call transcripts, Hansard)
  d) topic-specialist outlets when they directly quote the subject
     (Variety/Hollywood Reporter for entertainment; STAT/Endpoints for
     biotech; ESPN/The Athletic for breaking sports news)

ABSOLUTELY FORBIDDEN as evidence:
  - other prediction markets: Polymarket, PredictIt, Manifold, Kalshi
    itself, Robinhood prediction markets, Smarkets, Insight Prediction
  - betting aggregators: VegasInsider, OddsShark, Action Network,
    Pinnacle odds pages
  - anonymous social posts, opinion blogs, Reddit threads (unless they
    directly quote a primary source you ALSO cite separately)

Never invent URLs. If you didn't actually find a source for the regime
or the pattern, omit the word rather than fabricate.

The `evidence_url` field MUST be a real http(s):// URL pointing directly
at the dated source. Bare citation markers like "[1]", "[2]", footnote
numbers, source names without URLs, or empty strings will be REJECTED
by the post-filter and the word dropped. If you cannot produce a real
URL for a candidate, omit that candidate entirely.

# Confidence

confidence_pct should reflect the STRENGTH of the regime-event-word
match, not just how many sources you cited. A clear current-regime match
backed by 1-2 dated sources (one Truth post + one Reuters quote) beats a
vague topical guess with 5 weak sources.

If you cannot identify a current-regime driver AND a window event-surface
that plausibly produces this word, set confidence_pct < 60 (which will
exclude the word from output).

# Edge math

Compute edge against the EXECUTABLE side, not the last price:
  - side YES -> executable_pct = YES ask
  - side NO  -> executable_pct = 100 - YES bid (== NO ask)
  - For NO bets, model_pct_for_chosen_side = 100 - model_pct_yes
  - edge_pp = abs(model_pct_for_chosen_side - executable_pct)

Return only words where: edge_pp >= 10  AND  confidence_pct >= 70
Rank by edge_pp descending. Maximum 5 edges per event.

# Output format

Return ONE JSON object EXACTLY matching the provided schema. No prose
outside the JSON. Each edge object MUST include:
  ticker, word, model_pct (int 0..100, the YES-side probability),
  side ("YES"|"NO"), confidence_pct (int 0..100), reason, evidence_url.

The `reason` field MUST be ONE tight sentence that explicitly names:
  (i)   the current regime / news driver in the window,
  (ii)  a SPECIFIC SCHEDULED EVENT OR RECURRING SURFACE in the window
        where the word will (or won't) surface -- not "possibly a press
        gaggle" but a NAMED instance: "Friday 2pm WH gaggle on Iran",
        "Saturday rally in Pittsburgh", "Sunday Meet-the-Press hit",
        "the next Truth post about the pending NYAG ruling", "the FOMC
        presser at 2:30pm Wednesday", etc.,
  (iii) which of the three arbitrages (settlement, recency, event-
        surface) is in play and why this specific word fits or doesn't
        fit that intersection.

GOOD reason examples:
  "NY AG action on 4/23 dragged Trump Tower back into the cycle; he has
  a Friday 2pm WH gaggle where reporters will ask about the appeal;
  event-surface arbitrage -- he reflexively defends Trump Tower by name
  in every legal-news cycle gaggle, last 3 instances confirm."

  "Past regime in early April was Iran/Hormuz, cooled after 4/22
  de-escalation; no scheduled interview or rally in the 64-hour window
  where airport policy would come up; recency arb -- crowd is still
  pricing the cooled Iran cycle, last 5 Truth posts have been NY-legal
  and DC-arch focused with zero aviation."

BAD reason examples (do NOT produce these):
  "No mentions of Trump Tower in recent healthcare events."
  "Trump frequently mentions his properties."
  "Topical."
  "He said it last week."
  "Possibly a press gaggle."           <- not specific enough
  "High likelihood of Truth posts."    <- name the topic + driver

If no words clear the threshold, return:
  {"event_brief": "<2-3 sentences. Sentence 1: market structure
    (event-anchored vs window-anchored) + the specific anchor event or
    enumerated surfaces in the window. Sentence 2: current regime / news
    driver. Optional sentence 3: why no word in this list is mispriced
    given that intersection.>", "edges": []}

IMPORTANT: words still in the candidate table are by definition NOT yet
resolved -- the scanner has filtered out anything already said. Do not
assume a word has been uttered without dated evidence.
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


def _build_window_block(siblings: list[dict]) -> str:
    """Compute and render the WINDOW + qualifying SURFACE block. The model
    needs the explicit time horizon front-and-center so it doesn't collapse
    'will the word be uttered across the surface in the window' into 'did
    the subject say it in their last appearance'."""
    now = datetime.now(timezone.utc)
    # All live siblings in a mentions event share a single close_time
    # (the resolved early-close ones have already been filtered out).
    close_iso = (siblings[0].get("close_time") if siblings else "") or ""
    try:
        close_dt = datetime.fromisoformat(close_iso.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        close_dt = None

    if close_dt is not None:
        delta = close_dt - now
        hours = max(0, int(delta.total_seconds() // 3600))
        days = hours / 24.0
        close_str = close_dt.strftime("%Y-%m-%d %H:%M")
    else:
        hours, days, close_str = 0, 0.0, "?"

    return (
        "WINDOW\n"
        "======\n"
        f"TODAY (UTC):     {now.strftime('%Y-%m-%d %H:%M')}\n"
        f"Closes (UTC):    {close_str}\n"
        f"Hours remaining: {hours}  (~{days:.1f} days)\n"
        "\n"
        "DATE ANCHORING (CRITICAL): anything you 'know' about the\n"
        "subject's behavior from BEFORE today is HISTORICAL BASELINE\n"
        "ONLY -- it does NOT count as current evidence. To support a\n"
        "probability > 50% on any word you must cite a dated source\n"
        "from within the LAST 7 DAYS containing that exact word/phrase\n"
        "(or clear pattern-of-life evidence from the same window).\n"
        "Long-run habits (e.g. 'Sleepy Joe', 'Drill Baby Drill') do\n"
        "NOT carry without recent dated confirmation -- nicknames go\n"
        "stale when the foil leaves the news cycle.\n"
    )


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
        "Settlement source rules -- this defines the QUALIFYING SURFACE; "
        "every channel listed here counts as an opportunity for the word "
        "to be uttered during the window:\n"
        f"  {rules_secondary}\n\n"
        f"Early-close: {early_close}\n\n"
        f"{_build_window_block(siblings)}\n"
        "CANDIDATE WORDS  (already-resolved >95c contracts removed)\n"
        "==========================================================\n"
        f"{_build_word_table(siblings)}\n\n"
        "Identify the subject from the event title above. For each candidate "
        "word, walk the 5-stage chain (window -> past regime -> current "
        "regime -> event-type surface -> match) before assigning a "
        "probability. The `reason` field MUST name the current regime, the "
        "event-type opportunity in the window, and why the word fits or "
        "doesn't fit. Return JSON only."
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


def _is_real_url(s: str) -> bool:
    """A 'real' evidence URL must be http(s) and have a host with a dot.

    Rejects: empty strings, bare citation markers like '[2]', source
    names without a URL ('Reuters'), and obviously fake stems. The
    model frequently puts Perplexity's internal citation footnote
    numbers here ('[1]', '[2]') -- we treat that as no evidence and
    drop the edge."""
    if not isinstance(s, str):
        return False
    s = s.strip()
    if not (s.startswith("http://") or s.startswith("https://")):
        return False
    rest = s.split("://", 1)[1]
    host = rest.split("/", 1)[0]
    return "." in host and len(host) >= 4


def post_filter_edges(edges: list[dict], siblings: list[dict]) -> list[dict]:
    """Server-side enforcement of the same rules the prompt states.

    The LLM is asked to gate on edge >= 10pp vs executable side AND
    confidence >= 70 AND a real http(s) evidence URL. We re-check all
    three here so a sloppy or fabricating model can't sneak a weak bet
    through. Each surviving edge is decorated with last_pct,
    executable_pct, spread, edge_pp, and the linked sibling market dict
    so the renderer has everything it needs.
    """
    by_ticker = {m["ticker"]: m for m in siblings}
    out: list[dict] = []
    rejected = {"unknown_ticker": 0, "bad_side": 0, "bad_int": 0,
                "low_conf": 0, "low_edge": 0, "no_url": 0}
    for e in edges:
        m = by_ticker.get(e.get("ticker"))
        if m is None:
            rejected["unknown_ticker"] += 1
            continue
        side = e.get("side")
        if side not in ("YES", "NO"):
            rejected["bad_side"] += 1
            continue
        try:
            model_yes = int(e.get("model_pct"))
            conf = int(e.get("confidence_pct"))
        except (TypeError, ValueError):
            rejected["bad_int"] += 1
            continue
        if conf < CONFIDENCE_THRESHOLD:
            rejected["low_conf"] += 1
            continue
        if not _is_real_url(e.get("evidence_url", "")):
            # No verifiable source -> drop. The model will often emit
            # bare '[2]' citation markers; treat those as no evidence.
            rejected["no_url"] += 1
            continue
        last = _last_pct(m)
        execp = _executable_pct(m, side)
        # model probability for the SIDE we're betting
        model_side = model_yes if side == "YES" else 100 - model_yes
        edge_pp = abs(model_side - execp)
        if edge_pp < EDGE_THRESHOLD_PP:
            rejected["low_edge"] += 1
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
    if any(rejected.values()):
        # Surface in stdout so we can see WHY edges got dropped.
        rej = ", ".join(f"{k}={v}" for k, v in rejected.items() if v)
        print(f"    post-filter rejected: {rej}")
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
