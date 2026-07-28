"""Research: turn a filled TripState into targeted queries, then into a source registry."""
from __future__ import annotations

from datetime import date

import config
import llm
import search as search_mod
from state import TripState

QUERY_SYSTEM = """You plan web searches for a travel research agent.

Given a trip brief, write 8-12 SHORT search queries that will surface concrete, current,
citable information. Good queries are specific and use words that appear on official pages.

Cover, at minimum:
- top attractions and what they actually cost / opening hours
- events or festivals during the travel window
- which neighbourhood to stay in for this budget and group
- airport-to-city transfer and local transport passes
- food matching their interests and dietary needs
- one query on safety, scams, or etiquette
- one query on entry requirements from their origin country

Rules: no query longer than 10 words. No question marks. Include the destination in every query.
Include the month and year where recency matters.

Return ONLY: {"queries": ["...", "..."]}"""


def _fallback_queries(trip: TripState) -> list[str]:
    dest = trip.destination or ""
    origin = trip.origin or ""
    when = ""
    if trip.start_date:
        try:
            d = date.fromisoformat(trip.start_date)
            when = d.strftime("%B %Y")
        except ValueError:
            when = ""
    interests = trip.interests or ["food"]
    queries = [
        f"{dest} top attractions tickets prices",
        f"{dest} official tourism board",
        f"{dest} where to stay neighbourhood guide",
        f"{dest} airport to city centre transport",
        f"{dest} public transport tourist pass",
        f"{dest} events festivals {when}".strip(),
        f"{dest} weather {when}".strip(),
        f"{dest} safety scams tourists advice",
        f"{origin} passport visa requirements {dest}".strip(),
    ]
    for topic in interests[:3]:
        queries.append(f"{dest} best {topic}")
    if trip.dietary and trip.dietary.lower() not in ("none", "no", "-"):
        queries.append(f"{dest} {trip.dietary} restaurants")
    return [q for q in queries if q.strip()]


def plan_queries(trip: TripState, model=None) -> list[str]:
    """Ask the model for queries; fall back to the template set on any failure."""
    try:
        data = llm.json_chat(
            [
                {"role": "system", "content": QUERY_SYSTEM},
                {"role": "user", "content": f"Trip brief:\n{trip.brief()}\n\nToday is {date.today().isoformat()}."},
            ],
            model=model or config.FAST_MODEL,
            temperature=config.TEMP_QUERIES,
            max_tokens=config.MAX_TOKENS_QUERIES,
        )
        queries = [q.strip() for q in data.get("queries", []) if isinstance(q, str) and q.strip()]
    except Exception as exc:  # noqa: BLE001
        print(f"[research] query planning failed, using fallback: {exc}")
        queries = []

    # Top up from templates and dedupe case-insensitively. The old version only
    # topped up below 6 and compared case-sensitively, so near-duplicates survived.
    seen = {q.lower() for q in queries}
    for q in _fallback_queries(trip):
        if len(queries) >= config.MAX_QUERIES:
            break
        if q.lower() not in seen:
            queries.append(q)
            seen.add(q.lower())

    return queries[:config.MAX_QUERIES]


def gather(trip: TripState, model=None, progress=None) -> tuple[search_mod.SourceRegistry, list[str]]:
    """Full research pass. Returns (registry, queries_used)."""
    queries = plan_queries(trip, model=model)
    registry = search_mod.SourceRegistry()
    search_mod.run_queries(queries, registry, progress=progress)
    return registry, queries
