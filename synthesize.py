"""Synthesis: retrieved sources + trip brief -> structured itinerary JSON.

Hard rule enforced here: the model never emits a URL. It cites [S<id>] tokens and we
resolve them against the registry. Anything that looks like a raw URL is stripped.
"""
from __future__ import annotations

import re
from datetime import date

import config
import llm
from search import SourceRegistry
from state import TripState

SYSTEM = """You are the planning half of a travel agent. You write a concrete, realistic itinerary
using ONLY the numbered sources supplied to you.

ABSOLUTE RULES
1. Never write a URL, domain name, or link of any kind. Not one. Cite sources as ["S3", "S7"] in
   the `sources` array of the relevant object. The application resolves those to real links.
2. Every factual claim — an opening time, a price, a transport fare, an event date — must carry at
   least one source id. If no source supports it, either omit it or phrase it as a general
   suggestion with an empty sources array.
3. Do not invent hotel names, restaurant names, or prices that no source mentions. If sources are
   thin on a category, say so in `unknowns` rather than filling the gap with plausible fiction.

PLANNING QUALITY
- Group each day geographically. Do not bounce across the city and back.
- Respect the stated pace: relaxed = 2-3 anchors per day, balanced = 3-4, packed = 5+.
- Account for travel time between blocks and for jet lag on day 1 if the origin is far.
- Match food suggestions to the stated dietary needs and interests.
- Costs in the destination's local currency, with a per-person daily total.
- Be specific about neighbourhoods, not just "the city centre".

Return ONLY this JSON:
{
  "title": "string",
  "overview": "2-3 sentence framing of the trip",
  "best_time_note": {"text": "string", "sources": ["S1"]},
  "days": [
    {
      "day": 1,
      "theme": "short label",
      "area": "neighbourhood/district focus",
      "blocks": [
        {
          "time": "Morning | Midday | Afternoon | Evening",
          "activity": "what they do",
          "place": "the specific named place, for map lookup",
          "duration": "e.g. 2h",
          "cost": "e.g. JPY 1000 pp or Free",
          "why": "one line on why this fits them",
          "sources": ["S2"]
        }
      ],
      "food": [{"suggestion": "string", "place": "named place or area", "sources": []}],
      "notes": "logistics note for the day"
    }
  ],
  "stays": [{"area": "neighbourhood", "who_its_for": "string", "price_band": "string", "examples": "named properties only if a source names them", "sources": []}],
  "transport": [{"option": "string", "cost": "string", "tip": "string", "sources": []}],
  "budget": {
    "currency": "string",
    "per_day": [{"category": "Accommodation|Food|Transport|Activities|Misc", "amount": "string", "notes": "string"}],
    "trip_total_estimate": "string",
    "assumptions": "string"
  },
  "tips": [{"tip": "string", "sources": []}],
  "unknowns": ["things you could not verify from the sources and the user should confirm"]
}"""

_URL_RE = re.compile(r"https?://\S+|www\.\S+|\b[\w-]+\.(?:com|net|org|io|in|jp|co\.uk)\b\S*", re.I)
_SID_RE = re.compile(r"\bS\d+\b")


def _scrub(text: str) -> str:
    """Remove any URL the model snuck into prose."""
    if not isinstance(text, str):
        return text
    return _URL_RE.sub("", text).replace("  ", " ").strip()


def _clean_sources(values, registry: SourceRegistry) -> list[str]:
    out = []
    for v in values or []:
        if not isinstance(v, str):
            continue
        for sid in _SID_RE.findall(v.upper()):
            if registry.get(sid) and sid not in out:
                out.append(sid)
    return out


def _walk(node, registry: SourceRegistry):
    """Recursively scrub prose and validate every `sources` array."""
    if isinstance(node, dict):
        cleaned = {}
        for k, v in node.items():
            if k == "sources":
                cleaned[k] = _clean_sources(v, registry)
            else:
                cleaned[k] = _walk(v, registry)
        return cleaned
    if isinstance(node, list):
        return [_walk(v, registry) for v in node]
    if isinstance(node, str):
        return _scrub(node)
    return node


def build_itinerary(trip: TripState, registry: SourceRegistry) -> dict:
    context = registry.context_block(limit=55)
    user = (
        f"TRIP BRIEF\n{trip.brief()}\n\n"
        f"Today's date: {date.today().isoformat()}\n\n"
        f"SOURCES (cite these ids only)\n{context}\n\n"
        f"Write the {trip.duration_days}-day itinerary now."
    )
    data = llm.json_chat(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        temperature=config.TEMP_PLAN,
        max_tokens=config.MAX_TOKENS_PLAN,
    )
    return _walk(data, registry)


REFINE_SYSTEM = """You revise an existing travel itinerary JSON based on the user's request.
Same absolute rules as before: never write a URL, cite only the supplied source ids.
Change only what the request touches; leave everything else byte-identical.
Return the COMPLETE updated JSON object in the same schema."""


def refine_itinerary(itinerary: dict, trip: TripState, registry: SourceRegistry, request: str) -> dict:
    import json

    user = (
        f"CURRENT ITINERARY\n{json.dumps(itinerary, ensure_ascii=False)}\n\n"
        f"TRIP BRIEF\n{trip.brief()}\n\n"
        f"SOURCES\n{registry.context_block(limit=55)}\n\n"
        f"USER REQUEST: {request}"
    )
    data = llm.json_chat(
        [{"role": "system", "content": REFINE_SYSTEM}, {"role": "user", "content": user}],
        temperature=0.4,
        max_tokens=config.MAX_TOKENS_PLAN,
    )
    return _walk(data, registry)


def coverage_report(itinerary: dict, registry: SourceRegistry) -> dict:
    """How much of the plan is actually sourced. Useful as a trust signal in the UI."""
    total_claims = 0
    sourced = 0

    def visit(node):
        nonlocal total_claims, sourced
        if isinstance(node, dict):
            if "sources" in node:
                total_claims += 1
                if node["sources"]:
                    sourced += 1
            for v in node.values():
                visit(v)
        elif isinstance(node, list):
            for v in node:
                visit(v)

    visit(itinerary)
    used = set()

    def collect(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "sources":
                    used.update(v or [])
                else:
                    collect(v)
        elif isinstance(node, list):
            for v in node:
                collect(v)

    collect(itinerary)
    return {
        "claims": total_claims,
        "sourced": sourced,
        "pct": round(100 * sourced / total_claims) if total_claims else 0,
        "sources_used": len(used),
        "sources_retrieved": len(registry),
    }
