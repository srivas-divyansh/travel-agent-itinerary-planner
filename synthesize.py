"""Synthesis: retrieved sources + trip brief -> structured itinerary JSON.

Hard rule enforced here: the model never emits a URL. It cites [S<id>] tokens and we
resolve them against the registry. Anything that looks like a raw URL is stripped.
"""
from __future__ import annotations

import json
import re
import time
from datetime import date

from typing import Any

import config
import llm
import search
from llm import est_tokens
from search import SourceRegistry
from state import TripState

_RULES = """ABSOLUTE RULES
1. Never write a URL, domain name, or link of any kind. Cite sources as ["S3","S7"] in the
   `sources` array of the relevant object. The application resolves those to real links.
2. Every factual claim — an opening time, a price, a fare, an event date — must carry at least
   one source id. If no source supports it, omit it or phrase it as a general suggestion with
   an empty sources array.
3. Do not invent hotel names, restaurant names, or prices no source mentions. If sources are
   thin, say so in `unknowns` rather than filling the gap with plausible fiction."""

SKELETON_SYSTEM = f"""You are the planning half of a travel agent. Build the SHELL of an
itinerary from the supplied sources — not the day details yet.

{_RULES}

Plan the days geographically: each day gets one district or cluster so the traveller isn't
crossing the city and back. Account for arrival fatigue on day 1 and departure on the last day.

Return ONLY this JSON:
{{
 "title": "string",
 "overview": "2-3 sentences",
 "best_time_note": {{"text": "string", "sources": []}},
 "outline": [{{"day": 1, "theme": "short label", "area": "district or cluster",
               "anchor": "the one thing that day is built around"}}],
 "stays": [{{"area": "string", "who_its_for": "string", "price_band": "string",
             "examples": "only if a source names them", "sources": []}}],
 "transport": [{{"option": "string", "cost": "string", "tip": "string", "sources": []}}],
 "budget": {{"currency": "string",
             "per_day": [{{"category": "string", "amount": "string", "notes": "string",
                           "sources": []}}],
             "trip_total_estimate": "string", "assumptions": "string"}},
 "tips": [{{"tip": "string", "sources": []}}],
 "unknowns": ["what you could not verify from the sources"]
}}"""

DAY_SYSTEM = f"""You are the planning half of a travel agent. Expand ONE day of an itinerary
that has already been outlined. Stay inside that day's area — do not borrow other days' anchors.

{_RULES}

Respect the stated pace: relaxed = 2-3 blocks, balanced = 3-4, packed = 5+.
Account for travel time between blocks. Costs in local currency, per person.
Name specific places — `place` is used for a map lookup, so it must be findable.

Return ONLY this JSON:
{{"day": 1, "theme": "string", "area": "string",
  "blocks": [{{"time": "Morning|Midday|Afternoon|Evening", "activity": "string",
               "place": "specific named place", "duration": "2h",
               "cost": "JPY 1000 pp or Free", "why": "one line", "sources": []}}],
  "food": [{{"suggestion": "string", "place": "named place or area", "sources": []}}],
  "notes": "logistics note for the day"}}"""

_URL_RE = re.compile(r"https?://\S+|www\.\S+|\b[\w-]+\.(?:com|net|org|io|in|jp|co\.uk)\b\S*", re.I)
_SID_RE = re.compile(r"\bS\d+\b")

class TokenPacer:
    """Sliding 60-second window. Sleeps only when the next call wouldn't fit."""

    def __init__(self, limit: int, safety: int = 500):
        self.limit = limit - safety
        self.events: list[tuple[float, int]] = []

    def wait_for(self, tokens: int, notify=None) -> None:
        while True:
            now = time.time()
            self.events = [(t, n) for t, n in self.events if now - t < 60]
            used = sum(n for _, n in self.events)
            if used + tokens <= self.limit or not self.events:
                return
            oldest = min(t for t, _ in self.events)
            nap = max(1.0, 61 - (now - oldest))
            if notify:
                notify(nap, used, tokens)
            time.sleep(nap)

    def record(self, tokens: int) -> None:
        self.events.append((time.time(), tokens))


def compact_context(registry: SourceRegistry, limit: int | None = None,
                    snippet_chars: int | None = None) -> str:
    limit = limit or config.SYNTH_SOURCES
    snippet_chars = snippet_chars or config.SYNTH_SNIPPET_CHARS
    return "\n".join(
        f"[{s.id}] {s.title[:80]} — {s.domain}\n    {s.snippet[:snippet_chars]}"
        for s in registry.ranked(limit)
    )


def _call(system: str, user: str, model: str, pacer: TokenPacer,
          max_out: int, progress=None):
    """One paced JSON call. Trims max_out to whatever the window allows."""
    prompt_tokens = est_tokens(system + user)
    room = config.TPM_LIMIT - config.TPM_SAFETY - prompt_tokens
    max_out = max(600, min(max_out, room))
    pacer.wait_for(prompt_tokens + max_out,
                   notify=lambda nap, used, need: progress and
                   progress("wait", f"TPM window full ({used} used) — {nap:.0f}s"))
    data = llm.json_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=model, temperature=config.TEMP_PLAN, max_tokens=max_out,
    )
    pacer.record(prompt_tokens + max_out)
    return data

def _scrub(text: Any) -> Any:
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


def _walk(node: Any, registry: SourceRegistry) -> Any:
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


def build_itinerary(trip: TripState, registry: SourceRegistry, model=None,
                    progress=None) -> dict:
    """Skeleton call, then one call per day. Fits an 8000 TPM budget at any length."""
    model = model or config.PLAN_MODEL
    pacer = TokenPacer(config.TPM_LIMIT, config.TPM_SAFETY)
    ctx = compact_context(registry)
    days_n = trip.duration_days or 3

    if progress:
        progress("skeleton", "Shaping the trip…")
    skeleton = _call(
        SKELETON_SYSTEM,
        f"TRIP BRIEF\n{trip.brief()}\n\nToday: {date.today().isoformat()}\n\n"
        f"SOURCES (cite these ids only)\n{ctx}\n\n"
        f"Outline all {days_n} days, then fill the shell.",
        model, pacer, 2000, progress,
    )

    outline = {o.get("day"): o for o in skeleton.get("outline", []) if isinstance(o, dict)}
    day_ctx = compact_context(registry, limit=12)      # smaller per day = faster pacing

    days = []
    for n in range(1, days_n + 1):
        o = outline.get(n, {})
        if progress:
            progress("day", f"Day {n} — {o.get('area') or o.get('theme') or 'planning'}")
        others = "; ".join(f"d{k}: {v.get('area', '')}"
                            for k, v in outline.items() if k != n)
        day = _call(
            DAY_SYSTEM,
            f"TRIP BRIEF\n{trip.brief()}\n\n"
            f"THIS DAY: day {n} of {days_n} — theme: {o.get('theme','')}, "
            f"area: {o.get('area','')}, anchor: {o.get('anchor','')}\n"
            f"OTHER DAYS (do not duplicate): {others}\n\n"
            f"SOURCES\n{day_ctx}\n\nExpand day {n} now.",
            model, pacer, 1400, progress,
        )
        day["day"] = n
        day.setdefault("theme", o.get("theme", ""))
        day.setdefault("area", o.get("area", ""))
        days.append(day)

    itinerary = {k: v for k, v in skeleton.items() if k != "outline"}
    itinerary["days"] = days
    return _walk(itinerary, registry)

SCOPE_SYSTEM = """Decide what a revision request touches in a travel itinerary.
Return ONLY: {"days": [3], "sections": ["stays"], "queries": ["short web search", "..."]}
- `days`: day numbers to regenerate. Empty if the request isn't about a specific day.
- `sections`: any of stays, transport, budget, tips. Empty if none apply.
- `queries`: 1-3 SHORT web searches that would surface what's needed. Always include the
  destination. Empty only if the request needs no new information."""


def refine_itinerary(itinerary: dict, trip: TripState, registry: SourceRegistry,
                     request: str, model=None, progress=None) -> dict:
    """Scope the request, search fresh sources for it, regenerate only what's affected."""
    model = model or config.PLAN_MODEL
    pacer = TokenPacer(config.TPM_LIMIT, config.TPM_SAFETY)

    scope = llm.json_chat(
        [{"role": "system", "content": SCOPE_SYSTEM},
         {"role": "user", "content":
          f"Destination: {trip.destination}\nDays in trip: {trip.duration_days}\n"
          f"Request: {request}"}],
        model=config.FAST_MODEL, temperature=0.0, max_tokens=400,
    )

    # Fresh search. The registry is append-only, so new ids start after the
    # existing ones and every citation already in the plan stays valid.
    queries = [q for q in (scope.get("queries") or []) if isinstance(q, str)][:config.REFINE_QUERIES]
    if queries:
        if progress:
            progress("search", f"Searching: {'; '.join(queries)}")
        before = len(registry)
        search.run_queries(queries, registry)
        if progress:
            progress("search", f"+{len(registry) - before} new sources")

    ctx = compact_context(registry, limit=14)
    target_days = [d for d in (scope.get("days") or []) if isinstance(d, int)]
    sections = [s for s in (scope.get("sections") or []) if s in
                ("stays", "transport", "budget", "tips")]

    # Nothing scoped: fall back to regenerating day 1 so the request isn't silently dropped.
    if not target_days and not sections:
        target_days = [1]

    out = dict(itinerary)

    for n in target_days:
        if progress:
            progress("day", f"Reworking day {n}")
        old = next((d for d in out.get("days", []) if d.get("day") == n), {})
        new_day = _call(
            DAY_SYSTEM,
            f"TRIP BRIEF\n{trip.brief()}\n\n"
            f"CURRENT DAY {n}\n{json.dumps(old, ensure_ascii=False)}\n\n"
            f"SOURCES\n{ctx}\n\n"
            f"USER REQUEST: {request}\n\nRewrite day {n} to satisfy it. Keep what still works.",
            model, pacer, 1400, progress,
        )
        new_day["day"] = n
        out["days"] = [new_day if d.get("day") == n else d for d in out.get("days", [])]

    if sections:
        if progress:
            progress("section", f"Updating {', '.join(sections)}")
        current = {s: out.get(s) for s in sections}
        patch = _call(
            "You revise parts of a travel itinerary.\n" + _RULES +
            f"\n\nReturn ONLY a JSON object with exactly these keys: {sections}. "
            "Same schema as supplied.",
            f"TRIP BRIEF\n{trip.brief()}\n\n"
            f"CURRENT\n{json.dumps(current, ensure_ascii=False)}\n\n"
            f"SOURCES\n{ctx}\n\nUSER REQUEST: {request}",
            model, pacer, 1200, progress,
        )
        for s in sections:
            if s in patch:
                out[s] = patch[s]

    return _walk(out, registry)

def coverage_report(itinerary: dict, registry: SourceRegistry,
                    sources_shown: int | None = None) -> dict:
    """Grounding metrics. Measures citation density, NOT citation correctness."""
    factual = grounded = soft = 0
    unsupported: list[str] = []
    used: set[str] = set()

    def is_factual(node: dict) -> bool:
        cost = (node.get("cost") or "").strip().lower()
        cost = cost.replace("—", "-").replace("–", "-")
        if cost and cost not in ("free", "n/a", "-", "", "varies"):
            return True
        return "option" in node          # transport rows; tips are advice, not claims

    def visit(node, path=""):
        nonlocal factual, grounded, soft
        if isinstance(node, dict):
            if "sources" in node:
                srcs = node.get("sources") or []
                used.update(srcs)
                if is_factual(node):
                    factual += 1
                    if srcs:
                        grounded += 1
                    else:
                        label = node.get("activity") or node.get("option") or ""
                        unsupported.append(f"{path}: {label[:60]} ({node.get('cost','')})")
                else:
                    soft += 1
            for k, v in node.items():
                visit(v, f"{path}/{k}" if path else k)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                visit(v, f"{path}[{i}]")

    visit(itinerary)
    shown = sources_shown if sources_shown is not None else config.SYNTH_SOURCES
    return {
        "factual_claims": factual,
        "grounded": grounded,
        "pct": round(100 * grounded / factual) if factual else 0,
        "soft_suggestions": soft,
        "sources_used": len(used),
        "sources_shown": shown,
        "source_utilisation": f"{round(100 * len(used) / shown)}%" if shown else "—",
        "unsupported": unsupported,
    }