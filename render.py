"""Render itinerary JSON to markdown, resolving [S<id>] into real links."""
from __future__ import annotations

import links
from search import SourceRegistry
from state import TripState


def _cite(source_ids: list[str], registry: SourceRegistry) -> str:
    out = []
    for sid in source_ids or []:
        src = registry.get(sid)
        if src:
            out.append(f"[{src.domain}]({src.url})")
    return " " + " ".join(out) if out else ""


def itinerary_to_markdown(itinerary: dict, trip: TripState, registry: SourceRegistry) -> str:
    L: list[str] = []
    A = L.append

    A(f"# {itinerary.get('title', 'Your trip')}")
    if itinerary.get("overview"):
        A(f"\n{itinerary['overview']}\n")

    btn = itinerary.get("best_time_note") or {}
    if btn.get("text"):
        A(f"**When to go —** {btn['text']}{_cite(btn.get('sources'), registry)}\n")

    # --- days ---
    for day in itinerary.get("days", []):
        header = f"## Day {day.get('day')} — {day.get('theme', '')}"
        if day.get("area"):
            header += f"  \n*Focus: {day['area']}*"
        A(f"\n{header}\n")

        for b in day.get("blocks", []):
            place = b.get("place") or b.get("activity", "")
            map_url = links.map_link_for(place, trip.destination) if place else None
            title = f"**{b.get('time', '')} — {b.get('activity', '')}**"
            A(title)
            meta = []
            if b.get("duration"):
                meta.append(b["duration"])
            if b.get("cost"):
                meta.append(b["cost"])
            if map_url:
                meta.append(f"[map]({map_url})")
            if meta:
                A(f"  \n<sub>{' · '.join(meta)}</sub>")
            if b.get("why"):
                A(f"  \n{b['why']}{_cite(b.get('sources'), registry)}")
            A("")

        food = day.get("food") or []
        if food:
            A("**Eat:** " + "; ".join(
                f"{f.get('suggestion', '')}"
                + (f" — [{f['place']}]({links.map_link_for(f['place'], trip.destination)})" if f.get("place") else "")
                + _cite(f.get("sources"), registry)
                for f in food
            ))
        if day.get("notes"):
            A(f"\n> {day['notes']}")

    # --- stays ---
    stays = itinerary.get("stays") or []
    if stays:
        A("\n## Where to stay\n")
        for s in stays:
            area = s.get("area", "")
            A(f"**{area}** — {s.get('who_its_for', '')} · {s.get('price_band', '')}"
              f"{_cite(s.get('sources'), registry)}")
            if s.get("examples"):
                A(f"  \n<sub>Named in sources: {s['examples']}</sub>")
            if area:
                A(f"  \n<sub>[Search stays here]({links.booking_search(f'{area}, {trip.destination}', trip.start_date, None)})</sub>")
            A("")

    # --- transport ---
    transport = itinerary.get("transport") or []
    if transport:
        A("\n## Getting around\n")
        A("| Option | Cost | Tip | Source |")
        A("|---|---|---|---|")
        for t in transport:
            A(f"| {t.get('option','')} | {t.get('cost','')} | {t.get('tip','')} |"
              f"{_cite(t.get('sources'), registry) or ' —'} |")

    # --- budget ---
    budget = itinerary.get("budget") or {}
    if budget.get("per_day"):
        A(f"\n## Budget ({budget.get('currency','')})\n")
        A("| Category | Per day | Notes |")
        A("|---|---|---|")
        for row in budget["per_day"]:
            A(f"| {row.get('category','')} | {row.get('amount','')} | {row.get('notes','')} |")
        if budget.get("trip_total_estimate"):
            A(f"\n**Trip total estimate:** {budget['trip_total_estimate']}")
        if budget.get("assumptions"):
            A(f"  \n<sub>{budget['assumptions']}</sub>")

    # --- tips ---
    tips = itinerary.get("tips") or []
    if tips:
        A("\n## Good to know\n")
        for t in tips:
            A(f"- {t.get('tip','')}{_cite(t.get('sources'), registry)}")

    # --- deterministic booking links ---
    A("\n## Book it\n")
    for item in links.booking_bundle(trip):
        A(f"- [{item['label']}]({item['url']})")

    # --- unknowns ---
    unknowns = itinerary.get("unknowns") or []
    if unknowns:
        A("\n## Verify before you go\n")
        for u in unknowns:
            A(f"- {u}")

    # --- sources ---
    used = _used_source_ids(itinerary)
    if used:
        A("\n## Sources\n")
        for sid in sorted(used, key=lambda x: int(x[1:])):
            src = registry.get(sid)
            if src:
                A(f"- [{sid}] [{src.title}]({src.url})")

    return "\n".join(L)


def _used_source_ids(node, acc: set | None = None) -> set:
    acc = acc if acc is not None else set()
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "sources":
                acc.update(v or [])
            else:
                _used_source_ids(v, acc)
    elif isinstance(node, list):
        for v in node:
            _used_source_ids(v, acc)
    return acc
