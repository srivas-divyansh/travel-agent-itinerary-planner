"""Conversation state. The whole point of owning our own loop is owning this."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# Slot definitions: name -> (human label, required?, description for the LLM)
SLOTS: dict[str, tuple[str, bool, str]] = {
    "origin": ("Departing from", True, "City and country the traveller starts from. Needed for flights and visa rules."),
    "destination": ("Destination", True, "City/region/country they want to visit."),
    "start_date": ("Start date", False, "ISO date YYYY-MM-DD if known, else null."),
    "duration_days": ("Duration", True, "Total number of days of the trip, as an integer."),
    "travelers": ("Travellers", True, "Who is going: count, and whether adults/kids/seniors, e.g. '2 adults, 1 child aged 6'."),
    "budget": ("Budget", True, "Total or per-day budget with currency, or a tier: shoestring / mid-range / comfortable / luxury."),
    "interests": ("Interests", True, "List of themes: food, history, nightlife, nature, art, shopping, adventure, temples, photography..."),
    "pace": ("Pace", False, "packed | balanced | relaxed"),
    "accommodation": ("Stay preference", False, "hostel | budget hotel | boutique | 4-5 star | apartment | ryokan etc."),
    "dietary": ("Dietary needs", False, "vegetarian, jain, halal, vegan, allergies, or none."),
    "must_do": ("Must-do list", False, "Specific things they have already decided they want."),
    "constraints": ("Constraints", False, "Mobility limits, avoid-list, visa concerns, anything else."),
}

REQUIRED_SLOTS = [k for k, v in SLOTS.items() if v[1]]


@dataclass
class TripState:
    origin: str | None = None
    destination: str | None = None
    start_date: str | None = None
    duration_days: int | None = None
    travelers: str | None = None
    budget: str | None = None
    interests: list[str] = field(default_factory=list)
    pace: str | None = None
    accommodation: str | None = None
    dietary: str | None = None
    must_do: list[str] = field(default_factory=list)
    constraints: str | None = None

    # ---- helpers -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def missing_required(self) -> list[str]:
        out = []
        for name in REQUIRED_SLOTS:
            val = getattr(self, name)
            if val in (None, "", [], 0):
                out.append(name)
        return out

    def is_ready(self) -> bool:
        return not self.missing_required()

    def apply(self, updates: dict[str, Any]) -> list[str]:
        """Merge LLM-extracted slot values. Returns the names actually changed."""
        changed = []
        for key, value in (updates or {}).items():
            if key not in SLOTS or value in (None, "", [], "null", "unknown"):
                continue
            if key == "duration_days":
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    continue
                if value <= 0 or value > 90:
                    continue
            if key in ("interests", "must_do"):
                if isinstance(value, str):
                    value = [v.strip() for v in value.split(",") if v.strip()]
                existing = list(getattr(self, key) or [])
                merged = existing + [v for v in value if v not in existing]
                if merged != existing:
                    setattr(self, key, merged)
                    changed.append(key)
                continue
            if getattr(self, key) != value:
                setattr(self, key, value)
                changed.append(key)
        return changed

    def summary_lines(self) -> list[str]:
        lines = []
        for key, (label, _req, _desc) in SLOTS.items():
            val = getattr(self, key)
            if val in (None, "", []):
                continue
            if isinstance(val, list):
                val = ", ".join(val)
            lines.append(f"{label}: {val}")
        return lines

    def brief(self) -> str:
        """Compact text form fed into research + synthesis prompts."""
        return "\n".join(self.summary_lines()) or "(nothing collected yet)"
