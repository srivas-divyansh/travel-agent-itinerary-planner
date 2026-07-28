"""Intake: turn free-form chat into filled slots, one or two questions at a time."""
from __future__ import annotations

import json

import config
import llm
from state import SLOTS, TripState

_SCHEMA_DESC = "\n".join(
    f"- {name} ({'required' if req else 'optional'}): {desc}"
    for name, (_label, req, desc) in SLOTS.items()
)

SYSTEM = f"""You are the intake half of a travel planning agent. Your only job this turn is to
(a) extract trip details from what the user just said and (b) ask for what is still missing.

Slots you can fill:
{_SCHEMA_DESC}

Rules:
- Extract only what the user actually said or clearly implied. Never invent a budget, date, or interest.
- If the user says something like "surprise me" or "you decide" for a slot, fill it with a sensible
  value and note it in `assumptions`.
- Ask at most TWO questions per turn, phrased warmly and concretely. Offer example answers.
- If the user gave a vague destination ("somewhere warm in December"), ask one narrowing question
  rather than guessing a city.
- If all required slots are filled, set `ready` to true and make `reply` a short confirmation that
  you are about to research the trip. Do not ask more questions when ready is true.
- The destination must be a real place a traveller can physically reach by commercial
  transport. If it is fictional, extraterrestrial, or otherwise unreachable, leave
  `destination` null, set ready=false, and say plainly in `reply` that you can only plan
  trips to real destinations.

Return ONLY this JSON object:
{{
  "updates": {{ "<slot>": <value>, ... }},
  "assumptions": ["..."],
  "ready": false,
  "reply": "your message to the user"
}}"""


def intake_turn(trip: TripState, history: list[dict[str, str]], user_message: str) -> dict:
    """Run one intake turn. Mutates `trip` with extracted values."""
    known = json.dumps(trip.to_dict(), ensure_ascii=False)
    missing = trip.missing_required()

    convo = [{"role": "system", "content": SYSTEM}]
    convo.extend(history[-8:])
    convo.append(
        {
            "role": "user",
            "content": (
                f"Already collected: {known}\n"
                f"Still missing (required): {missing or 'nothing'}\n\n"
                f"User just said: {user_message}"
            ),
        }
    )

    data = llm.json_chat(
        convo,
        temperature=config.TEMP_EXTRACT,
        max_tokens=config.MAX_TOKENS_EXTRACT,
    )

    changed = trip.apply(data.get("updates", {}))
    ready = trip.is_ready()

    return {
        "reply": data.get("reply") or "Got it.",
        "changed": changed,
        "assumptions": data.get("assumptions", []),
        "ready": ready,
        "missing": trip.missing_required(),
    }


OPENING = (
    "Hi! I'll build you a day-by-day itinerary with real, working links for everything "
    "I recommend.\n\nTo start: **where are you headed**, **roughly when or for how many days**, "
    "and **where are you travelling from**?"
)
