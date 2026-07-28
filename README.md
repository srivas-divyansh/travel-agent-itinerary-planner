# Travel Planner Agent

Conversational itinerary builder. Collects trip details through chat, researches the
destination on the open web, and returns a day-by-day plan where **every link is real**.

## Why it's built this way

LLMs invent URLs. So the model here is never allowed to write one:

- **Source registry** — every DuckDuckGo result is stored as `{id, title, url, snippet, domain}`.
  The model cites `[S3]`; `render.py` resolves that to the actual URL.
- **Deterministic links** — Google Maps, Flights, Booking, Rome2Rio, visa lookups are built by
  string templating from the trip slots. They can't rot.
- **Validation pass** — `synthesize._walk()` walks the generated JSON, strips anything URL-shaped
  from prose, and drops any source id not present in the registry.
- **Coverage metric** — the sidebar shows what % of claims carry a source, so a thin research pass
  is visible instead of silently producing confident fiction.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # add your key from https://console.groq.com/keys
export GROQ_API_KEY=gsk_...
streamlit run app.py
```

## Flow

```
chat → intake.py (slot filling)  →  research.py (query planning + DDG)
                                          ↓
                                  search.py SourceRegistry
                                          ↓
                            synthesize.py (JSON + citation validation)
                                          ↓
                            render.py (markdown, links resolved)
```

## Files

| File | Role |
|---|---|
| `config.py` | Model id, temperatures, search tuning, domain preferences |
| `state.py` | `TripState` + slot schema; what counts as required |
| `llm.py` | Groq client, JSON-mode call with parse retry |
| `search.py` | DDG wrapper + `SourceRegistry` |
| `links.py` | Deterministic booking/map URL builders |
| `intake.py` | One conversational turn: extract slots, ask what's missing |
| `research.py` | LLM-planned queries with a template fallback |
| `synthesize.py` | Itinerary generation, refinement, validation, coverage |
| `render.py` | JSON → markdown with real hyperlinks |
| `app.py` | Streamlit chat |

## Known limits

- DuckDuckGo rate-limits hard. `config.SEARCH_DELAY_SECONDS` is the knob; if you see empty
  registries, raise it or swap the backend in `search.search()`.
- Sources are search snippets, not full pages. Fetching and chunking the top 5 pages would
  improve grounding a lot — that's the natural next step.
- No live pricing. Prices come from whatever the sources say, which may be stale; they're
  labelled as estimates and the booking links go to live search pages.
  
  
## Demo Link
https://drive.google.com/file/d/1-wRHi7i41KFkIvU6zaM5LFrTBJQEBwI4/view?usp=drive_link
