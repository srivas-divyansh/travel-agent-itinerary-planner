"""Central configuration. Everything tunable lives here."""
import os

# Two-model split. FAST handles extraction and query planning (mechanical, no
# reasoning needed). PLAN handles synthesis, where reasoning improves day
# sequencing and geographic grouping.
FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "llama-3.3-70b-versatile")
PLAN_MODEL = os.getenv("GROQ_PLAN_MODEL", "openai/gpt-oss-120b")

# Per-model extra kwargs. gpt-oss reasons by default at medium effort and will
# return EMPTY content if the token budget is spent thinking. These suppress it.
# llama models reject these params, so they're only sent to models that accept them.
MODEL_EXTRA = {
    "openai/gpt-oss-120b": {"reasoning_effort": "low", "include_reasoning": False},
    "openai/gpt-oss-20b":  {"reasoning_effort": "low", "include_reasoning": False},
}

def extra_kwargs(model: str) -> dict:
    return dict(MODEL_EXTRA.get(model, {}))

TEMP_EXTRACT = 0.0     # slot extraction is transcription, not creativity
TEMP_QUERIES = 0.2
TEMP_PLAN    = 0.6     # itinerary prose reads badly at 0.0

MAX_TOKENS_EXTRACT = 1200
MAX_TOKENS_QUERIES = 800
MAX_TOKENS_PLAN    = 3500      # was 8000 — see TPM note below

# Groq free tier: 8000 tokens per MINUTE, and max_tokens counts toward it.
# Reserving 8000 output tokens alone consumes the entire budget -> HTTP 413.
# Check your actual limit at https://console.groq.com/settings/limits
TPM_LIMIT  = int(os.getenv("GROQ_TPM_LIMIT", "8000"))
TPM_SAFETY = 500               # headroom for prompt-overhead estimation error

# How many sources get packed into the synthesis prompt, and how much of each.
SYNTH_SOURCES = 18
SYNTH_SNIPPET_CHARS = 180

SEARCH_RESULTS_PER_QUERY = 5
SEARCH_DELAY_SECONDS     = 1.5   # raise to 3.0 if DDG starts returning nothing
SEARCH_MAX_RETRIES       = 3
SEARCH_REGION            = "wt-wt"
MAX_QUERIES              = 14    # research.py was hardcoding this
REFINE_QUERIES           = 3     # searches fired per refinement turn

# Domains we trust more when ranking sources. Official > aggregator > listicle.
PREFERRED_DOMAIN_HINTS = (
    ".gov", ".gov.in", ".gov.uk", ".go.jp", ".gouv.fr",
    "tourism", "visit", "official", "museum", "railway",
    "transport", "metro", "unesco", "wikipedia.org", "wikivoyage.org",
)

# Domains that are usually SEO chaff for this use case.
DEMOTED_DOMAIN_HINTS = (
    "pinterest.", "quora.", "facebook.", "instagram.", "reddit.com/r/all",
)

# Strong title signals only — "visit"/"museum" were too weak and inflated listicles.
TITLE_STRONG_HINTS = ("official travel guide", "official website", "official site")

def require_api_key() -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
            "and export it, or put it in a .env file."
        )
    return key
