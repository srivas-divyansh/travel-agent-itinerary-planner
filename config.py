"""Central configuration. Everything tunable lives here."""
import os

# Groq model. Swap freely — list live models with:
#   curl -H "Authorization: Bearer $GROQ_API_KEY" https://api.groq.com/openai/v1/models
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Lower temp for slot extraction (must be literal), higher for the itinerary prose.
TEMP_EXTRACT = 0.0
TEMP_PLAN = 0.6

MAX_TOKENS_EXTRACT = 1200
MAX_TOKENS_PLAN = 8000

# DuckDuckGo tuning. DDG rate-limits aggressively; these keep us under the line.
SEARCH_RESULTS_PER_QUERY = 5
SEARCH_DELAY_SECONDS = 1.2
SEARCH_MAX_RETRIES = 3
SEARCH_REGION = "wt-wt"

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


def require_api_key() -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
            "and export it, or put it in a .env file."
        )
    return key
