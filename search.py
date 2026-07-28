"""DuckDuckGo retrieval + the source registry.

The registry is the security guarantee of this whole project: the model may only
cite [S<id>] tokens, and those ids resolve to URLs we actually retrieved.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from urllib.parse import urlparse

import config

# The package was renamed from duckduckgo_search to ddgs. Support both.
try:  # pragma: no cover
    from ddgs import DDGS
except ImportError:  # pragma: no cover
    from duckduckgo_search import DDGS


@dataclass
class Source:
    id: str
    title: str
    url: str
    snippet: str
    domain: str
    query: str
    score: float = 0.0

    def as_context_line(self) -> str:
        return f"[{self.id}] {self.title} — {self.domain}\n    {self.snippet[:400]}"


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:  # noqa: BLE001
        return ""


def _score(url: str, title: str) -> float:
    d = _domain(url).lower()
    t = (title or "").lower()
    score = 1.0
    for hint in config.PREFERRED_DOMAIN_HINTS:
        if hint in d or hint in t:
            score += 1.0
            break
    for hint in config.DEMOTED_DOMAIN_HINTS:
        if hint in d:
            score -= 1.5
            break
    if any(w in t for w in ("top 10", "top 20", "best 15", "you won't believe")):
        score -= 0.4
    return score


class SourceRegistry:
    """Append-only store of retrieved sources with stable ids."""

    def __init__(self) -> None:
        self._sources: dict[str, Source] = {}
        self._seen_urls: set[str] = set()
        self._counter = 0

    def __len__(self) -> int:
        return len(self._sources)

    def add(self, title: str, url: str, snippet: str, query: str) -> Source | None:
        if not url or url in self._seen_urls:
            return None
        self._counter += 1
        src = Source(
            id=f"S{self._counter}",
            title=(title or "").strip() or url,
            url=url,
            snippet=(snippet or "").strip(),
            domain=_domain(url),
            query=query,
            score=_score(url, title),
        )
        self._sources[src.id] = src
        self._seen_urls.add(url)
        return src

    def get(self, sid: str) -> Source | None:
        return self._sources.get(sid.upper())

    def all(self) -> list[Source]:
        return list(self._sources.values())

    def ranked(self, limit: int | None = None) -> list[Source]:
        ordered = sorted(self._sources.values(), key=lambda s: -s.score)
        return ordered[:limit] if limit else ordered

    def context_block(self, limit: int = 60) -> str:
        return "\n".join(s.as_context_line() for s in self.ranked(limit))

    def to_list(self) -> list[dict]:
        return [asdict(s) for s in self.all()]


def search(query: str, max_results: int | None = None) -> list[dict]:
    """One DDG text search with backoff. Returns raw result dicts."""
    max_results = max_results or config.SEARCH_RESULTS_PER_QUERY
    last_err = None
    for attempt in range(config.SEARCH_MAX_RETRIES):
        try:
            with DDGS() as ddgs:
                results = list(
                    ddgs.text(
                        query,
                        region=config.SEARCH_REGION,
                        safesearch="moderate",
                        max_results=max_results,
                    )
                )
            return results
        except Exception as exc:  # noqa: BLE001 - DDG throws a zoo of errors
            last_err = exc
            time.sleep(config.SEARCH_DELAY_SECONDS * (attempt + 2))
    print(f"[search] giving up on {query!r}: {last_err}")
    return []


def run_queries(queries: list[str], registry: SourceRegistry, progress=None) -> SourceRegistry:
    """Execute queries sequentially, feeding everything into the registry."""
    for i, q in enumerate(queries, 1):
        if progress:
            progress(i, len(queries), q)
        for r in search(q):
            registry.add(
                title=r.get("title", ""),
                url=r.get("href") or r.get("url", ""),
                snippet=r.get("body") or r.get("description", ""),
                query=q,
            )
        time.sleep(config.SEARCH_DELAY_SECONDS)
    return registry
