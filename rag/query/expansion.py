"""Query expansion isolated from retrieval and generation."""
from __future__ import annotations
import json
import re
from typing import Callable


def parse(raw: str) -> list[str]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I | re.S).strip()
    for candidate in (cleaned, (re.search(r"\{.*\}", cleaned, flags=re.S) or [None])[0]):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        queries = data.get("queries", []) if isinstance(data, dict) else []
        if isinstance(queries, list):
            return [str(q).strip() for q in queries if str(q).strip()]
    return []


def expand(query: str, generate: Callable[[str], str] | None = None, max_variants: int = 2) -> list[str]:
    """Return original query plus model-generated retrieval variants."""
    words = re.findall(r"\w+", query.lower())
    complex_query = len(words) > 10 and any(
        x in query.lower() for x in ("compare", "difference", "versus", " vs ", "why", "how", "factors", "relationship", "conditions")
    )
    if not complex_query or generate is None:
        return [query]
    raw = generate(
        'Rewrite the weather search query into two short retrieval queries. '
        'Preserve location, date, hazard and activity terms. Return JSON only as '
        '{"queries":["...","..."]}. Query: ' + query
    )
    return [query, *parse(raw)[:max_variants]]
