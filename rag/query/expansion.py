"""Provider-independent query expansion."""
from __future__ import annotations

import json
import re
from typing import Callable


_EXPANSION_TRIGGERS = (
    "compare",
    "difference",
    "versus",
    " vs ",
    "why",
    "how",
    "factors",
    "relationship",
    "conditions",
)


def parse(raw: str) -> list[str]:
    """Parse model output containing a JSON object with a ``queries`` list."""
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I | re.S
    ).strip()
    candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if match and match.group(0) != cleaned:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        queries = data.get("queries", []) if isinstance(data, dict) else []
        if isinstance(queries, list):
            return [str(q).strip() for q in queries if str(q).strip()]
    return []


def expand(
    query: str,
    generate: Callable[[str], str] | None = None,
    max_variants: int = 2,
) -> list[str]:
    """Return the original query plus up to ``max_variants`` expansions.

    Expansion is intentionally conservative: only sufficiently substantial
    queries with an explicit analytical/comparison signal are expanded.
    """
    if max_variants < 0:
        raise ValueError("max_variants must be non-negative")

    text = query.strip()
    if not text:
        return [query]
    if generate is None or max_variants == 0:
        return [query]

    lower = text.lower()
    words = re.findall(r"\w+", lower)
    complex_query = len(words) >= 8 and any(
        trigger in lower for trigger in _EXPANSION_TRIGGERS
    )
    if not complex_query:
        return [query]

    raw = generate(
        'Rewrite the weather search query into two short retrieval queries. '
        'Preserve location, date, hazard and activity terms. Return JSON only '
        'as {"queries":["...","..."]}. Query: '
        + text
    )
    variants = parse(raw)

    # Preserve order while removing empty/duplicate variants and the original
    # query itself. The caller always receives the original first.
    unique: list[str] = []
    seen = {text.casefold()}
    for variant in variants:
        key = variant.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(variant)
        if len(unique) >= max_variants:
            break

    return [query, *unique]
