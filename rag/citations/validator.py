"""Validate generated citations against the supplied source set."""
from __future__ import annotations

import re
from typing import Any

_CITATION_RE = re.compile(r"\[(S\d+)\]")


def _clean_removed_citation_spacing(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Removing a citation immediately before punctuation must not leave a space.
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def validate(answer: str, sources: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Keep only citations explicitly present in ``answer`` and backed by ``sources``.

    Unknown citation IDs are removed. Valid sources are never fabricated into the
    answer merely because they were supplied; the model must explicitly cite them.
    """
    text = (answer or "").strip()
    valid = {
        str(source.get("citation")): source
        for source in sources
        if source.get("citation")
    }

    cited: list[str] = []
    for citation in _CITATION_RE.findall(text):
        if citation in valid and citation not in cited:
            cited.append(citation)

    text = _CITATION_RE.sub(
        lambda match: match.group(0) if match.group(1) in valid else "",
        text,
    )
    text = _clean_removed_citation_spacing(text)
    text = re.sub(r"\n\nSources:\s*(?:\n- \[S\d+\].*)+$", "", text, flags=re.I)

    if not cited:
        return text, []

    text += "\n\nSources:\n" + "\n".join(
        f"- [{citation}] {valid[citation].get('source') or valid[citation].get('title') or 'Weather knowledge source'}"
        for citation in cited
    )
    return text, [valid[citation] for citation in cited]
