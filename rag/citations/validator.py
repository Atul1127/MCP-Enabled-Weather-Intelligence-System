"""Validate and align generated citations with retrieved sources."""
from __future__ import annotations
import re
from typing import Any


def validate(answer: str, sources: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    text = (answer or "").strip()
    valid = {str(s.get("citation")): s for s in sources if s.get("citation")}
    cited = []
    for citation in re.findall(r"\[(S\d+)\]", text):
        if citation in valid and citation not in cited:
            cited.append(citation)
    if not cited and valid:
        first = next(iter(valid))
        cited = [first]
        text = f"{text} [{first}]" if text else f"[{first}]"
    if not cited:
        return text, []
    text = re.sub(r"\n\nSources:\s*(?:\n- \[S\d+\].*)+$", "", text, flags=re.I)
    text += "\n\nSources:\n" + "\n".join(
        f"- [{c}] {valid[c].get('source') or valid[c].get('title') or 'Weather knowledge source'}"
        for c in cited
    )
    return text, [valid[c] for c in cited]
