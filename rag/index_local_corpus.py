"""Load the bundled JSONL knowledge corpus into PostgreSQL."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import lakebase


def _stable_id(row: dict) -> str:
    explicit = row.get("id")
    if explicit: return str(explicit)
    raw = json.dumps(row, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_corpus(path: str | Path = "data/weather_knowledge.jsonl") -> int:
    lakebase.ensure_weather_tables(embedding_dim=384)
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows: return 0
    sql = """
    INSERT INTO weather_documents
      (id, location, state, district, source, source_type, headline,
       narrative_text, forecast_date, payload, synced_at)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now())
    ON CONFLICT (id) DO UPDATE SET
      location=EXCLUDED.location, state=EXCLUDED.state, district=EXCLUDED.district,
      source=EXCLUDED.source, source_type=EXCLUDED.source_type,
      headline=EXCLUDED.headline, narrative_text=EXCLUDED.narrative_text,
      forecast_date=EXCLUDED.forecast_date, payload=EXCLUDED.payload, synced_at=now()
    """
    with lakebase.get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.executemany(sql, [(
                _stable_id(row), row.get("location"), row.get("state"), row.get("district"),
                row.get("source", "bundled"), row.get("source_type", "knowledge"),
                row.get("title", "Weather knowledge"), row.get("text", row.get("narrative_text", "")),
                row.get("date", row.get("forecast_date")), json.dumps(row, default=str),
            ) for row in rows])
        connection.commit()
    return len(rows)


if __name__ == "__main__":
    print(f"Loaded {load_corpus()} documents into PostgreSQL")
