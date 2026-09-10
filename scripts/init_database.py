"""Initialize the RAG database outside the request path.

This script is intentionally runnable both as ``python scripts/init_database.py``
and from the repository root. It does not start PostgreSQL itself; the database
must already be reachable through the configured connection settings.
"""
from __future__ import annotations

import sys
from pathlib import Path

# When this file is executed directly, Python puts ``scripts/`` on sys.path
# rather than the repository root. Add the root explicitly so imports such as
# ``lakebase`` and ``rag`` work consistently on Windows, Linux, and CI.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lakebase
from rag.index_local_corpus import load_corpus


def main() -> None:
    try:
        lakebase.ensure_weather_tables(embedding_dim=384)
        count = load_corpus()
    except Exception as exc:
        print(f"Database initialization failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "Make sure PostgreSQL is running and DATABASE_URL/DATABASE_BACKEND "
            "point to a reachable database, then run this script again.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    print(f"Database schema initialized; corpus rows processed: {count}")


if __name__ == "__main__":
    main()
