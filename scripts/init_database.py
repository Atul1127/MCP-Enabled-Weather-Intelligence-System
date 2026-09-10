"""Initialize the RAG database outside the request path."""
from __future__ import annotations

import lakebase
from rag.index_local_corpus import load_corpus


def main() -> None:
    lakebase.ensure_weather_tables(embedding_dim=384)
    count = load_corpus()
    print(f"Database schema initialized; corpus rows processed: {count}")


if __name__ == "__main__":
    main()
