# Repository scripts

| Script | Purpose |
|---|---|
| `init_database.py` | Creates the PostgreSQL + pgvector schema and loads `data/weather_knowledge.jsonl`. Idempotent; run before the API starts (Docker Compose `bootstrap` service). |
| `render_start.sh` | Render start command: retries `init_database.py` until the database accepts connections, then starts Gunicorn. |
| `validate_production_config.py` | Static checks for the production Docker configuration, secret placeholders, and the runtime/optional-ML dependency boundary. Run by GitHub Actions. |
