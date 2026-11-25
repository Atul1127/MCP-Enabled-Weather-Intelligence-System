#!/bin/sh
# Render start command: wait for Postgres, initialise schema/corpus, start Gunicorn.
set -e
attempt=1
until python scripts/init_database.py; do
  if [ "$attempt" -ge 12 ]; then
    echo "Database still unavailable after $attempt attempts; giving up." >&2
    exit 1
  fi
  echo "Database not ready (attempt $attempt); retrying in 5s..." >&2
  attempt=$((attempt + 1))
  sleep 5
done
exec gunicorn --config docker/gunicorn.conf.py app:app
