"""Static production-configuration checks used by CI."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    runtime = (ROOT / "requirements-runtime.txt").read_text(encoding="utf-8")
    optional_ml = (ROOT / "requirements-rag-ml.txt").read_text(encoding="utf-8")
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")

    required_compose = (
        "DATABASE_BACKEND: local",
        "GEMINI_API_KEY: ${GEMINI_API_KEY:?GEMINI_API_KEY must be set}",
        "read_only: true",
        "cap_drop:\n      - ALL",
        "no-new-privileges:true",
        "WEATHER_ALLOW_SYNC: \"0\"",
    )
    for marker in required_compose:
        assert marker in compose, f"docker-compose.yml missing: {marker}"

    for marker in ("USER appuser", 'CMD ["gunicorn"', "HEALTHCHECK"):
        assert marker in dockerfile, f"Dockerfile missing: {marker}"

    assert "torch" not in runtime
    assert "sentence-transformers" not in runtime
    assert "sentence-transformers>=" in optional_ml
    assert "torch>=" in optional_ml
    assert "GEMINI_API_KEY=" in env_example
    assert "GEMINI_FALLBACK_MODELS=gemini-3.5-flash-lite" in env_example

    print("production configuration: OK")


if __name__ == "__main__":
    main()
