FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-runtime.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-runtime.txt \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

COPY . .

ENV DATABASE_BACKEND=local \
    GEMINI_MODEL=gemini-3.6-flash \
    GEMINI_FALLBACK_MODELS=gemini-3.5-flash-lite,gemini-2.5-flash-lite \
    GEMINI_THINKING_LEVEL=low \
    FLASK_RUN_HOST=0.0.0.0 \
    FLASK_RUN_PORT=8000 \
    WEATHER_ALLOW_SYNC=0

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=5 --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=3)"

CMD ["gunicorn", "--config", "docker/gunicorn.conf.py", "app:app"]
