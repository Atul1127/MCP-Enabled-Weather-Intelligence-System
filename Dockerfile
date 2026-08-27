FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV DATABASE_BACKEND=local
ENV GEMINI_MODEL=gemini-3.6-flash
ENV GEMINI_FALLBACK_MODELS=gemini-3.5-flash-lite,gemini-2.5-flash-lite
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8000
ENV WEATHER_ALLOW_SYNC=0

EXPOSE 8000

CMD ["python", "app.py"]
