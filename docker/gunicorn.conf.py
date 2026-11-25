import os

# Render, Railway, Heroku etc. inject PORT; fall back to FLASK_RUN_PORT locally.
bind = f"{os.getenv('FLASK_RUN_HOST', '0.0.0.0')}:{os.getenv('PORT') or os.getenv('FLASK_RUN_PORT', '8000')}"
workers = int(os.getenv('GUNICORN_WORKERS', '2'))
threads = int(os.getenv('GUNICORN_THREADS', '4'))
timeout = int(os.getenv('GUNICORN_TIMEOUT', '120'))
keepalive = int(os.getenv('GUNICORN_KEEPALIVE', '5'))
accesslog = '-'
errorlog = '-'
loglevel = os.getenv('GUNICORN_LOG_LEVEL', 'info')
worker_class = 'gthread'
preload_app = True
