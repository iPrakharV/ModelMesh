FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements/base.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY gateway ./gateway
HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()"
CMD ["sh", "-c", "uvicorn gateway.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
