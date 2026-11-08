# Single-service image: web app + agent + MCP server (stdio child process) + FAISS index.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE=/app/data/models
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# Build the RAG index at image build time (bakes the embedding model into the image => faster cold starts).
ARG EMBEDDING_PROVIDER=auto
RUN EMBEDDING_PROVIDER=${EMBEDDING_PROVIDER} python -m rag.ingest

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health')"
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
