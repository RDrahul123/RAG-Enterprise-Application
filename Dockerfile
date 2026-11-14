# CPU-only image: no CUDA wheels, much smaller and faster to build.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    RAG_DOCS_DIR=/app/data/docs \
    RAG_INDEX_DIR=/app/data/index

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install torch from the CPU-only index first so pip does not pull ~2GB of CUDA.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r requirements.txt

COPY rag.py answer.py app.py ingest.py make_corpus.py ./
COPY ui ./ui

# Bake the embedder and reranker into the image so startup needs no downloads.
# The 3GB answering model is fetched on first request, or mount a HF cache volume.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('BAAI/bge-small-en-v1.5'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Ship with the sample corpus already indexed so the container works on start.
RUN python make_corpus.py && python ingest.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
