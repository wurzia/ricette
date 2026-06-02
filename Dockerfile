FROM python:3.12.13 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app


RUN python -m venv .venv
COPY requirements.txt ./
RUN .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN .venv/bin/pip install -r requirements.txt
RUN .venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"
FROM python:3.12.13-slim
WORKDIR /app
COPY --from=builder /app/.venv .venv/
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface
COPY . .
CMD ["/app/.venv/bin/fastapi", "run", "server.py"]
