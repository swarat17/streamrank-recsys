# ── Stage 1: dependency builder ──────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ src/
COPY pyproject.toml .

# Copy trained model artifacts (must exist before docker build)
COPY models/ models/

# Install the package in editable-equivalent mode
RUN pip install --no-cache-dir -e . 2>/dev/null || true

ENV PYTHONUNBUFFERED=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8000

EXPOSE 8000

CMD ["uvicorn", "src.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
