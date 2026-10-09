# =============================================================================
#  Project Fairness — Backend Dockerfile
#  Multi-stage build:
#    builder  — install Python deps into an isolated venv (cached layer)
#    runtime  — copy venv + source; used by both api AND ml_worker
#
#  GPU note:
#    This image is CPU-only by default.  When running inside Docker with
#    NVIDIA Container Toolkit, the GPU is exposed via the Compose
#    deploy.resources.reservations block — no changes to this file needed.
# =============================================================================

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Build tools for C-extension packages (psycopg2, Pillow, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev       \
        libgl1          \
        libglib2.0-0    \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Isolated virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirement files first to maximise Docker layer cache hits
COPY backend/requirements.txt ./backend-requirements.txt
COPY requirements.txt          ./ml-requirements.txt

# Install backend + ML deps (celery[redis] pulled in by backend requirements)
RUN pip install --upgrade pip wheel && \
    pip install --no-cache-dir -r backend-requirements.txt && \
    pip install --no-cache-dir -r ml-requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Project Fairness"
LABEL description="FastAPI backend + Celery ML worker for skin lesion classification"

# Runtime libs only (no compiler)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5          \
        libgl1          \
        libglib2.0-0    \
        curl            \
    && rm -rf /var/lib/apt/lists/*

# Non-root security
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ── Copy source code ──────────────────────────────────────────────────────────
# Root-level ML modules (imported by backend services via PYTHONPATH)
COPY --chown=appuser:appuser \
    preprocessing.py  \
    dataset.py        \
    diffusion_model.py \
    generate.py       \
    evaluate_fid.py   \
    fairness_balancer.py \
    classifier.py     \
    evaluate_classifier.py \
    gradcam.py        \
    shap_analysis.py  \
    explainer.py      \
    ./

# Optional sub-packages (copy only if they exist — Docker ignores missing dirs)
COPY --chown=appuser:appuser ddpm/     ./ddpm/

# Backend application package
COPY --chown=appuser:appuser backend/  ./backend/

# ── Directory structure ───────────────────────────────────────────────────────
# Pre-create writable dirs for static files + uploads
RUN mkdir -p \
        backend/static/uploads      \
        backend/static/generated    \
        backend/static/explanations \
        uploads                      \
        models                       \
    && chown -R appuser:appuser \
        backend/static/ \
        uploads/        \
        models/

# Switch to non-root
USER appuser

# Set PYTHONPATH so backend can import root-level ML modules
ENV PYTHONPATH="/app:/app/backend"

EXPOSE 8000

WORKDIR /app/backend

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: run FastAPI (overridden in docker-compose for ml_worker)
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000",     \
     "--workers", "2",     \
     "--log-level", "info"]
