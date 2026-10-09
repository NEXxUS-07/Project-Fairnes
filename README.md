# Project Fairness — Skin Lesion Classification System

> **Fairness-aware AI for dermatological image classification.**  
> EfficientNet-B3 trained on HAM10000, balanced with class-conditional DDPM synthetic images,
> with Grad-CAM + SHAP explainability and a production-ready FastAPI + React web interface.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start (Docker)](#quick-start-docker)
4. [Service Reference](#service-reference)
5. [GPU Setup](#gpu-setup)
6. [Local Development](#local-development)
7. [ML Pipeline](#ml-pipeline)
8. [Fairness Metrics](#fairness-metrics)
9. [API Reference](#api-reference)
10. [Project Structure](#project-structure)
11. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        Client Browser                            │
└───────────────────────────┬──────────────────────────────────────┘
                            │ :3000
                ┌───────────▼────────────┐
                │  Nginx (frontend)      │  React + Tailwind SPA
                │  /api/* → api:8000     │
                └───────────┬────────────┘
                            │
          ┌─────────────────┼──────────────────┐
          │                 │                  │
┌─────────▼──────┐  ┌───────▼──────┐  ┌───────▼──────┐
│  FastAPI (api) │  │  Celery      │  │  PostgreSQL  │
│  :8000         │  │  ml_worker   │  │  :5432       │
│  EfficientNet  │  │  DDPM gen.   │  │  skinlesion  │
│  Grad-CAM/SHAP │  │  async tasks │  │  _db         │
└────────────────┘  └──────────────┘  └──────────────┘
          │                 │
          └────────┬────────┘
                   │
          ┌────────▼────────┐
          │  Redis :6379    │  Celery broker + result backend
          └─────────────────┘
```

---

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Docker Engine | ≥ 24.0 | `docker --version` |
| Docker Compose | ≥ 2.20 | `docker compose version` |
| NVIDIA drivers | ≥ 525 | Only for GPU mode |
| nvidia-container-toolkit | latest | Only for GPU mode — [install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) |

---

## Quick Start (Docker)

### 1 — Clone & configure

```bash
git clone https://github.com/your-org/project-fairness.git
cd project-fairness

# Create your local environment file
cp .env.example .env
# Edit .env — change SECRET_KEY and POSTGRES_PASSWORD at minimum
```

### 2 — Place ML model files

The containers mount `./models/` as read-only. Create it and add your trained files:

```
models/
├── best_model.pth               ← EfficientNet-B3 checkpoint (from train_classifier.py)
├── fairness_report.json         ← Fairness report  (from fairness_balancer.py)
├── metrics.json                 ← Per-class metrics (from evaluate_classifier.py)
└── diffusion/
    └── ema_epoch0200.pt         ← DDPM checkpoint  (from train_diffusion.py)
```

If you don't have models yet, set `ML_DUMMY_MODE=true` and `DIFFUSION_DUMMY_MODE=true` in `.env` to run with dummy responses.

### 3 — Start all services

```bash
docker compose up -d
```

This will:
1. Pull `postgres:15-alpine` and `redis:7-alpine`
2. Build the backend image (installs Python deps — may take a few minutes first run)
3. Build the frontend image (npm install + Vite build)
4. Run Alembic migrations (one-shot container)
5. Start `api`, `ml_worker`, and `frontend`

### 4 — Verify

```bash
# Check all services are healthy
docker compose ps

# Stream API logs
docker compose logs -f api

# Open in browser
open http://localhost:3000     # React frontend
open http://localhost:8000/docs  # FastAPI Swagger UI
```

---

## Service Reference

| Service | Container | Port | Description |
|---------|-----------|------|-------------|
| `db` | `fairness_db` | 5432 | PostgreSQL 15 — persistent predictions + metrics |
| `redis` | `fairness_redis` | 6379 | Redis 7 — Celery broker + result backend |
| `migrate` | `fairness_migrate` | — | One-shot Alembic migration runner |
| `api` | `fairness_api` | 8000 | FastAPI — prediction, history, metrics, generation endpoints |
| `ml_worker` | `fairness_ml_worker` | — | Celery worker — async DDPM image generation jobs |
| `frontend` | `fairness_frontend` | 3000 | Nginx — React SPA + reverse proxy to api |

### Useful compose commands

```bash
# Start / stop individual services
docker compose up -d db redis                  # start only infra
docker compose stop ml_worker                  # pause worker

# Rebuild after code changes
docker compose build api --no-cache
docker compose up -d api

# Run a one-off command inside the API container
docker compose exec api python -c "import torch; print(torch.cuda.is_available())"

# View Celery worker logs
docker compose logs -f ml_worker

# Database shell
docker compose exec db psql -U fairness -d skinlesion_db

# Full teardown (removes volumes too)
docker compose down -v
```

---

## GPU Setup

### Requirements
- NVIDIA GPU with CUDA ≥ 11.8
- NVIDIA drivers ≥ 525 on the host
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Enable GPU in .env

```dotenv
ML_DEVICE=cuda
DIFFUSION_DEVICE=cuda
CELERY_CONCURRENCY=1        # set to number of GPUs
```

The `api` and `ml_worker` services already have the GPU `deploy` block:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

> **Important:** The `deploy.resources` GPU block only takes effect with `docker compose up` (Compose v2). It has **no effect** with plain `docker run`.

### Verify GPU inside container

```bash
docker compose exec api python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Local Development

### Backend (without Docker)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate

# Install deps
pip install -r requirements.txt
pip install -r ../requirements.txt   # ML deps

# Set environment variables
export DATABASE_URL="sqlite+aiosqlite:///./dev.db"
export ML_DUMMY_MODE=true

# Run migrations (SQLite)
alembic upgrade head

# Start dev server
uvicorn main:app --reload --port 8000
```

### Frontend (without Docker)

```bash
cd frontend
npm install --legacy-peer-deps
cp .env.example .env.local

# Dev server (proxies /api → localhost:8000)
npm run dev
# → http://localhost:5173
```

### Celery worker (locally)

```bash
# In a separate terminal (venv active, from project root)
export CELERY_BROKER_URL="redis://localhost:6379/0"
export CELERY_RESULT_URL="redis://localhost:6379/1"
celery -A backend.tasks worker --loglevel=info --queues=generation,default
```

---

## ML Pipeline

Run these scripts in order to train the full pipeline:

```bash
# 0 — Activate venv
source .venv/bin/activate

# 1 — Preprocess HAM10000 dataset
python preprocessing.py

# 2 — Train diffusion model (generates synthetic images)
python train_diffusion.py

# 3 — Generate synthetic images for minority classes
python generate.py --class mel --num_images 2000
python generate.py --class bkl --num_images 1500
# ...etc for all minority classes

# 4 — Evaluate generative quality
python evaluate_fid.py

# 5 — Apply fairness balancing
python fairness_balancer.py --strategy hybrid

# 6 — Train classifier on balanced dataset
python train_classifier.py

# 7 — Evaluate classifier + generate metrics.json
python evaluate_classifier.py

# Copy outputs to models/ for Docker
cp results/best_model.pth     models/
cp fairness_report.json        models/
cp results/metrics.json        models/
cp -r checkpoints_diffusers/  models/diffusion/
```

---

## Fairness Metrics

The system tracks three key fairness metrics before/after dataset balancing:

| Metric | Description | Target |
|--------|-------------|--------|
| **Demographic Parity Difference (DPD)** | `max(prediction_rate) - min(prediction_rate)` | < 10% |
| **Equal Opportunity Difference (EOD)** | `max(recall) - min(recall)` across 7 classes | < 20% |
| **Average Recall Gap** | `recall(majority) - mean(recall)` across all classes | < 15% |

### Balancing Strategies

Configure in `fairness_balancer.py` via `--strategy`:

| Strategy | Description |
|----------|-------------|
| `equal` | Fill all classes to the size of the majority class |
| `proportional` | Reduce majority + fill minority to proportional target |
| `hybrid` | Cap majority at 2× minority, fill remainder with synthetic images |

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/predict` | Upload image, returns class + confidence + XAI URLs |
| `GET` | `/api/history` | Paginated prediction history with filters |
| `GET` | `/api/metrics/fairness` | Per-class metrics + DPD fairness report |
| `GET` | `/api/dashboard/stats` | Aggregate stats for dashboard |
| `POST` | `/api/generate` | Queue synthetic image generation job |
| `GET` | `/api/generate/{job_id}` | Poll generation job status + result URLs |
| `GET` | `/health` | Liveness probe (used by Docker healthcheck) |
| `GET` | `/ready` | Readiness probe (checks DB + model loaded) |
| `GET` | `/docs` | Swagger UI (auto-generated) |

---

## Project Structure

```
project-fairness/
│
├── 📁 backend/                    FastAPI application
│   ├── main.py                   App entry point, middleware, routers
│   ├── config.py                 Pydantic-Settings configuration
│   ├── database.py               Async SQLAlchemy engine + session
│   ├── tasks.py                  Celery task definitions
│   ├── models/db_models.py       SQLAlchemy ORM models
│   ├── routers/                  API route handlers
│   │   ├── predict.py            POST /api/predict
│   │   ├── history.py            GET  /api/history
│   │   ├── metrics.py            GET  /api/metrics/fairness
│   │   └── generate.py           POST /api/generate
│   ├── services/                 Business logic singletons
│   │   ├── ml_service.py         EfficientNet-B3 + Grad-CAM + SHAP
│   │   └── diffusion_service.py  DDPM image generation
│   ├── schemas/prediction.py     Pydantic v2 request/response models
│   ├── migrations/               Alembic migration files
│   ├── requirements.txt          Backend Python deps
│   └── Dockerfile.dev            (optional) dev image with hot-reload
│
├── 📁 frontend/                   React + Tailwind SPA
│   ├── src/
│   │   ├── pages/                Dashboard, Predict, History, Metrics, Generator
│   │   ├── components/           Layout, Navbar, Sidebar
│   │   ├── hooks/                usePrediction, useMetrics
│   │   └── utils/api.js          Axios API client
│   ├── Dockerfile                node:20 build → nginx:alpine serve
│   └── nginx.conf                SPA routing + /api proxy
│
├── 📁 models/                     ← ML artifacts (git-ignored, mounted into containers)
│   ├── best_model.pth
│   ├── fairness_report.json
│   ├── metrics.json
│   └── diffusion/ema_epoch0200.pt
│
├── 📁 uploads/                    Uploaded images (git-ignored)
│
├── docker-compose.yml             Full stack orchestration
├── Dockerfile                     Backend multi-stage build
├── .env.example                   Environment variable template
├── .gitignore
├── requirements.txt               ML Python deps (PyTorch, diffusers, etc.)
│
├── diffusion_model.py             DDPM U-Net architecture
├── train_diffusion.py             Diffusion model training loop
├── generate.py                    Synthetic image generation script
├── evaluate_fid.py                FID / SSIM / PSNR evaluation
├── fairness_balancer.py           Dataset balancing with 3 strategies
├── classifier.py                  EfficientNet-B3 model definition
├── train_classifier.py            Classifier training loop
├── evaluate_classifier.py         Evaluation + metrics.json export
├── gradcam.py                     Grad-CAM implementation
├── shap_analysis.py               SHAP GradientExplainer
└── explainer.py                   Combined XAI pipeline
```

---

## Troubleshooting

### `migrate` container exits with error

```bash
docker compose logs migrate
```

**Common causes:**
- `db` not ready yet → wait and retry: `docker compose restart migrate`
- Wrong `POSTGRES_PASSWORD` → check `.env` matches

### API returns 503 on `/api/predict`

```bash
docker compose logs api | grep "ERROR\|model"
```

- If you see `"ML model not loaded"`: model file not found at `/app/models/best_model.pth`
- Set `ML_DUMMY_MODE=true` in `.env` to bypass, then `docker compose up -d api`

### GPU not detected inside container

```bash
# Test on host
nvidia-smi

# Test inside container
docker compose exec api nvidia-smi
```

If the second command fails, the nvidia-container-toolkit is not installed correctly. Follow the [official guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

### Frontend shows blank page

```bash
docker compose logs frontend
```

- Usually a build error → `docker compose build frontend --no-cache`
- Check `VITE_API_BASE_URL` in `frontend/.env` if running without Docker

### Reset everything

```bash
docker compose down -v --remove-orphans
docker system prune -f
docker compose up -d
```
