# ==============================================================================
# MLTrainer - Multi-Stage Dockerfile for Hugging Face Spaces & Production
# ==============================================================================

# Stage 1: Build Frontend (React 19 + TypeScript + Vite)
FROM node:20-alpine AS frontend-builder
WORKDIR /frontend

COPY frontend/package*.json ./
RUN npm ci || npm install

COPY frontend/ ./
RUN npm run build

# Stage 2: Python Backend & Static Server (Hugging Face Spaces)
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HOST=0.0.0.0

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy backend code, models, and demo dataset scripts
COPY backend/ /app/backend/
COPY download_dataset.py /app/download_dataset.py
COPY README.md /app/README.md

# Copy built frontend distribution from builder
COPY --from=frontend-builder /frontend/dist /app/frontend/dist

# Create runtime directories for dataset storage, model checkpoints, and exports
RUN mkdir -p /app/data /app/checkpoints /app/exports /app/temp_downloads

# Hugging Face Spaces runs containers with UID 1000
RUN useradd -m -u 1000 appuser && \
    chown -R 1000:1000 /app && \
    chmod -R 777 /app/data /app/checkpoints /app/exports /app/temp_downloads

USER 1000

# Hugging Face Spaces expects traffic on port 7860
EXPOSE 7860

# Launch Uvicorn server on port 7860 (serves both API and frontend UI)
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
