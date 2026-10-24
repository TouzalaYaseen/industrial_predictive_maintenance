# ==============================================================================
# Production Dockerfile for Industrial Predictive Maintenance Engine
# ==============================================================================

# 1. Base Image: Slim Debian with standard glibc for compiled ML wheels (XGBoost/SHAP)
FROM python:3.11-slim

# 2. Environment Configurations
# - PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files to disk
# - PYTHONUNBUFFERED: Ensures stdout/stderr stream directly to terminal/logs
# - PYTHONPATH: Ensures imports like 'from api.schemas' and 'from src.features' resolve cleanly
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app"

# 3. Set Working Directory inside the container
WORKDIR /app

# 4. Dependency Layer Caching
# Copy dependency specification and install BEFORE copying application code.
# This ensures application edits do NOT bust the pip download/install cache.
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy Application Code & Assets
# Leverages .dockerignore to filter out dev notebooks, .git, and local virtual environments
COPY . .

# 6. Production Security: Non-Root User
# Running as an unprivileged user mitigates container breakout vulnerabilities
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# 7. Expose Container Port
EXPOSE 8000

# 8. Container Healthcheck
# Periodically probes the /health endpoint to ensure model and server are operational
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; port = os.environ.get('PORT', '8000'); urllib.request.urlopen(f'http://localhost:{port}/health')" || exit 1

# 9. Container Entrypoint
# Launch production FastAPI microservice binding to all network interfaces (0.0.0.0)
# Dynamically resolves PORT from cloud environments (e.g., Render) with fallback to 8000
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
