# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

FROM base AS deps
COPY apps/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS runtime
COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Every file create_ensemble_from_artifacts() needs must be present in the
# image; relying on the compose bind-mount made a plain `docker run` fail with
# FileNotFoundError on the joblib/manifest files.
COPY ml/artifacts/ ./ml/artifacts/
COPY apps/api/tbcare_api ./tbcare_api

ENV PORT=8080
EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "4", "--preload", "--timeout", "60", "tbcare_api.app:create_app()"]