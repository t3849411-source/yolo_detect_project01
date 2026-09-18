FROM node:22-alpine AS frontend-builder

WORKDIR /build/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/index.html frontend/vite.config.js ./
COPY frontend/src ./src
RUN npm run build


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000 \
    MODEL_PATH=/app/models/best.pt \
    FRONTEND_DIST=/app/frontend/dist \
    RESULTS_DIR=/app/outputs/results

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.deploy.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements.deploy.txt

COPY backend ./backend
COPY models/best.pt ./models/best.pt
COPY --from=frontend-builder /build/frontend/dist ./frontend/dist

RUN mkdir -p /app/outputs/results \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 10000

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
