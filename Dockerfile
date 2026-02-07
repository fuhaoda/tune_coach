# syntax=docker/dockerfile:1.7

FROM node:20-alpine AS web-build
WORKDIR /webapp
COPY webapp/package.json webapp/package-lock.json* ./
RUN npm install
COPY webapp ./
RUN npm run build

FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY tune_coach ./tune_coach
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir .

COPY --from=web-build /webapp/dist ./webapp/dist

EXPOSE 8000
CMD ["uvicorn", "tune_coach.web.server:app", "--host", "0.0.0.0", "--port", "8000"]
