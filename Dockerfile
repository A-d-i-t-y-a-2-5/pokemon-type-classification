FROM python:3.10-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y \
    build-essential \
    curl

WORKDIR /app
COPY pyproject.toml uv.lock ./

RUN uv sync

COPY ./app ./app

EXPOSE 8000
EXPOSE 8501