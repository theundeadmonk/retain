FROM python:3.12-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock README.md ./
COPY src/ src/
COPY alembic.ini ./
COPY alembic/ alembic/
RUN uv sync --frozen --no-dev

EXPOSE 8000
CMD sh -c ".venv/bin/alembic upgrade head && .venv/bin/uvicorn retain.main:app --host 0.0.0.0 --port 8000"
