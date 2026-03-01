FROM python:3.11-slim

WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy dependency manifests first for layer caching
COPY pyproject.toml uv.lock* ./

# Install production dependencies only
RUN uv sync --no-dev

# Copy application code
COPY src/ src/
COPY alembic/ alembic/
COPY alembic.ini .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.api.app:create_app", "--host", "0.0.0.0", "--port", "8000", "--factory"]
