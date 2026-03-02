#!/usr/bin/env bash
# Start infrastructure (PostgreSQL + Redis) and run migrations.
# Usage: ./scripts/infra-up.sh
#
# Idempotent: safe to run repeatedly. Alembic skips already-applied migrations.
# Requires: docker compose, uv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "Starting PostgreSQL and Redis..."
sudo docker compose up -d postgres redis

echo "Waiting for PostgreSQL to be healthy..."
until sudo docker compose exec -T postgres pg_isready -U geopol -q 2>/dev/null; do
    sleep 1
done
echo "PostgreSQL is ready."

echo "Running Alembic migrations..."
uv run alembic upgrade head

echo "Infrastructure ready."
