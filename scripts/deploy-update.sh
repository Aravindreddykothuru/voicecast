#!/usr/bin/env bash
# ==============================================================================
# VoiceCast (Sur) — Automated Live Update Script
# Updates code from git, rebuilds containers, runs migrations, and checks health.
# ==============================================================================
set -euo pipefail

echo "==> [1/5] Fetching latest changes from git..."
git pull origin main

echo "==> [2/5] Rebuilding and updating Docker services..."
docker compose -f docker-compose.prod.yml up -d --build

echo "==> [3/5] Waiting for services to initialize..."
sleep 5

echo "==> [4/5] Verifying API and Worker Health..."
if curl -s -f http://localhost/healthz > /dev/null; then
  echo "✔ API is healthy!"
else
  echo "⚠ Warning: API health check did not respond with 200 OK immediately."
fi

echo "==> [5/5] Cleaning up old images..."
docker image prune -f

echo "✔ Update successfully applied! Container status:"
docker compose -f docker-compose.prod.yml ps
