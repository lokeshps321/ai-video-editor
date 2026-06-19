#!/usr/bin/env bash
set -euo pipefail

cleanup() {
  jobs -p | xargs -r kill
}

trap cleanup EXIT INT TERM

BACKEND_PORT="${BACKEND_PORT:-8001}"
FRONTEND_PORT="${FRONTEND_PORT:-5174}"
export TRANSCRIBE_BACKEND="${TRANSCRIBE_BACKEND:-auto}"
export TRANSCRIBE_ALLOW_MOCK_FALLBACK="${TRANSCRIBE_ALLOW_MOCK_FALLBACK:-false}"

cd backend
source .venv/bin/activate
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port "${BACKEND_PORT}" &

cd ../frontend
VITE_API_BASE="http://localhost:${BACKEND_PORT}" npm run dev -- --host 0.0.0.0 --port "${FRONTEND_PORT}" &

wait
