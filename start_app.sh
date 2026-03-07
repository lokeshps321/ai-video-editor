#!/usr/bin/env bash
set -euo pipefail

cleanup() {
  jobs -p | xargs -r kill
}

trap cleanup EXIT INT TERM

cd backend
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

cd ../frontend
npm run dev -- --host 0.0.0.0 --port 5173 &

wait
