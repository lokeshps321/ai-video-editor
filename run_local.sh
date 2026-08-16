#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
API_BASE="http://127.0.0.1:${BACKEND_PORT}"

cleanup() {
  jobs -p | xargs -r kill
}

trap cleanup EXIT INT TERM

cd "${ROOT_DIR}/backend"
if [ ! -x .venv/bin/python ]; then
  echo "Backend venv missing. Run:"
  echo "  cd backend && python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

. .venv/bin/activate
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port "${BACKEND_PORT}" &

cd "${ROOT_DIR}/frontend"
if [ ! -d node_modules ]; then
  npm install
fi

# Polling avoids ENOSPC when Linux inotify watcher limits are exhausted.
CHOKIDAR_USEPOLLING=1 \
WATCHPACK_POLLING=true \
VITE_API_BASE="${API_BASE}" \
VITE_TIMELINE_CORE_V2=true \
npm run dev -- --host 127.0.0.1 --port "${FRONTEND_PORT}" &

echo "Backend: ${API_BASE}"
echo "Frontend: http://127.0.0.1:${FRONTEND_PORT}"
echo "Timeline Core V2: enabled (kratos test)"

wait
