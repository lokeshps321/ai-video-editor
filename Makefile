SHELL := /bin/bash

.PHONY: backend-dev backend-test frontend-dev frontend-build ci docker-build docker-up docker-down worker-dev redis-dev

backend-dev:
	cd backend && if [ -x .venv/bin/python ]; then .venv/bin/python -m uvicorn app.main:app --reload --port 8000; else python -m uvicorn app.main:app --reload --port 8000; fi

backend-test:
	cd backend && if [ -x .venv/bin/pytest ]; then .venv/bin/pytest -q; else pytest -q; fi

language-audit:
	cd backend && python3 scripts/language_live_audit.py

frontend-dev:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

ci:
	cd backend && if [ -x .venv/bin/pytest ]; then .venv/bin/pytest -q; else pytest -q; fi
	cd frontend && npm run build

docker-build:
	docker compose build

docker-up:
	docker compose up --build

docker-down:
	docker compose down

# Local development with RQ workers (requires Redis)
redis-dev:
	@echo "Starting Redis server..."
	redis-server

worker-dev:
	cd backend && if [ -x .venv/bin/python ]; then .venv/bin/python worker.py --queues renders,ingests --verbosity INFO; else python worker.py --queues renders,ingests --verbosity INFO; fi
