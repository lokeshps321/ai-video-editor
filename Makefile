SHELL := /bin/bash

.PHONY: backend-dev backend-test frontend-dev frontend-build ci docker-build docker-up docker-down

backend-dev:
	cd backend && if [ -x .venv/bin/uvicorn ]; then .venv/bin/uvicorn app.main:app --reload --port 8000; else uvicorn app.main:app --reload --port 8000; fi

backend-test:
	cd backend && if [ -x .venv/bin/pytest ]; then .venv/bin/pytest -q; else pytest -q; fi

frontend-dev:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

ci:
	cd backend && if [ -x .venv/bin/pytest ]; then .venv/bin/pytest tests/test_transcription_service.py tests/test_transcript_router_helpers.py -q; else pytest tests/test_transcription_service.py tests/test_transcript_router_helpers.py -q; fi
	cd frontend && npm run build

docker-build:
	docker compose build

docker-up:
	docker compose up --build

docker-down:
	docker compose down
