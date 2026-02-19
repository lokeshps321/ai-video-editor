SHELL := /bin/bash

.PHONY: backend-dev backend-test frontend-dev frontend-build

backend-dev:
	cd backend && uvicorn app.main:app --reload --port 8000

backend-test:
	cd backend && pytest -q

frontend-dev:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

