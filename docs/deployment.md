# ClipMind Deployment

## Local Secret Handling

1. Copy `backend/.env.example` to `backend/.env`
2. Copy `backend/.env.local.example` to `backend/.env.local`
3. Put real API keys only in `backend/.env.local`

`backend/.env.local` is ignored by git and overrides values from `backend/.env`.

## Production Config

1. Copy `backend/.env.production.example` to `backend/.env.production`
2. Fill in real values for:
   - `DATABASE_URL`
   - `ALLOWED_ORIGINS`
   - `GROQ_API_KEY`
   - any stock / planner / generative provider keys you use
3. Keep `backend/.env.production` out of git

## Docker Compose

```bash
cp backend/.env.production.example backend/.env.production
docker compose build
docker compose up -d
```

The production stack includes:

- `db`: PostgreSQL 16
- `backend`: FastAPI + Uvicorn
- `frontend`: Nginx serving the Vite build and proxying `/api` and `/static`

## Deployment Checklist

- Rotate any keys that were ever committed to git before deploying.
- Replace the default PostgreSQL password.
- Set `ALLOWED_ORIGINS` to the real frontend domain.
- Keep the backend private behind the frontend reverse proxy when possible.
- Put the public app behind auth if this will be internet-facing.
- Back up the PostgreSQL volume and uploaded media volumes.
