# CogniVara - Project Overview

## Summary

CogniVara processes voice recordings and transcript-derived features to estimate cognitive risk trends. The repository is organized into separated frontend, Node service, Python backend, tests, tools, docs, and archive assets.

## Architecture

- `frontend/`: static UI application
- `services/node/`: Express API for transcription and analysis flow
- `backend/`: FastAPI service for cognitive analytics, persistence, and dashboard APIs
- `tests/node/`: Node API smoke script
- `docs/`: project documentation and reports
- `tools/`: helper scripts
- `archive/`: preserved legacy frontend snapshots

## Frontend

- Entry: `frontend/index.html`
- Styling: `frontend/styles/main.css`
- Main UI logic: `frontend/scripts/app.js`
- Background animation: `frontend/scripts/background.js`
- This is the entire runtime frontend. An earlier componentized layout (`api.js`, `recording.js`, `ui.js`, `component-loader.js`, `components/*.html`) was never wired into `index.html` and has been moved to `archive/frontend/`.

## Node Service

- Entry: `services/node/server.js`
- HF integration: `services/node/hf.service.js`
- Groq integration: `services/node/groq.service.js` (transcript semantic analysis + personalized daily check-in prompts)
- Serves static frontend from `frontend/`
- Main endpoints:
  - `GET /health`
  - `POST /transcribe`
  - `POST /analyze` (HF emotion/sentiment + Groq semantic pass)
  - `POST /api/daily-prompt` (LLM-generated check-in prompt, personalizes using recent transcript history if provided)

## Python Backend

- Entry: `backend/main.py`
- Routes:
  - `backend/routes/auth.py` — signup/login/logout/me, issues opaque bearer tokens
  - `backend/routes/audio.py` — `POST /api/upload`, authenticated, persists real check-in sessions
  - `backend/routes/analysis.py`
  - `backend/routes/dashboard.py` — `/api/dashboard/me`, `/api/baseline/me`, authenticated
  - `backend/routes/demo.py` — `/api/demo/extract`, `/api/demo/finalize`; unauthenticated, stateless, no DB writes
- Services:
  - `backend/services/preprocessing.py`
  - `backend/services/acoustic_features.py`
  - `backend/services/temporal_features.py`
  - `backend/services/linguistic_features.py`
  - `backend/services/feature_pipeline.py` — shared audio-to-feature glue used by both `/api/upload` and `/api/demo/extract`
  - `backend/services/baseline.py` — includes pure `compute_baseline_stats`/`compute_z_scores` usable without a DB session
  - `backend/services/drift.py` — includes pure `compute_drift_stats`
  - `backend/services/csi.py`
  - `backend/services/security.py` — password hashing/verification, token generation (stdlib only)
- Models:
  - `backend/models/user.py`
  - `backend/models/session.py`
  - `backend/models/baseline.py`
  - `backend/models/auth_token.py`

## Data and Reports

- SQLite DB: `backend/cognivara.db`
- Audio uploads: `backend/uploads/`
- Tech stack report: `docs/reports/PROJECT_TECH_STACK_REPORT_v2.pdf`

## Run

### Node

```powershell
npm install
npm start
```

### Python

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
python backend/main.py
```

Generated on: 2026-02-27
