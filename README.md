# CogniVara

CogniVara is a multi-service cognitive speech analysis project with:

- A static frontend (`frontend/`)
- A Node service for transcription, analysis proxy, and LLM-generated check-in prompts (`services/node/`)
- A Python FastAPI backend for cognitive analytics and authentication (`backend/`)

The product has two entry points: a no-login **demo** (`POST /api/demo/*`, three recordings, one-time result, nothing persisted) and a real, authenticated **check-in** flow (`POST /api/upload`, one recording per check-in, unlimited check-ins, builds a real longitudinal baseline). See `backend/routes/auth.py` and `backend/routes/demo.py`.

## Repository Status

The repository is functional and separated by runtime concern.

The backend previously had two parallel layouts: a live flat structure (`backend/routes`, `backend/services`, `backend/models`) and an unused package-oriented structure (`backend/app/...`, plus `backend/alembic/`) that was never wired into `main.py`, the Dockerfile, or `render.yaml`, and had no working Celery producer despite a `docker-compose.yml` worker service referencing it. That dead package has been removed; `backend/routes`, `backend/services`, and `backend/models` are the single canonical backend layout.

See [PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for the authoritative structure guide.

## Project Structure

```text
test_case_2/
  frontend/
    index.html
    styles/
      main.css
    scripts/
      app.js
      background.js
  services/
    node/
      server.js
      hf.service.js
  backend/
    main.py
    config.py
    database.py
    routes/
    services/
    models/
    uploads/
    requirements.txt
    cognivara.db
  tests/
    node/
      test_request.js
  docs/
    PROJECT_OVERVIEW.md
    FRONTEND_STRUCTURE.md
    reports/
      PROJECT_TECH_STACK_REPORT_v2.pdf
  tools/
    generate_pdf.py
  archive/
    frontend/
      test_1.html
      new_frontend.txt
      components/       # unused componentized layout, superseded by app.js
      scripts/          # api.js, recording.js, ui.js, component-loader.js
  package.json
  package-lock.json
  pyrightconfig.json
  .env
  .env.example
```

## Structure Notes

- `frontend/` contains the browser application. It is intentionally small: `index.html`, `main.css`, `app.js`, `background.js` are the entire runtime frontend.
- `services/node/` contains the Node.js service that serves the frontend and proxies external inference calls.
- `backend/` contains the Python backend (`routes/`, `services/`, `models/`), with its own ad hoc SQLite schema upgrades in `database.py` — there is no separate migrations tool wired in.
- `backend/uploads/` and `backend/cognivara.db` are local runtime artifacts and should not be treated as source code.
- `archive/` is reference material only, including a superseded componentized frontend layout that was never wired into `index.html`.

## Run Node Service

```bash
npm install
npm start
```

- Serves frontend from `frontend/`
- Runs API endpoints at `http://localhost:3000`

## Run Python Backend

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r backend/requirements.txt
python backend/main.py
```

FastAPI defaults to `http://localhost:8000`.

## Utility Commands

```bash
npm run test:node
```

Runs `tests/node/test_request.js` against the Node endpoint.
