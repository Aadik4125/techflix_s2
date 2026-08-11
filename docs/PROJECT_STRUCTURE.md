# Project Structure

This repository is organized as a multi-service application with a browser frontend, a Node.js service, and a Python backend.

## Current Assessment

The project is separated by major runtime concerns. The backend previously carried a second, unused package-oriented layout under `backend/app/...` (plus a `backend/alembic/` migration setup that only targeted that package's models). Nothing imported it — `main.py`, the Dockerfile, and `render.yaml` all ran the flat layout, and the `docker-compose.yml` worker service that referenced it had no producer ever enqueuing work. It has been removed; `backend/routes`, `backend/services`, and `backend/models` are now the single canonical backend layout.

## Canonical Layout

```text
test_case_2/
  backend/
    main.py       # backend process entry point
    config.py     # env-driven settings
    database.py   # engine/session + ad hoc SQLite schema upgrades
    routes/       # FastAPI routers (auth, audio, analysis, dashboard, demo)
    services/     # feature extraction, baseline, drift, CSI scoring, security
    models/       # SQLAlchemy models (user, session, baseline, auth_token)
    uploads/      # runtime audio storage, not source code
    requirements.txt
    Dockerfile
  frontend/
    scripts/      # browser-side logic (app.js, background.js only)
    styles/       # CSS
    index.html
  services/
    node/
      server.js   # Node service entry point
      hf.service.js
      groq.service.js
  docs/
    reports/      # generated reports and architecture writeups
    PROJECT_OVERVIEW.md
    FRONTEND_STRUCTURE.md
    PROJECT_STRUCTURE.md
  tests/
    node/
  tools/
  archive/        # historical snapshots only; not active app code, includes a
                   # superseded componentized frontend layout (see FRONTEND_STRUCTURE.md)
```

## Repository Rules

- `backend/routes`, `backend/services`, and `backend/models` are the backend source of truth. There is no second backend layout to reconcile against.
- `backend/uploads/` and `backend/cognivara.db` are runtime artifacts, not source code.
- `archive/` is for reference only and should not be treated as active implementation.
- `docs/reports/` is acceptable for generated reports, but those reports should not define the runtime structure.
- If real background-job processing (Celery, etc.) or S3-backed storage becomes necessary, wire it directly into the flat `backend/` layout rather than reintroducing a parallel package — half-built parallel structures are exactly what was just removed.
- If real schema migrations become necessary beyond the ad hoc `_apply_sqlite_migrations` upgrades in `database.py`, set up Alembic against the models in `backend/models/`, not a separate model layer.

## What Already Looks Professional

- Clear separation between frontend, backend, and service layers
- Dedicated `docs/`, `tests/`, and `tools/` directories
- Environment files separated from source code
- A single, unambiguous backend entry point and layout

## What Still Needs Attention

1. `database.py` manages schema via hand-written `ALTER TABLE` statements rather than a real migration tool. Fine at current scale; revisit if the schema grows.
2. Move backend runtime storage (`backend/uploads/`, `backend/cognivara.db`) into a dedicated non-source path such as `backend/data/` or `var/` if this deploys somewhere with persistent volumes.
3. Add a Python test suite for the backend — only `tests/node/` exists today.

## Efficiency Verdict

Folder structure efficiency: good.

- Organization by runtime area: strong
- Source-of-truth clarity: strong (single backend layout, no dead parallel package)
- Runtime artifact isolation: moderate
- Maintainability for a growing team: good
