from contextlib import asynccontextmanager
import logging
import os
import sys
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# Ensure backend-local absolute imports (config/database/routes/...) work
# when launched as either:
# - python backend/main.py
# - uvicorn backend.main:app
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO').upper(),
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger('cognivara')

from config import (
    CORS_ORIGINS,
    DATABASE_URL,
    ENVIRONMENT,
    FASTAPI_PORT,
    GOOGLE_CLIENT_ID,
    RESEND_API_KEY,
    SENTRY_DSN,
)
from database import SessionLocal, create_tables
from models.auth_token import AuthToken
from models.session import Session
from models.user import User

# Must run before the FastAPI app is constructed so Sentry's Starlette/FastAPI integration
# can actually instrument it. Fully optional and silent when SENTRY_DSN isn't set — nothing
# about this can block startup.
if SENTRY_DSN:
    import sentry_sdk

    sentry_sdk.init(dsn=SENTRY_DSN, environment=ENVIRONMENT)
else:
    logger.warning('SENTRY_DSN missing. Unhandled errors will only be visible in local/platform logs.')

# Import routes
from routes.auth import router as auth_router
from routes.audio import router as audio_router
from routes.analysis import router as analysis_router
from routes.dashboard import router as dashboard_router
from routes.demo import router as demo_router


def _warm_up_audio_pipeline() -> None:
    """
    librosa's first decode/resample call in a process pays a one-time JIT/backend
    warmup cost (observed ~5-10s even for a tiny clip; every call after is ~0ms).
    Pay that cost once at startup instead of during a user's first recording.
    """
    try:
        import io

        import numpy as np
        import librosa
        from scipy.io import wavfile

        sample_rate = 16000
        silence = np.zeros(int(sample_rate * 0.5), dtype=np.int16)
        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, silence)
        buffer.seek(0)

        librosa.load(buffer, sr=sample_rate, mono=True)
    except Exception:
        # Warmup is an optimization, never a requirement — must not block startup.
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):

    create_tables()

    # Do not download at startup. Offline environments should still boot.
    import nltk

    for resource in [
        'tokenizers/punkt_tab',
        'taggers/averaged_perceptron_tagger_eng',
        'corpora/stopwords',
    ]:
        try:
            nltk.data.find(resource)
        except LookupError:
            pass

    _warm_up_audio_pipeline()

    yield


app = FastAPI(
    title='CogniVara - Cognitive Risk Analysis API',
    version='1.0.0',
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.middleware('http')
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.monotonic() - start) * 1000
        logger.exception('%s %s -> unhandled error (%.1fms)', request.method, request.url.path, duration_ms)
        raise
    duration_ms = (time.monotonic() - start) * 1000
    log = logger.warning if response.status_code >= 500 else logger.info
    log('%s %s -> %s (%.1fms)', request.method, request.url.path, response.status_code, duration_ms)
    return response

app.include_router(auth_router, prefix='/api/auth', tags=['Auth'])
app.include_router(audio_router, prefix='/api', tags=['Audio'])
app.include_router(analysis_router, prefix='/api', tags=['Analysis'])
app.include_router(dashboard_router, prefix='/api', tags=['Dashboard'])
app.include_router(demo_router, prefix='/api/demo', tags=['Demo'])


@app.get('/api/health')
def health_check():
    db_kind = 'postgres' if DATABASE_URL.startswith('postgresql+') else 'sqlite'
    users = None
    sessions = None
    db = None
    try:
        db = SessionLocal()
        users = db.query(User).count()
        sessions = db.query(Session).count()
    except Exception:
        pass
    finally:
        try:
            if db is not None:
                db.close()
        except Exception:
            pass

    return {
        'status': 'ok',
        'service': 'cognivara-backend',
        'database': db_kind,
        'users': users,
        'sessions': sessions,
        'integrations': {
            'google_signin_configured': bool(GOOGLE_CLIENT_ID),
            'email_verification_configured': bool(RESEND_API_KEY),
            'error_tracking_configured': bool(SENTRY_DSN),
        },
    }


@app.get('/')
def root():
    return {'service': 'cognivara-backend', 'status': 'ok', 'docs': '/docs'}


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=FASTAPI_PORT, reload=False)
