"""
CogniVara — Configuration
Loads environment variables for database, server, and analysis settings.
"""

import os
from dotenv import load_dotenv


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {'1', 'true', 'yes', 'on'}

# Load .env from project root (one level up from backend/)
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# ── Database ──────────────────────────────────────────────
# Default: SQLite locally, but require DATABASE_URL on Render to avoid silent fallback.
_database_url = os.getenv('DATABASE_URL', '').strip()
if _database_url:
    DATABASE_URL = _database_url
else:
    if os.getenv('RENDER') == 'true':
        raise RuntimeError('DATABASE_URL is required on Render. Connect your Postgres instance.')
    DATABASE_URL = 'sqlite:///' + os.path.join(os.path.dirname(__file__), 'cognivara.db')

if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql+psycopg://', 1)
elif DATABASE_URL.startswith('postgresql://'):
    DATABASE_URL = DATABASE_URL.replace('postgresql://', 'postgresql+psycopg://', 1)

# ── Server ────────────────────────────────────────────────
FASTAPI_PORT = int(os.getenv('FASTAPI_PORT', '8000'))
CORS_ORIGINS = os.getenv(
    'CORS_ORIGINS',
    'http://localhost:3000,http://127.0.0.1:3000,http://localhost:5500,http://127.0.0.1:5500,null',
).split(',')

# ── Analysis Tuning ──────────────────────────────────────
BASELINE_SESSION_COUNT = int(os.getenv('BASELINE_SESSION_COUNT', '3'))
DRIFT_ROLLING_WINDOW = int(os.getenv('DRIFT_ROLLING_WINDOW', '3'))
# NOTE: services/baseline.py's compute_z_scores tanh-compresses + shrinks (_Z_TANH_SCALE * _Z_SHRINK)
# so |z| asymptotically maxes out at 1.5 * 0.90 = 1.35, no matter how extreme the raw deviation is.
# A threshold >= 1.35 (the old default of 1.5) can mathematically never be crossed, which silently
# disabled all drift flagging. Keep this below 1.35 with real headroom for genuine deviations.
DRIFT_Z_THRESHOLD = float(os.getenv('DRIFT_Z_THRESHOLD', '1.0'))

# ── Auth ──────────────────────────────────────────────────
AUTH_TOKEN_TTL_DAYS = int(os.getenv('AUTH_TOKEN_TTL_DAYS', '30'))
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID', '').strip()

# ── Error tracking (Sentry) ───────────────────────────────
SENTRY_DSN = os.getenv('SENTRY_DSN', '').strip()
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')

# ── Email (Resend) ────────────────────────────────────────
RESEND_API_KEY = os.getenv('RESEND_API_KEY', '').strip()
RESEND_FROM_EMAIL = os.getenv('RESEND_FROM_EMAIL', 'onboarding@resend.dev')
FRONTEND_BASE_URL = os.getenv('FRONTEND_BASE_URL', 'http://localhost:3000')
EMAIL_VERIFICATION_TTL_HOURS = int(os.getenv('EMAIL_VERIFICATION_TTL_HOURS', '24'))

# ── Audio ─────────────────────────────────────────────────
AUDIO_UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(AUDIO_UPLOAD_DIR, exist_ok=True)

# Object storage for raw recordings (S3-compatible: real AWS S3, Cloudflare R2, Backblaze
# B2, MinIO, ...). Falls back to local disk when unset — matches every other optional
# integration in this codebase. Local disk is fine for dev, but on a host with an ephemeral
# filesystem (e.g. Render's free tier) local files are lost on every redeploy/restart, so
# this should be configured before relying on raw audio surviving in production.
S3_BUCKET = os.getenv('S3_BUCKET', '').strip()
S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL', '').strip()  # empty = real AWS S3
S3_ACCESS_KEY_ID = os.getenv('S3_ACCESS_KEY_ID', '').strip()
S3_SECRET_ACCESS_KEY = os.getenv('S3_SECRET_ACCESS_KEY', '').strip()
S3_REGION = os.getenv('S3_REGION', 'us-east-1').strip()  # R2 users: set this to "auto"

# ── CSI Weights (equal by default) ────────────────────────
CSI_WEIGHTS = {
    'mfcc_variance_avg': 0.10,
    'pitch_var': 0.07,
    'jitter_local': 0.06,
    'hnr_mean': 0.07,
    'shimmer_local': 0.06,
    'spectral_centroid_mean': 0.05,
    'energy_var': 0.06,
    'speech_rate': 0.08,
    'response_latency': 0.07,
    'rhythm_consistency': 0.06,
    'pause_variability': 0.08,
    'speed_variability': 0.07,
    'mean_pause_duration': 0.05,
    'lexical_diversity': 0.08,
    'filler_ratio': 0.04,
    'content_word_ratio': 0.03,
    'syntactic_complexity': 0.04,
    'vocabulary_richness': 0.05,
    'sentence_length_mean': 0.03,
    'speech_ratio': 0.01,
    'speech_segment_count': 0.01,
}

# Optional performance mode for constrained cloud instances.
FAST_ANALYSIS_MODE = _env_bool(
    'FAST_ANALYSIS_MODE',
    default=False,
)
