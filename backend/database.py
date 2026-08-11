
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

from config import DATABASE_URL


# SQLite needs connect_args for FastAPI threading; PostgreSQL does not
_connect_args = {'check_same_thread': False} if DATABASE_URL.startswith('sqlite') else {}
engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=not DATABASE_URL.startswith('sqlite'),
    connect_args=_connect_args,
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

Base = declarative_base()


def get_db():
    """FastAPI dependency: yields a DB session and auto-closes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Create all tables (called on startup)."""
    Base.metadata.create_all(bind=engine)
    if DATABASE_URL.startswith('sqlite'):
        _apply_sqlite_migrations()
    else:
        _apply_postgres_migrations()


def _apply_sqlite_migrations():
    """Apply lightweight SQLite-only migrations for backward compatibility."""
    with engine.begin() as conn:
        rows = conn.exec_driver_sql('PRAGMA table_info(users)').fetchall()
        existing_cols = {row[1] for row in rows}

        if 'latest_csi_score' not in existing_cols:
            conn.execute(text('ALTER TABLE users ADD COLUMN latest_csi_score INTEGER'))
        if 'total_sessions' not in existing_cols:
            conn.execute(text('ALTER TABLE users ADD COLUMN total_sessions INTEGER DEFAULT 0'))
        if 'last_session_at' not in existing_cols:
            conn.execute(text('ALTER TABLE users ADD COLUMN last_session_at DATETIME'))
        if 'gender' not in existing_cols:
            conn.execute(text('ALTER TABLE users ADD COLUMN gender VARCHAR(24)'))
        if 'password_hash' not in existing_cols:
            conn.execute(text('ALTER TABLE users ADD COLUMN password_hash VARCHAR(255)'))
        if 'google_sub' not in existing_cols:
            conn.execute(text('ALTER TABLE users ADD COLUMN google_sub VARCHAR(255)'))
            conn.execute(text('CREATE UNIQUE INDEX IF NOT EXISTS ix_users_google_sub ON users (google_sub)'))
        if 'email_verified' not in existing_cols:
            conn.execute(text('ALTER TABLE users ADD COLUMN email_verified BOOLEAN DEFAULT 0'))

        session_rows = conn.exec_driver_sql('PRAGMA table_info(sessions)').fetchall()
        existing_session_cols = {row[1] for row in session_rows}
        if 'idempotency_key' not in existing_session_cols:
            conn.execute(text('ALTER TABLE sessions ADD COLUMN idempotency_key VARCHAR(100)'))
            conn.execute(text(
                'CREATE UNIQUE INDEX IF NOT EXISTS ix_sessions_user_idempotency '
                'ON sessions (user_id, idempotency_key)'
            ))


def _apply_postgres_migrations():
    """
    Add any columns/indexes that exist on the current models but may not exist yet on an
    already-provisioned Postgres database. create_all() only creates missing TABLES — it never
    alters a table that already exists, so a column added to a model after the database was
    first provisioned would otherwise never reach production until someone ran the ALTER TABLE
    by hand. Postgres's IF NOT EXISTS DDL makes every statement here idempotent, so this is safe
    to run unconditionally on every startup: on a brand-new database create_all() already created
    every column, so each statement below is a no-op; on an older one, it backfills exactly the
    columns that are actually missing.
    """
    with engine.begin() as conn:
        conn.execute(text('ALTER TABLE users ADD COLUMN IF NOT EXISTS latest_csi_score INTEGER'))
        conn.execute(text('ALTER TABLE users ADD COLUMN IF NOT EXISTS total_sessions INTEGER DEFAULT 0'))
        conn.execute(text('ALTER TABLE users ADD COLUMN IF NOT EXISTS last_session_at TIMESTAMP'))
        conn.execute(text('ALTER TABLE users ADD COLUMN IF NOT EXISTS gender VARCHAR(24)'))
        conn.execute(text('ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash VARCHAR(255)'))
        conn.execute(text('ALTER TABLE users ADD COLUMN IF NOT EXISTS google_sub VARCHAR(255)'))
        conn.execute(text('CREATE UNIQUE INDEX IF NOT EXISTS ix_users_google_sub ON users (google_sub)'))
        conn.execute(text('ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verified BOOLEAN DEFAULT FALSE'))
        conn.execute(text('ALTER TABLE sessions ADD COLUMN IF NOT EXISTS idempotency_key VARCHAR(100)'))
        conn.execute(text(
            'CREATE UNIQUE INDEX IF NOT EXISTS ix_sessions_user_idempotency '
            'ON sessions (user_id, idempotency_key)'
        ))
