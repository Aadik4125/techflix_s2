"""
Shared pytest fixtures for the CogniVara Python backend.

Critical ordering constraint: DATABASE_URL (and other env vars the backend reads
at import time via backend/config.py) must be set BEFORE any backend module is
imported anywhere in the test session — config.py reads them once at import time
and Python caches the module after that. This file is imported by pytest before
test collection, so it's the right place to do it.
"""

import os
import sys
import tempfile

_TEST_DB_FD, _TEST_DB_PATH = tempfile.mkstemp(suffix='.db', prefix='cognivara_test_')
os.close(_TEST_DB_FD)

os.environ['DATABASE_URL'] = f'sqlite:///{_TEST_DB_PATH}'
os.environ.setdefault('FAST_ANALYSIS_MODE', 'true')  # keep tests fast; real extraction is exercised separately
os.environ.setdefault('GOOGLE_CLIENT_ID', '')
os.environ.setdefault('RESEND_API_KEY', '')
os.environ.setdefault('CORS_ORIGINS', 'http://localhost:3000')

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND_DIR = os.path.join(ROOT_DIR, 'backend')
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from database import SessionLocal, create_tables, engine  # noqa: E402
from main import app  # noqa: E402


@pytest.fixture(scope='session', autouse=True)
def _test_database():
    create_tables()
    yield
    engine.dispose()
    try:
        os.remove(_TEST_DB_PATH)
    except OSError:
        pass


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def signup(client, email='test@example.com', password='testpass123', name='Test User', age=30, gender='Other'):
    """Helper: sign up a fresh user and return (user_id, token)."""
    resp = client.post(
        '/api/auth/signup',
        data={'name': name, 'email': email, 'password': password, 'age': age, 'gender': gender},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    return body['user_id'], body['token']


def auth_headers(token):
    return {'Authorization': f'Bearer {token}'}
