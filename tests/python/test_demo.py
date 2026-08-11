"""
Demo mode must genuinely be stateless: no auth required, and no trace left in the
database — that's the entire point of a "one-time, nothing saved" demo.
"""

from database import SessionLocal
from models.baseline import Baseline
from models.session import Session
from models.user import User


def _counts():
    db = SessionLocal()
    try:
        return {
            'users': db.query(User).count(),
            'sessions': db.query(Session).count(),
            'baselines': db.query(Baseline).count(),
        }
    finally:
        db.close()


def test_demo_extract_requires_no_authentication(client):
    resp = client.post(
        '/api/demo/extract',
        files={'audio': ('test.wav', b'fake-audio-bytes', 'audio/wav')},
        data={'transcript': 'A quick demo transcript.'},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert 'acoustic_features' in body
    assert 'linguistic_features' in body


def test_demo_extract_writes_nothing_to_the_database(client):
    before = _counts()
    for i in range(3):
        resp = client.post(
            '/api/demo/extract',
            files={'audio': ('test.wav', b'fake-audio-bytes', 'audio/wav')},
            data={'transcript': f'Demo session {i}.'},
        )
        assert resp.status_code == 200
    after = _counts()
    assert after == before


def test_demo_finalize_requires_at_least_two_sessions(client):
    resp = client.post('/api/demo/finalize', json={'sessions': [{'pitch_mean': 100.0}]})
    assert resp.status_code == 400


def test_demo_finalize_produces_real_scores_and_writes_nothing(client):
    before = _counts()

    sessions = []
    for _ in range(3):
        extract_resp = client.post(
            '/api/demo/extract',
            files={'audio': ('test.wav', b'fake-audio-bytes', 'audio/wav')},
            data={'transcript': 'I had a pretty calm and ordinary day today.'},
        )
        body = extract_resp.json()
        merged = {**body['acoustic_features'], **body['temporal_features'], **body['linguistic_features']}
        sessions.append(merged)

    resp = client.post('/api/demo/finalize', json={'sessions': sessions})
    assert resp.status_code == 200
    body = resp.json()
    assert body['baseline_ready'] is True
    assert 'csi_score' in body['csi']
    assert 0 <= body['csi']['csi_score'] <= 100

    after = _counts()
    assert after == before
