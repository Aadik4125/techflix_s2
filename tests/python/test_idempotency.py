"""
/api/upload idempotency: a retried request for the same recording (same idempotency_key)
must replay the original result instead of creating a second session row — this is what
protects the baseline/session_number sequence from a flaky-network retry silently double-
counting one physical recording as two.
"""

from conftest import auth_headers, signup


def _upload(client, token, transcript, idempotency_key=None):
    data = {'transcript': transcript}
    if idempotency_key is not None:
        data['idempotency_key'] = idempotency_key
    return client.post(
        '/api/upload',
        headers=auth_headers(token),
        files={'audio': ('test.wav', b'fake-audio-bytes', 'audio/wav')},
        data=data,
    )


def test_same_idempotency_key_replays_instead_of_duplicating(client):
    _, token = signup(client, email='idem-a@example.com')

    first = _upload(client, token, 'First attempt at session one.', idempotency_key='key-1')
    assert first.status_code == 200
    second = _upload(client, token, 'First attempt at session one.', idempotency_key='key-1')
    assert second.status_code == 200

    assert first.json()['session_id'] == second.json()['session_id']
    assert second.json()['replayed'] is True
    assert first.json().get('replayed') is not True  # the original response is not itself a replay

    sessions = client.get('/api/sessions/me', headers=auth_headers(token)).json()
    assert sessions['session_count'] == 1


def test_different_idempotency_keys_create_separate_sessions(client):
    _, token = signup(client, email='idem-b@example.com')

    first = _upload(client, token, 'Session one.', idempotency_key='key-a')
    second = _upload(client, token, 'Session two.', idempotency_key='key-b')
    assert first.json()['session_id'] != second.json()['session_id']

    sessions = client.get('/api/sessions/me', headers=auth_headers(token)).json()
    assert sessions['session_count'] == 2


def test_missing_idempotency_key_behaves_exactly_as_before(client):
    """No key supplied (e.g. an older client) must not be treated as a collision with anything."""
    _, token = signup(client, email='idem-c@example.com')

    first = _upload(client, token, 'Session one, no key.')
    second = _upload(client, token, 'Session two, no key.')
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()['session_id'] != second.json()['session_id']

    sessions = client.get('/api/sessions/me', headers=auth_headers(token)).json()
    assert sessions['session_count'] == 2


def test_same_key_is_scoped_per_user(client):
    """Two different users reusing the same client-generated key must not collide."""
    _, token_a = signup(client, email='idem-d1@example.com')
    _, token_b = signup(client, email='idem-d2@example.com')

    resp_a = _upload(client, token_a, 'User A session.', idempotency_key='shared-key')
    resp_b = _upload(client, token_b, 'User B session.', idempotency_key='shared-key')
    assert resp_a.status_code == 200
    assert resp_b.status_code == 200
    assert resp_a.json()['session_id'] != resp_b.json()['session_id']
