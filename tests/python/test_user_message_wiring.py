"""
Confirms the plain-language user_message is actually wired into the real HTTP responses,
not just correct in isolation -- both /api/upload and /api/dashboard/me should carry it.
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


_VALID_TIERS = {'insufficient_data', 'stable', 'one_off', 'sustained'}


def test_upload_response_includes_user_message(client):
    _, token = signup(client, email='msg-a@example.com')
    resp = _upload(client, token, 'A quick note about today.', idempotency_key='k1')
    assert resp.status_code == 200
    msg = resp.json()['user_message']
    assert msg['tier'] in _VALID_TIERS
    assert isinstance(msg['headline'], str) and msg['headline']
    assert 'detail' in msg
    assert 'doctor_suggestion' in msg


def test_replayed_upload_also_includes_user_message(client):
    _, token = signup(client, email='msg-b@example.com')
    first = _upload(client, token, 'Same recording.', idempotency_key='k1')
    second = _upload(client, token, 'Same recording.', idempotency_key='k1')
    assert second.json()['replayed'] is True
    assert second.json()['user_message']['tier'] in _VALID_TIERS
    assert first.json()['session_id'] == second.json()['session_id']


def test_dashboard_response_includes_user_message(client):
    _, token = signup(client, email='msg-c@example.com')
    _upload(client, token, 'First session.', idempotency_key='k1')
    resp = client.get('/api/dashboard/me', headers=auth_headers(token))
    assert resp.status_code == 200
    assert resp.json()['user_message']['tier'] in _VALID_TIERS


def test_dashboard_with_no_sessions_gives_insufficient_data(client):
    _, token = signup(client, email='msg-d@example.com')
    resp = client.get('/api/dashboard/me', headers=auth_headers(token))
    assert resp.status_code == 200
    assert resp.json()['user_message']['tier'] == 'insufficient_data'
