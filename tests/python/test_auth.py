"""
Auth flow and ownership tests: password verification, token issuance, and the
guarantee that every {user}-scoped route only ever returns the caller's own data.
"""

from conftest import auth_headers, signup


def test_signup_returns_token_and_unverified_email(client):
    user_id, token = signup(client, email='alice@example.com')
    assert user_id > 0
    assert token

    resp = client.get('/api/auth/me', headers=auth_headers(token))
    assert resp.status_code == 200
    body = resp.json()
    assert body['email'] == 'alice@example.com'
    assert body['email_verified'] is False
    assert body['google_linked'] is False


def test_duplicate_signup_email_is_rejected(client):
    signup(client, email='dupe@example.com')
    resp = client.post(
        '/api/auth/signup',
        data={'name': 'Someone Else', 'email': 'dupe@example.com', 'password': 'anotherpass1', 'age': 40, 'gender': 'Other'},
    )
    assert resp.status_code == 409


def test_signup_rejects_weak_password(client):
    resp = client.post(
        '/api/auth/signup',
        data={'name': 'Weak', 'email': 'weak@example.com', 'password': 'short', 'age': 25, 'gender': 'Other'},
    )
    assert resp.status_code == 422


def test_signup_rejects_invalid_email(client):
    resp = client.post(
        '/api/auth/signup',
        data={'name': 'Bad Email', 'email': 'not-an-email', 'password': 'validpass1', 'age': 25, 'gender': 'Other'},
    )
    assert resp.status_code == 422


def test_login_with_correct_password_succeeds(client):
    signup(client, email='bob@example.com', password='correcthorse1')
    resp = client.post('/api/auth/login', data={'email': 'bob@example.com', 'password': 'correcthorse1'})
    assert resp.status_code == 200
    assert resp.json()['token']


def test_login_with_wrong_password_fails(client):
    signup(client, email='carol@example.com', password='correcthorse1')
    resp = client.post('/api/auth/login', data={'email': 'carol@example.com', 'password': 'wrongpassword'})
    assert resp.status_code == 401


def test_login_with_unknown_email_fails(client):
    resp = client.post('/api/auth/login', data={'email': 'nobody@example.com', 'password': 'whatever123'})
    assert resp.status_code == 401


def test_protected_route_without_token_is_rejected(client):
    resp = client.get('/api/dashboard/me')
    assert resp.status_code == 401


def test_protected_route_with_garbage_token_is_rejected(client):
    resp = client.get('/api/dashboard/me', headers=auth_headers('not-a-real-token'))
    assert resp.status_code == 401


def test_logout_invalidates_the_token(client):
    _, token = signup(client, email='dave@example.com')
    resp = client.post('/api/auth/logout', headers=auth_headers(token))
    assert resp.status_code == 200

    resp = client.get('/api/auth/me', headers=auth_headers(token))
    assert resp.status_code == 401


class TestOwnership:
    """The whole point of moving off client-supplied user_id: verify it actually holds."""

    def test_dashboard_only_ever_returns_the_callers_own_data(self, client):
        _, token_a = signup(client, email='owner-a@example.com', name='Owner A')
        _, token_b = signup(client, email='owner-b@example.com', name='Owner B')

        resp_a = client.get('/api/dashboard/me', headers=auth_headers(token_a))
        resp_b = client.get('/api/dashboard/me', headers=auth_headers(token_b))
        assert resp_a.json()['user']['email'] == 'owner-a@example.com'
        assert resp_b.json()['user']['email'] == 'owner-b@example.com'

    def test_upload_ties_the_session_to_the_token_owner_not_a_form_field(self, client):
        user_a_id, token_a = signup(client, email='upload-a@example.com')
        user_b_id, token_b = signup(client, email='upload-b@example.com')

        resp = client.post(
            '/api/upload',
            headers=auth_headers(token_a),
            files={'audio': ('test.wav', b'fake-audio-bytes', 'audio/wav')},
            data={'transcript': 'This session belongs to user A.'},
        )
        assert resp.status_code == 200
        assert resp.json()['user_id'] == user_a_id
        assert resp.json()['user_id'] != user_b_id

        # User B's history must not include user A's session.
        resp_b_sessions = client.get('/api/sessions/me', headers=auth_headers(token_b))
        assert resp_b_sessions.json()['session_count'] == 0

    def test_settings_update_only_ever_touches_the_callers_own_account(self, client):
        _, token_a = signup(client, email='settings-a@example.com', name='Original A')
        _, token_b = signup(client, email='settings-b@example.com', name='Original B')

        resp = client.patch('/api/auth/me', headers=auth_headers(token_a), data={'name': 'Renamed A'})
        assert resp.status_code == 200
        assert resp.json()['name'] == 'Renamed A'

        resp_b = client.get('/api/auth/me', headers=auth_headers(token_b))
        assert resp_b.json()['name'] == 'Original B'  # untouched by user A's update


def test_admin_dump_endpoints_no_longer_exist(client):
    assert client.get('/api/users').status_code == 404
    assert client.get('/api/sessions').status_code == 404


def test_legacy_upsert_by_email_endpoint_no_longer_exists(client):
    resp = client.post('/api/user', data={'name': 'x', 'email': 'legacy@example.com'})
    assert resp.status_code == 404
