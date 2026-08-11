"""
Domain flag-streak detection: must count CONSECUTIVE flagged check-ins per domain, stopping
at the first gap, and must not double-count the current session when it's already been
committed to the DB with drift_scores set (the dashboard's "latest session" case).
"""

from conftest import signup
from models.session import Session
from services.drift import get_domain_flag_streaks


def _make_session(db_session, user_id, session_number, flagged_features):
    sess = Session(
        user_id=user_id,
        session_number=session_number,
        drift_scores={'flagged_features': flagged_features},
        csi_score=50,
    )
    db_session.add(sess)
    db_session.commit()
    db_session.refresh(sess)
    return sess


def test_streak_stops_at_the_first_gap(client, db_session):
    user_id, _ = signup(client, email='streak-a@example.com')

    _make_session(db_session, user_id, 1, ['jitter_local'])       # acoustic
    _make_session(db_session, user_id, 2, ['jitter_local'])       # acoustic
    _make_session(db_session, user_id, 3, ['lexical_diversity'])  # linguistic -- breaks the streak

    streaks = get_domain_flag_streaks(db_session, user_id, ['jitter_local'])
    assert streaks['acoustic'] == 1  # only the current session counts; session 3 broke continuity


def test_streak_extends_across_consecutive_sessions(client, db_session):
    user_id, _ = signup(client, email='streak-b@example.com')

    _make_session(db_session, user_id, 1, ['jitter_local'])
    _make_session(db_session, user_id, 2, ['hnr_mean'])
    _make_session(db_session, user_id, 3, ['shimmer_local'])

    streaks = get_domain_flag_streaks(db_session, user_id, ['pitch_var'])  # also acoustic
    assert streaks['acoustic'] == 4  # current + 3 prior, all in the acoustic domain


def test_unrelated_domain_streak_is_unaffected(client, db_session):
    user_id, _ = signup(client, email='streak-c@example.com')

    _make_session(db_session, user_id, 1, ['jitter_local'])
    streaks = get_domain_flag_streaks(db_session, user_id, ['jitter_local'])
    assert streaks['temporal'] == 0
    assert streaks['linguistic'] == 0


def test_exclude_session_id_prevents_double_counting_the_latest_session(client, db_session):
    """
    The dashboard passes the latest session's own flagged_features as `current` while that
    session is ALSO already sitting in the DB history -- without exclude_session_id it would
    be counted twice.
    """
    user_id, _ = signup(client, email='streak-d@example.com')

    _make_session(db_session, user_id, 1, ['jitter_local'])
    _make_session(db_session, user_id, 2, ['jitter_local'])
    latest = _make_session(db_session, user_id, 3, ['jitter_local'])

    without_exclude = get_domain_flag_streaks(db_session, user_id, ['jitter_local'])
    with_exclude = get_domain_flag_streaks(
        db_session, user_id, ['jitter_local'], exclude_session_id=latest.id
    )

    assert without_exclude['acoustic'] == 4  # session 3 double-counted (current + itself in history)
    assert with_exclude['acoustic'] == 3  # correct: current (=session 3) + sessions 2, 1
