"""
CogniVara - Cognitive Drift Detection
Rolling window averages, trend slopes, and deviation flagging.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sqlalchemy.orm import Session as DBSession

from config import DRIFT_ROLLING_WINDOW, DRIFT_Z_THRESHOLD
from models.session import Session as SessionModel
from services.baseline import TRACKED_FEATURES
from services.csi import FEATURE_DOMAINS


def _get_recent_z_scores(
    db: DBSession, user_id: int, window: int = DRIFT_ROLLING_WINDOW
) -> list[dict[str, float]]:
    """Fetch the Z-scores from the last N sessions."""
    sessions = (
        db.query(SessionModel)
        .filter(SessionModel.user_id == user_id, SessionModel.z_scores.isnot(None))
        .order_by(SessionModel.session_number.desc())
        .limit(window)
        .all()
    )

    result: list[dict[str, float]] = []
    for sess in reversed(sessions):
        if isinstance(sess.z_scores, dict):
            result.append({k: float(v) for k, v in sess.z_scores.items()})
    return result


def compute_drift(
    db: DBSession, user_id: int, current_z_scores: dict[str, float]
) -> dict[str, Any]:
    """Thin DB-backed wrapper: fetch recent Z-score history, then compute drift."""
    recent = _get_recent_z_scores(db, user_id, DRIFT_ROLLING_WINDOW)
    return compute_drift_stats(recent, current_z_scores)


def compute_drift_stats(
    z_score_history: list[dict[str, float]], current_z_scores: dict[str, float]
) -> dict[str, Any]:
    """
    Compute drift detection metrics from an in-memory Z-score history. Pure — no DB access.
    - Rolling average of Z-scores (last N sessions)
    - Trend slope (linear regression on last N Z-scores)
    - Deviation flags (|Z| > threshold AND negative slope)
    """
    all_z = z_score_history + [current_z_scores] if current_z_scores else z_score_history

    if len(all_z) < 2:
        return {
            'per_feature': {},
            'flagged_features': [],
            'overall_drift_score': 0.0,
            'sessions_analyzed': len(all_z),
        }

    per_feature: dict[str, dict[str, Any]] = {}
    flagged: list[str] = []

    for key in TRACKED_FEATURES:
        values = [float(z.get(key, 0.0)) for z in all_z]
        arr = np.array(values, dtype=float)

        rolling_avg = float(np.mean(arr))

        if len(arr) >= 2:
            x = np.arange(len(arr), dtype=float)
            coeffs = np.polyfit(x, arr, 1)
            slope = float(coeffs[0])
        else:
            slope = 0.0

        current_z = float(current_z_scores.get(key, 0.0)) if current_z_scores else 0.0

        is_flagged = abs(current_z) > DRIFT_Z_THRESHOLD and slope < 0

        per_feature[key] = {
            'rolling_avg': round(rolling_avg, 4),
            'slope': round(slope, 4),
            'current_z': round(current_z, 4),
            'flagged': is_flagged,
        }

        if is_flagged:
            flagged.append(key)

    all_current_z = [abs(float(current_z_scores.get(k, 0.0))) for k in TRACKED_FEATURES]
    overall_drift = float(np.mean(all_current_z)) if all_current_z else 0.0

    return {
        'per_feature': per_feature,
        'flagged_features': flagged,
        'overall_drift_score': round(overall_drift, 4),
        'sessions_analyzed': len(all_z),
    }


def _domains_from_features(features: list[str]) -> set[str]:
    return {FEATURE_DOMAINS[f] for f in features if f in FEATURE_DOMAINS}


def get_domain_flag_streaks(
    db: DBSession,
    user_id: int,
    current_flagged_features: list[str],
    lookback: int = 6,
    exclude_session_id: int | None = None,
) -> dict[str, int]:
    """
    For each domain, count how many CONSECUTIVE most-recent scored sessions (starting from
    the current one and walking backward) had at least one flagged feature in that domain.

    Used to tell a single off day apart from a sustained pattern: one flagged session isn't
    meaningful on its own (see the earlier nervousness-vs-drift discussion — a single session's
    deviation is expected noise), but the same domain flagging several check-ins in a row is
    the "constantly going down" pattern worth surfacing to the user.

    `current_flagged_features` is treated as the most recent point in the streak. When the
    session it came from is already committed to the DB with drift_scores set (e.g. reporting
    on the latest stored session from the dashboard, rather than a session being scored right
    now), pass its id as `exclude_session_id` so the history query below doesn't count it twice.
    """
    query = db.query(SessionModel).filter(
        SessionModel.user_id == user_id, SessionModel.drift_scores.isnot(None)
    )
    if exclude_session_id is not None:
        query = query.filter(SessionModel.id != exclude_session_id)
    sessions = query.order_by(SessionModel.session_number.desc()).limit(lookback).all()

    # Most recent first: the current (not-yet-stored) session, then prior scored sessions.
    history_domains: list[set[str]] = [_domains_from_features(current_flagged_features)]
    for sess in sessions:
        flagged = (sess.drift_scores or {}).get('flagged_features', [])
        history_domains.append(_domains_from_features(flagged))

    streaks: dict[str, int] = {}
    for domain in set(FEATURE_DOMAINS.values()):
        count = 0
        for domains_flagged in history_domains:
            if domain in domains_flagged:
                count += 1
            else:
                break
        streaks[domain] = count
    return streaks
