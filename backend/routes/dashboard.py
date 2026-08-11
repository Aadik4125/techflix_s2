"""
CogniVara — Dashboard Routes
GET /api/dashboard/me → Full dashboard data (CSI, trends, features) for the authenticated user
GET /api/baseline/me  → Current baseline status for the authenticated user
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session as DBSession

from config import BASELINE_SESSION_COUNT
from routes.auth import _user_payload
from database import get_db
from models.user import User
from models.session import Session
from models.baseline import Baseline
from routes.auth import get_current_user
from services.baseline import TRACKED_FEATURES, _merge_features
from services.csi import compute_csi
from services.drift import get_domain_flag_streaks
from services.interpretation import build_user_message

router = APIRouter()


@router.get('/dashboard/me')
def get_dashboard(user: User = Depends(get_current_user), db: DBSession = Depends(get_db)):

    user_id = user.id

    sessions = (
        db.query(Session)
        .filter(Session.user_id == user_id)
        .order_by(Session.session_number.asc())
        .all()
    )

    baseline = db.query(Baseline).filter(Baseline.user_id == user_id).first()

    # Latest session data
    latest = sessions[-1] if sessions else None
    latest_csi = latest.csi_score if latest else None
    latest_drift = latest.drift_scores if latest else None

    # Recompute the pure CSI output (cheap — no audio/DB work) purely to recover `confidence`,
    # which isn't persisted on the session row, so the plain-language message can tell a
    # low-confidence ("not enough data yet") result apart from a genuinely stable one.
    latest_flagged = (latest_drift or {}).get('flagged_features', [])
    if latest is not None:
        latest_csi_data = compute_csi(latest.z_scores or {}, latest_drift or {})
        domain_streaks = get_domain_flag_streaks(
            db, user_id, latest_flagged, exclude_session_id=latest.id
        )
        user_message = build_user_message(
            csi_score=latest_csi if latest_csi is not None else 50,
            confidence=latest_csi_data.get('confidence', 0.0),
            flagged_features=latest_flagged,
            domain_streaks=domain_streaks,
        )
    else:
        user_message = build_user_message(
            csi_score=50, confidence=0.0, flagged_features=[], domain_streaks={}
        )

    # Build longitudinal trend arrays
    trends = {key: [] for key in TRACKED_FEATURES}
    csi_trend = []
    session_labels = []

    for sess in sessions:
        session_labels.append(f'Session {sess.session_number}')
        csi_trend.append(sess.csi_score if sess.csi_score is not None else 50)

        merged = _merge_features(sess)

        for key in TRACKED_FEATURES:
            trends[key].append(merged.get(key, 0.0))

    # Key feature summary for the dashboard cards
    feature_summary = {}
    if latest:
        merged_latest = _merge_features(latest)

        feature_summary = {
            'mfcc_variance': merged_latest.get('mfcc_variance_avg', 0),
            'pitch_mean': merged_latest.get('pitch_mean', 0),
            'jitter': merged_latest.get('jitter_local', 0),
            'shimmer': merged_latest.get('shimmer_local', 0),
            'speech_rate': merged_latest.get('speech_rate', 0),
            'pause_variability': merged_latest.get('pause_variability', 0),
            'response_latency': merged_latest.get('response_latency', 0),
            'lexical_diversity': merged_latest.get('lexical_diversity', 0),
            'filler_ratio': merged_latest.get('filler_ratio', 0),
            'syntactic_complexity': merged_latest.get('syntactic_complexity', 0),
        }

    return {
        'user': _user_payload(user),
        'session_count': len(sessions),
        'baseline_ready': baseline is not None and baseline.feature_means is not None,
        'baseline_sessions': baseline.session_count if baseline else 0,
        'sessions_needed': BASELINE_SESSION_COUNT,

        # Latest scores
        'latest_csi': latest_csi,
        'latest_risk_level': (latest_drift or {}).get('per_feature', {}),
        'flagged_features': (latest_drift or {}).get('flagged_features', []),
        'user_message': user_message,

        # Feature summary (latest session)
        'feature_summary': feature_summary,

        # Longitudinal trends
        'trends': {
            'labels': session_labels,
            'csi': csi_trend,
            'features': trends,
        },
    }


@router.get('/baseline/me')
def get_baseline_status(user: User = Depends(get_current_user), db: DBSession = Depends(get_db)):
    """Check baseline status for the authenticated user."""
    user_id = user.id

    baseline = db.query(Baseline).filter(Baseline.user_id == user_id).first()
    session_count = db.query(Session).filter(Session.user_id == user_id).count()

    return {
        'user_id': user_id,
        'baseline_ready': baseline is not None and baseline.feature_means is not None,
        'sessions_completed': session_count,
        'sessions_needed': BASELINE_SESSION_COUNT,
        'feature_means': baseline.feature_means if baseline else None,
        'feature_stds': baseline.feature_stds if baseline else None,
    }
