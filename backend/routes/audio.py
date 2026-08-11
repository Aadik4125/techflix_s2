
from __future__ import annotations

import os
import uuid
from typing import Any

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as DBSession

from config import BASELINE_SESSION_COUNT, FAST_ANALYSIS_MODE
from database import get_db
from models.session import Session
from models.user import User
from routes.auth import get_current_user
from services.audio_storage import save_audio
from services.baseline import compute_baseline, compute_z_scores
from services.csi import compute_csi
from services.drift import compute_drift, get_domain_flag_streaks
from services.feature_pipeline import extract_session_features
from services.interpretation import build_user_message

router = APIRouter()


def _replay_response(session_row: Session, user: User) -> dict[str, Any]:
    """
    Build the same response shape a fresh /upload call would return, from a session that
    was already scored — used when a request replays an idempotency key that's already been
    processed, so a retried upload gets back the original result instead of being reprocessed
    (and instead of silently creating a second session row for the same recording).
    """
    csi_data: dict[str, Any] = {'csi_score': session_row.csi_score}
    drift_data = session_row.drift_scores
    flagged_features = (drift_data or {}).get('flagged_features', [])
    user_message = build_user_message(
        csi_score=csi_data['csi_score'],
        confidence=csi_data.get('confidence', 0.0),
        flagged_features=flagged_features,
        domain_streaks={},
    )
    return {
        'session_id': session_row.id,
        'session_number': session_row.session_number,
        'user_id': user.id,
        'user_latest_csi_score': user.latest_csi_score,
        'user_total_sessions': user.total_sessions,
        'acoustic_features': session_row.acoustic_features,
        'temporal_features': session_row.temporal_features,
        'linguistic_features': session_row.linguistic_features,
        'baseline_ready': session_row.z_scores is not None,
        'z_scores': session_row.z_scores,
        'drift': drift_data,
        'csi': csi_data,
        'user_message': user_message,
        'analysis_mode': 'fast' if FAST_ANALYSIS_MODE else 'full',
        'replayed': True,
    }


@router.post('/upload')
async def upload_and_analyze(
    audio: UploadFile = File(...),
    transcript: str = Form(''),
    idempotency_key: str | None = Form(None),
    user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    """
    Full cognitive analysis pipeline:
    1. Save audio
    2. Load & preprocess (noise filter, VAD, segmentation)
    3. Extract acoustic features (MFCCs, pitch, jitter, shimmer, etc.)
    4. Extract temporal features (latency, rhythm, pauses)
    5. Extract linguistic features from transcript (NLTK)
    6. Compute baseline (if enough sessions)
    7. Compute Z-scores
    8. Compute drift detection
    9. Compute CSI
    10. Store everything in DB
    """
    user_id = user.id
    idempotency_key = (idempotency_key or '').strip() or None

    if idempotency_key:
        existing = (
            db.query(Session)
            .filter(Session.user_id == user_id, Session.idempotency_key == idempotency_key)
            .first()
        )
        if existing is not None:
            return _replay_response(existing, user)

    last_session = (
        db.query(Session)
        .filter(Session.user_id == user_id)
        .order_by(Session.session_number.desc())
        .first()
    )
    session_number = (last_session.session_number + 1) if last_session else 1

    file_ext = os.path.splitext(audio.filename or 'recording.wav')[1] or '.wav'
    filename = f'user_{user_id}_session_{session_number}_{uuid.uuid4().hex[:8]}{file_ext}'

    audio_bytes = await audio.read()
    storage_path = save_audio(audio_bytes, filename)

    features = extract_session_features(audio_bytes, transcript)
    preprocess_result = features['preprocessing']
    acoustic = features['acoustic']
    temporal = features['temporal']
    linguistic = features['linguistic']

    session_row = Session(
        user_id=user_id,
        session_number=session_number,
        raw_audio_path=storage_path,
        transcript=transcript,
        acoustic_features=acoustic,
        temporal_features=temporal,
        linguistic_features=linguistic,
        idempotency_key=idempotency_key,
    )
    db.add(session_row)
    try:
        db.commit()
    except IntegrityError:
        # Lost a race against another request carrying the same idempotency key (e.g. two
        # near-simultaneous retries of the same upload) — the other request's row already won,
        # so replay it instead of failing or creating a duplicate.
        db.rollback()
        if idempotency_key:
            existing = (
                db.query(Session)
                .filter(Session.user_id == user_id, Session.idempotency_key == idempotency_key)
                .first()
            )
            if existing is not None:
                return _replay_response(existing, user)
        raise
    db.refresh(session_row)

    baseline = compute_baseline(db, user_id)

    z_scores: dict[str, float] | None = None
    drift_data: dict[str, Any] | None = None
    domain_streaks: dict[str, int] = {}
    csi_data: dict[str, Any] = {
        'csi_score': 50,
        'risk_level': 'unknown',
        'interpretation': 'Baseline not yet established. Complete more sessions.',
    }

    # The baseline is built from the first BASELINE_SESSION_COUNT sessions (compute_baseline
    # includes whichever session just crossed that count). A session must never be scored
    # against a baseline that was partly built from its own features — that self-comparison
    # silently pulls its own z-scores toward "normal" — so only sessions AFTER the baseline
    # window are ever compared to it. The baseline-completing session just establishes the
    # baseline and stays at the neutral placeholder, same as the sessions before it.
    if baseline is not None and session_number > BASELINE_SESSION_COUNT:
        all_features: dict[str, Any] = {}
        all_features.update(acoustic)
        all_features.update(temporal)
        all_features.update(linguistic)

        z_scores = compute_z_scores(baseline.feature_means, baseline.feature_stds, all_features)
        drift_data = compute_drift(db, user_id, z_scores)
        csi_data = compute_csi(z_scores, drift_data)
        domain_streaks = get_domain_flag_streaks(db, user_id, drift_data.get('flagged_features', []))

        # Smooth CSI against the previous scored session to reduce abrupt jumps. Only applies
        # once a real, previously-computed CSI exists to smooth against — the baseline-completing
        # session's csi_score is the neutral placeholder (50), not a real score, so smoothing the
        # first post-baseline session against it would clamp the true CSI into a narrow band
        # around 50 regardless of the actual computed value.
        if (
            last_session
            and last_session.csi_score is not None
            and last_session.session_number > BASELINE_SESSION_COUNT
        ):
            raw_csi = int(csi_data['csi_score'])
            prev_csi = int(last_session.csi_score)
            smoothed_csi = int(round((0.50 * prev_csi) + (0.50 * raw_csi)))
            max_step = 8
            smoothed_csi = max(prev_csi - max_step, min(prev_csi + max_step, smoothed_csi))
            csi_data['raw_csi_score'] = raw_csi
            csi_data['csi_score'] = smoothed_csi
    elif baseline is not None:
        csi_data = {
            'csi_score': 50,
            'risk_level': 'unknown',
            'interpretation': 'Baseline established from this session. Your first drift-tracked score will appear on your next check-in.',
        }

    session_row.z_scores = z_scores
    session_row.drift_scores = drift_data
    session_row.csi_score = int(csi_data['csi_score'])
    user.latest_csi_score = session_row.csi_score
    user.total_sessions = session_number
    user.last_session_at = session_row.created_at
    db.commit()

    user_message = build_user_message(
        csi_score=csi_data['csi_score'],
        confidence=csi_data.get('confidence', 0.0),
        flagged_features=(drift_data or {}).get('flagged_features', []),
        domain_streaks=domain_streaks,
    )

    return {
        'session_id': session_row.id,
        'session_number': session_number,
        'user_id': user_id,
        'user_latest_csi_score': user.latest_csi_score,
        'user_total_sessions': user.total_sessions,
        'preprocessing': {
            'duration_sec': preprocess_result['duration_sec'],
            'speech_duration_sec': preprocess_result['speech_duration_sec'],
            'speech_ratio': preprocess_result['speech_ratio'],
            'num_segments': preprocess_result['num_segments'],
        },
        'acoustic_features': acoustic,
        'temporal_features': temporal,
        'linguistic_features': linguistic,
        'baseline_ready': baseline is not None,
        'z_scores': z_scores,
        'drift': drift_data,
        'csi': csi_data,
        'user_message': user_message,
        'analysis_mode': 'fast' if FAST_ANALYSIS_MODE else 'full',
    }
