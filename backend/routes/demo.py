"""
CogniVara - Demo routes.
Stateless mirror of the real /api/upload mechanism: no auth, no DB writes,
no audio persisted to disk. Exists purely to show the scoring mechanism
in a single sitting, with nothing carried forward afterward.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from services.baseline import compute_baseline_stats, compute_z_scores
from services.csi import compute_csi
from services.drift import compute_drift_stats
from services.feature_pipeline import extract_session_features

router = APIRouter()

# Minimal in-memory per-IP throttle. This is the one unauthenticated,
# compute-heavy surface in the app, so it gets a cheap stdlib-only guard.
_RATE_LIMIT_WINDOW_SEC = 60
_RATE_LIMIT_MAX_REQUESTS = 20
_request_log: dict[str, list[float]] = {}


def _check_rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else 'unknown'
    now = time.monotonic()
    recent = [t for t in _request_log.get(ip, []) if now - t < _RATE_LIMIT_WINDOW_SEC]
    if len(recent) >= _RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(status_code=429, detail='Too many demo requests, please slow down')
    recent.append(now)
    _request_log[ip] = recent


@router.post('/extract')
async def demo_extract(
    request: Request,
    audio: UploadFile = File(...),
    transcript: str = Form(''),
):
    """One demo recording's audio in, its raw feature dict out. No DB write, no file saved."""
    _check_rate_limit(request)

    audio_bytes = await audio.read()
    features = extract_session_features(audio_bytes, transcript)

    return {
        'preprocessing': features['preprocessing'],
        'acoustic_features': features['acoustic'],
        'temporal_features': features['temporal'],
        'linguistic_features': features['linguistic'],
    }


class DemoFinalizeRequest(BaseModel):
    sessions: list[dict[str, float]]


@router.post('/finalize')
def demo_finalize(req: DemoFinalizeRequest, request: Request):
    """
    Takes the accumulated feature dicts from /extract and computes baseline + z-scores
    + drift + CSI for the final session. The baseline is built only from the sessions
    BEFORE the final one — the final session is never part of its own baseline, since
    scoring a session against a baseline that includes its own features silently pulls
    its z-scores toward "normal" (the same self-comparison bug fixed on the real
    /api/upload path).
    """
    _check_rate_limit(request)

    if len(req.sessions) < 2:
        raise HTTPException(status_code=400, detail='At least 2 sessions are required to finalize a demo')

    baseline_sessions = req.sessions[:-1]
    means, stds = compute_baseline_stats(baseline_sessions)

    z_score_history = [compute_z_scores(means, stds, sess) for sess in baseline_sessions]
    final_z_scores = compute_z_scores(means, stds, req.sessions[-1])
    drift_data: dict[str, Any] = compute_drift_stats(z_score_history, final_z_scores)
    csi_data = compute_csi(final_z_scores, drift_data)

    return {
        'baseline_ready': True,
        'z_scores': final_z_scores,
        'drift': drift_data,
        'csi': csi_data,
    }
