"""
CogniVara - Personal Baseline Modeling
Computes per-user baseline statistics and Z-scores.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sqlalchemy.orm import Session as DBSession

from config import BASELINE_SESSION_COUNT
from models.baseline import Baseline
from models.session import Session


TRACKED_FEATURES = [
    'mfcc_variance_avg',
    'pitch_mean',
    'pitch_var',
    'pitch_range',
    'voiced_fraction',
    'jitter_local',
    'hnr_mean',
    'shimmer_local',
    'spectral_centroid_mean',
    'spectral_centroid_var',
    'energy_mean',
    'energy_var',
    'speech_rate',
    'duration_sec',
    'response_latency',
    'rhythm_consistency',
    'pause_variability',
    'speed_variability',
    'mean_pause_duration',
    'max_pause_duration',
    'pause_count',
    'speech_ratio',
    'speech_duration_sec',
    'speech_segment_count',
    'sentence_length_mean',
    'lexical_diversity',
    'avg_word_length',
    'filler_ratio',
    'content_word_ratio',
    'syntactic_complexity',
    'vocabulary_richness',
]

_ABS_STD_FLOOR = 0.10
_REL_STD_FLOOR = 0.06
_Z_CLIP = 3.5
_Z_TANH_SCALE = 1.5
_Z_SHRINK = 0.90


def _merge_features(session_row: Session) -> dict[str, Any]:
    """Merge all feature dicts from a session into a flat dict."""
    merged: dict[str, Any] = {}
    for feat_dict in [
        session_row.acoustic_features,
        session_row.temporal_features,
        session_row.linguistic_features,
    ]:
        if isinstance(feat_dict, dict):
            merged.update(feat_dict)
    return merged


def compute_baseline_stats(
    feature_dicts: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute per-feature mean/std from a list of merged feature dicts. Pure — no DB access."""
    feature_matrix: dict[str, list[float]] = {key: [] for key in TRACKED_FEATURES}
    for merged in feature_dicts:
        for key in TRACKED_FEATURES:
            val = merged.get(key, 0.0)
            feature_matrix[key].append(float(val) if val is not None else 0.0)

    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for key in TRACKED_FEATURES:
        arr = np.array(feature_matrix[key], dtype=float)
        means[key] = round(float(np.mean(arr)), 6)
        stds[key] = round(float(np.std(arr)), 6)

    return means, stds


def compute_baseline(db: DBSession, user_id: int) -> Baseline | None:
    """Compute baseline from the first N sessions and persist it."""
    sessions = (
        db.query(Session)
        .filter(Session.user_id == user_id)
        .order_by(Session.session_number.asc())
        .limit(BASELINE_SESSION_COUNT)
        .all()
    )

    if len(sessions) < BASELINE_SESSION_COUNT:
        return None

    means, stds = compute_baseline_stats([_merge_features(sess) for sess in sessions])

    baseline = db.query(Baseline).filter(Baseline.user_id == user_id).first()
    if baseline is not None:
        baseline.feature_means = means
        baseline.feature_stds = stds
        baseline.session_count = len(sessions)
    else:
        baseline = Baseline(
            user_id=user_id,
            feature_means=means,
            feature_stds=stds,
            session_count=len(sessions),
        )
        db.add(baseline)

    db.commit()
    db.refresh(baseline)
    return baseline


def compute_z_scores(
    feature_means: dict[str, Any],
    feature_stds: dict[str, Any],
    current_features: dict[str, Any],
) -> dict[str, float]:
    """Compute Z-score for each tracked feature. Pure — no DB access."""
    if not isinstance(feature_means, dict) or not isinstance(feature_stds, dict):
        return {}

    z_scores: dict[str, float] = {}
    for key in TRACKED_FEATURES:
        current_val = current_features.get(key, 0.0)
        mean = float(feature_means.get(key, 0.0))
        std = float(feature_stds.get(key, 0.0))
        std_floor = max(_ABS_STD_FLOOR, abs(mean) * _REL_STD_FLOOR)
        effective_std = max(std, std_floor)

        raw_z = (float(current_val) - mean) / effective_std
        z = math.tanh(raw_z / _Z_TANH_SCALE) * _Z_TANH_SCALE
        z *= _Z_SHRINK
        z = max(-_Z_CLIP, min(_Z_CLIP, z))

        z_scores[key] = round(float(z), 4)

    return z_scores
