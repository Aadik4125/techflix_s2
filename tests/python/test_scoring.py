"""
Tests for the core scoring pipeline: does the CSI/drift math actually mean anything?

These exercise the pure functions in services/baseline.py, services/drift.py, and
services/csi.py directly — no DB, no HTTP — so they're fast and precise about what
they're checking.
"""

import random

from services.baseline import TRACKED_FEATURES, compute_baseline_stats, compute_z_scores
from services.csi import compute_csi
from services.drift import compute_drift_stats


def _feature_dict(overrides=None):
    """A full TRACKED_FEATURES dict with sane defaults, overridable per-test."""
    base = {key: 1.0 for key in TRACKED_FEATURES}
    if overrides:
        base.update(overrides)
    return base


class TestBaselineStats:
    def test_mean_and_std_are_correct(self):
        sessions = [
            _feature_dict({'pitch_mean': 100.0}),
            _feature_dict({'pitch_mean': 110.0}),
            _feature_dict({'pitch_mean': 120.0}),
        ]
        means, stds = compute_baseline_stats(sessions)
        assert means['pitch_mean'] == 110.0
        assert stds['pitch_mean'] > 0

    def test_missing_feature_defaults_to_zero(self):
        sessions = [{'pitch_mean': 100.0}, {'pitch_mean': 100.0}]
        means, _ = compute_baseline_stats(sessions)
        # Any TRACKED_FEATURES key absent from the input dicts should default to 0, not error.
        assert means['jitter_local'] == 0.0


class TestZScores:
    def test_value_at_the_mean_is_near_zero(self):
        means = {key: 50.0 for key in TRACKED_FEATURES}
        stds = {key: 5.0 for key in TRACKED_FEATURES}
        z = compute_z_scores(means, stds, _feature_dict({'pitch_mean': 50.0}))
        assert abs(z['pitch_mean']) < 0.1

    def test_value_far_from_mean_has_larger_magnitude_z(self):
        means = {key: 50.0 for key in TRACKED_FEATURES}
        stds = {key: 5.0 for key in TRACKED_FEATURES}
        z_near = compute_z_scores(means, stds, _feature_dict({'pitch_mean': 52.0}))
        z_far = compute_z_scores(means, stds, _feature_dict({'pitch_mean': 80.0}))
        assert abs(z_far['pitch_mean']) > abs(z_near['pitch_mean'])

    def test_z_score_is_clipped(self):
        means = {key: 50.0 for key in TRACKED_FEATURES}
        stds = {key: 1.0 for key in TRACKED_FEATURES}
        z = compute_z_scores(means, stds, _feature_dict({'pitch_mean': 100000.0}))
        assert abs(z['pitch_mean']) <= 3.5  # _Z_CLIP

    def test_missing_stats_returns_empty(self):
        assert compute_z_scores(None, None, _feature_dict()) == {}


class TestDriftStats:
    def test_insufficient_history_returns_no_flags(self):
        result = compute_drift_stats([], {'pitch_var': 0.0})
        assert result['flagged_features'] == []
        assert result['sessions_analyzed'] < 2

    def test_stable_noisy_feature_is_not_flagged(self):
        """Session-to-session noise around a flat baseline should NOT be treated as drift."""
        random.seed(42)
        history = [{'energy_mean': random.uniform(-0.2, 0.2)} for _ in range(4)]
        current = {'energy_mean': random.uniform(-0.2, 0.2)}
        result = compute_drift_stats(history, current)
        assert 'energy_mean' not in result['flagged_features']

    def test_consistent_downward_trend_is_flagged(self):
        """A genuine, sustained decline should be flagged — this is the entire point of drift detection."""
        history = [{'pitch_var': 0.0}, {'pitch_var': -0.3}, {'pitch_var': -0.6}]
        current = {'pitch_var': -1.2}
        result = compute_drift_stats(history, current)
        assert 'pitch_var' in result['flagged_features']
        assert result['per_feature']['pitch_var']['slope'] < 0

    def test_upward_trend_is_not_flagged(self):
        """Drift detection only flags declining trends (slope < 0) by design — confirm that holds."""
        history = [{'pitch_var': 0.0}, {'pitch_var': 1.0}, {'pitch_var': 2.0}]
        current = {'pitch_var': 3.0}
        result = compute_drift_stats(history, current)
        assert 'pitch_var' not in result['flagged_features']


class TestCSI:
    def test_no_z_scores_returns_neutral_unknown(self):
        result = compute_csi({}, {})
        assert result['csi_score'] == 50
        assert result['risk_level'] == 'unknown'

    def test_stable_z_scores_score_high(self):
        z_scores = {key: 0.0 for key in TRACKED_FEATURES}
        result = compute_csi(z_scores, {'flagged_features': [], 'overall_drift_score': 0.0})
        assert result['csi_score'] >= 75
        assert result['risk_level'] == 'low'

    def test_large_deviations_score_lower_than_stable(self):
        stable = {key: 0.0 for key in TRACKED_FEATURES}
        deviated = {key: 3.0 for key in TRACKED_FEATURES}
        stable_score = compute_csi(stable, {'flagged_features': [], 'overall_drift_score': 0.0})['csi_score']
        deviated_score = compute_csi(deviated, {'flagged_features': [], 'overall_drift_score': 3.0})['csi_score']
        assert deviated_score < stable_score

    def test_flagged_features_reduce_score(self):
        z_scores = {key: 0.5 for key in TRACKED_FEATURES}
        no_flags = compute_csi(z_scores, {'flagged_features': [], 'overall_drift_score': 0.0})
        with_flags = compute_csi(z_scores, {'flagged_features': ['pitch_var', 'jitter_local'], 'overall_drift_score': 0.0})
        assert with_flags['csi_score'] <= no_flags['csi_score']


class TestSyntheticDriftBacktest:
    """
    The core validation question: does this pipeline actually distinguish a real,
    sustained decline in a biomarker from ordinary session-to-session noise?

    Simulates ~2 weeks of check-ins for one user: a stable 3-session baseline,
    then one feature (pitch_var) drifting steadily downward while a control
    feature (energy_mean) stays flat with only noise. Confirms the drifting
    feature gets flagged and scores decline, while the noisy-but-stable
    control does not get flagged and the CSI does not falsely alarm on it alone.
    """

    def test_engineered_decline_is_detected_and_noise_is_not(self):
        random.seed(7)

        # 3-session baseline: pitch_var stable around 100, energy_mean stable around 50 — both noisy.
        baseline_sessions = [
            _feature_dict({
                'pitch_var': 100.0 + random.uniform(-3, 3),
                'energy_mean': 50.0 + random.uniform(-2, 2),
            })
            for _ in range(3)
        ]
        means, stds = compute_baseline_stats(baseline_sessions)

        # 6 subsequent check-ins: pitch_var declines steadily and meaningfully;
        # energy_mean keeps wobbling around the same baseline value (noise only).
        z_history = []
        flagged_at_any_point = False
        csi_scores = []
        for i in range(1, 7):
            session_features = _feature_dict({
                'pitch_var': 100.0 - (i * 15) + random.uniform(-2, 2),
                'energy_mean': 50.0 + random.uniform(-2, 2),
            })
            current_z = compute_z_scores(means, stds, session_features)
            drift = compute_drift_stats(z_history, current_z)
            csi = compute_csi(current_z, drift)
            csi_scores.append(csi['csi_score'])

            if 'pitch_var' in drift['flagged_features']:
                flagged_at_any_point = True
            # The control feature should never be flagged purely from noise around a flat baseline.
            assert 'energy_mean' not in drift['flagged_features'], (
                f'session {i}: noisy-but-stable feature was incorrectly flagged as drift'
            )

            z_history.append(current_z)

        assert flagged_at_any_point, 'a sustained, engineered decline was never flagged by drift detection'
        # CSI should trend down as the engineered decline progresses, not stay flat or rise.
        assert csi_scores[-1] < csi_scores[0], (
            f'CSI did not decline under a sustained engineered drift: {csi_scores}'
        )
