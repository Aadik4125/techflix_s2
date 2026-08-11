"""
CogniVara - Composite Cognitive Stability Index (CSI)
Conservative multi-domain score built from per-user z-score deviations.
Higher CSI = more cognitive stability. Lower CSI = more concern.
"""

from __future__ import annotations

import math

from config import CSI_WEIGHTS


_Z_SCORE_NOISE_FLOOR = 0.35
_Z_SCORE_CAP = 3.5
_CSI_DECAY = 1.05
_DOMAIN_BLEND = {
    'acoustic': 0.40,
    'temporal': 0.35,
    'linguistic': 0.25,
}
FEATURE_DOMAINS = {
    'mfcc_variance_avg': 'acoustic',
    'pitch_mean': 'acoustic',
    'pitch_var': 'acoustic',
    'pitch_range': 'acoustic',
    'voiced_fraction': 'acoustic',
    'jitter_local': 'acoustic',
    'hnr_mean': 'acoustic',
    'shimmer_local': 'acoustic',
    'spectral_centroid_mean': 'acoustic',
    'spectral_centroid_var': 'acoustic',
    'energy_mean': 'acoustic',
    'energy_var': 'acoustic',
    'speech_rate': 'acoustic',
    'duration_sec': 'acoustic',
    'response_latency': 'temporal',
    'rhythm_consistency': 'temporal',
    'pause_variability': 'temporal',
    'speed_variability': 'temporal',
    'mean_pause_duration': 'temporal',
    'max_pause_duration': 'temporal',
    'pause_count': 'temporal',
    'speech_ratio': 'temporal',
    'speech_duration_sec': 'temporal',
    'speech_segment_count': 'temporal',
    'sentence_length_mean': 'linguistic',
    'lexical_diversity': 'linguistic',
    'avg_word_length': 'linguistic',
    'filler_ratio': 'linguistic',
    'content_word_ratio': 'linguistic',
    'syntactic_complexity': 'linguistic',
    'vocabulary_richness': 'linguistic',
}
_DOMAIN_FEATURES = {
    domain: tuple(key for key, mapped in FEATURE_DOMAINS.items() if mapped == domain)
    for domain in _DOMAIN_BLEND
}


def _normalize_z_deviation(value: float) -> float:
    """
    Normalize absolute z-score to 0..1 with a noise floor and hard cap.

    - |z| <= 0.5 is treated as normal session-to-session noise.
    - |z| >= 3.0 is treated as maximum concerning deviation.
    """
    abs_z = abs(float(value))
    if abs_z <= _Z_SCORE_NOISE_FLOOR:
        return 0.0
    scaled = (abs_z - _Z_SCORE_NOISE_FLOOR) / (_Z_SCORE_CAP - _Z_SCORE_NOISE_FLOOR)
    return max(0.0, min(1.0, scaled))


def _weighted_average(items: dict[str, float], keys: tuple[str, ...]) -> float:
    total_weight = 0.0
    total_value = 0.0
    for key in keys:
        if key not in items:
            continue
        weight = float(CSI_WEIGHTS.get(key, 0.0))
        if weight <= 0:
            continue
        total_weight += weight
        total_value += weight * float(items[key])
    if total_weight <= 0:
        return 0.0
    return total_value / total_weight


def _domain_coverage(z_scores: dict[str, float], domain: str) -> float:
    features = _DOMAIN_FEATURES.get(domain, ())
    if not features:
        return 0.0
    available = sum(1 for key in features if key in z_scores)
    return available / len(features)


def _build_interpretation(
    csi_score: int,
    confidence: float,
    dominant_domains: list[str],
    flagged_features: list[str],
) -> tuple[str, str]:
    domain_text = ', '.join(dominant_domains) if dominant_domains else 'limited signals'
    if confidence < 0.45:
        return (
            'unknown',
            'CSI computed with limited evidence. Collect more full-analysis sessions for a more reliable trend.',
        )

    if csi_score >= 75:
        return (
            'low',
            f'Cognitive speech patterns appear stable relative to your personal baseline, with strongest evidence from {domain_text}.',
        )

    if csi_score >= 45:
        flag_text = ', '.join(f.replace('_', ' ') for f in flagged_features[:3]) if flagged_features else domain_text
        return (
            'moderate',
            f'Moderate deviation from your baseline is present, especially in {flag_text}. Continued monitoring is recommended.',
        )

    return (
        'high',
        f'Sustained deviation from your baseline is present across {domain_text}. This score should be treated as a monitoring signal rather than a diagnosis.',
    )


def compute_csi(z_scores: dict, drift_data: dict) -> dict:
    """Compute the Composite Cognitive Stability Index."""
    if not z_scores:
        return {
            'csi_score': 50,
            'components': {},
            'interpretation': 'Insufficient data for CSI computation.',
            'risk_level': 'unknown',
            'confidence': 0.0,
            'feature_coverage': 0.0,
            'domain_coverage': {},
        }

    raw_components = {
        key: float(z_scores[key])
        for key in FEATURE_DOMAINS
        if key in z_scores
    }
    components = {key: _normalize_z_deviation(value) for key, value in raw_components.items()}

    domain_components: dict[str, float] = {}
    domain_coverage: dict[str, float] = {}
    for domain in _DOMAIN_BLEND:
        domain_components[domain] = round(_weighted_average(components, _DOMAIN_FEATURES[domain]), 4)
        domain_coverage[domain] = round(_domain_coverage(z_scores, domain), 4)

    available_weight = sum(CSI_WEIGHTS.get(key, 0.0) for key in raw_components)
    total_weight = sum(CSI_WEIGHTS.values()) or 1.0
    feature_coverage = max(0.0, min(1.0, available_weight / total_weight))
    domain_presence_count = sum(1 for value in domain_coverage.values() if value > 0.0)
    domain_completeness = sum(domain_coverage.values()) / max(len(domain_coverage), 1)
    confidence = max(
        0.0,
        min(
            1.0,
            (0.55 * feature_coverage)
            + (0.30 * (domain_presence_count / 3.0))
            + (0.15 * domain_completeness),
        ),
    )

    weighted_mean = _weighted_average(components, tuple(raw_components.keys()))
    max_component = max(components.values()) if components else 0.0
    domain_weighted = sum(
        _DOMAIN_BLEND[domain] * domain_components.get(domain, 0.0)
        for domain in _DOMAIN_BLEND
    )
    weighted_sum = (0.50 * weighted_mean) + (0.25 * max_component) + (0.25 * domain_weighted)

    raw_csi = 100.0 * math.exp(-_CSI_DECAY * weighted_sum)
    conservative_csi = (confidence * raw_csi) + ((1.0 - confidence) * 50.0)
    csi_score = int(max(0, min(100, round(conservative_csi))))

    flagged_features = drift_data.get('flagged_features', []) if drift_data else []
    overall_drift = float(drift_data.get('overall_drift_score', 0.0)) if drift_data else 0.0
    flag_penalty = min(15, len(flagged_features) * 2)
    drift_penalty = min(8, int(round(overall_drift * 2.0)))
    low_confidence_penalty = 6 if confidence < 0.45 else 0
    csi_score = max(0, csi_score - flag_penalty - drift_penalty - low_confidence_penalty)

    dominant_domains = [
        domain
        for domain, value in sorted(domain_components.items(), key=lambda item: item[1], reverse=True)
        if value > 0
    ][:2]
    risk_level, interpretation = _build_interpretation(
        csi_score=csi_score,
        confidence=confidence,
        dominant_domains=dominant_domains,
        flagged_features=flagged_features,
    )

    return {
        'csi_score': csi_score,
        'components': {k: round(v, 4) for k, v in components.items()},
        'raw_components_z': {k: round(abs(float(v)), 4) for k, v in raw_components.items()},
        'weighted_drift': round(weighted_sum, 4),
        'domain_components': domain_components,
        'domain_coverage': domain_coverage,
        'feature_coverage': round(feature_coverage, 4),
        'confidence': round(confidence, 4),
        'flag_penalty': flag_penalty,
        'drift_penalty': drift_penalty,
        'formula_version': 'csi_v5_multidomain_conservative',
        'interpretation': interpretation,
        'risk_level': risk_level,
    }
