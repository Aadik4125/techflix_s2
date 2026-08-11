"""
CogniVara — Acoustic Feature Extraction
MFCCs, pitch, jitter, shimmer, HNR, spectral centroid, energy, speech rate.
"""

import numpy as np
import librosa

from services.utils import safe_float as _safe_float


# ── MFCCs ─────────────────────────────────────────────────

def extract_mfccs(y: np.ndarray, sr: int, n_mfcc: int = 13) -> dict:
    """Extract MFCC coefficients — mean + variance for each."""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    variances = np.var(mfccs, axis=1)
    result = {}
    for i in range(n_mfcc):
        result[f'mfcc_{i}_mean'] = _safe_float(np.mean(mfccs[i]))
        result[f'mfcc_{i}_var'] = _safe_float(variances[i])
    result['mfcc_variance_avg'] = _safe_float(np.mean(variances))
    return result


# ── Pitch + Jitter (single pYIN call) ────────────────────

def _extract_f0(y: np.ndarray, sr: int,
                fmin: float = 50.0, fmax: float = 500.0):
    """Run faster YIN pitch extraction, return voiced F0 array and voiced fraction."""
    f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=1024, hop_length=256)
    voiced_flag = np.isfinite(f0) & (f0 > 0)
    f0_voiced = f0[voiced_flag]
    voiced_frac = float(np.mean(voiced_flag)) if len(f0) > 0 else 0.0
    return f0_voiced, voiced_frac


def extract_pitch_and_jitter(y: np.ndarray, sr: int) -> dict:
    """Extract pitch stats + jitter from a single pYIN pass."""
    f0_voiced, voiced_frac = _extract_f0(y, sr)

    if len(f0_voiced) == 0:
        return {
            'pitch_mean': 0.0, 'pitch_var': 0.0, 'pitch_range': 0.0,
            'voiced_fraction': 0.0, 'jitter_local': 0.0, 'jitter_abs': 0.0,
        }

    result = {
        'pitch_mean': _safe_float(np.mean(f0_voiced)),
        'pitch_var': _safe_float(np.var(f0_voiced)),
        'pitch_range': _safe_float(np.ptp(f0_voiced)),
        'voiced_fraction': _safe_float(voiced_frac),
    }

    # Jitter (F0 period perturbation)
    if len(f0_voiced) >= 2:
        periods = 1.0 / (f0_voiced + 1e-10)
        diffs = np.abs(np.diff(periods))
        mean_period = np.mean(periods)
        result['jitter_local'] = _safe_float(np.mean(diffs) / (mean_period + 1e-10))
        result['jitter_abs'] = _safe_float(np.mean(diffs))
    else:
        result['jitter_local'] = 0.0
        result['jitter_abs'] = 0.0

    return result


# ── Harmonics-to-Noise Ratio ──────────────────────────────

def extract_hnr(y: np.ndarray, sr: int,
                 fmin: float = 50.0, fmax: float = 500.0,
                 frame_length: int = 2048, hop_length: int = 512) -> dict:
    """
    Harmonics-to-Noise Ratio via normalized autocorrelation (Boersma, 1993) — the same
    method Praat uses, and one of the most consistently reported acoustic markers in
    Parkinson's/dysarthria voice research. Lower HNR = breathier, noisier voicing.
    """
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)
    window = np.hanning(frame_length)

    hnr_values = []
    for start in range(0, max(len(y) - frame_length, 0) + 1, hop_length):
        frame = y[start:start + frame_length]
        if len(frame) < frame_length or np.max(np.abs(frame)) < 1e-6:
            continue

        windowed = frame * window
        acf = np.correlate(windowed, windowed, mode='full')
        acf = acf[len(acf) // 2:]
        zero_lag = acf[0]
        if zero_lag <= 0:
            continue
        acf = acf / zero_lag

        if max_lag >= len(acf) or min_lag >= max_lag:
            continue
        candidate = acf[min_lag:max_lag]
        if len(candidate) == 0:
            continue

        r_max = float(np.max(candidate))
        r_max = min(max(r_max, 0.0), 0.999999)  # keep log() well-defined
        if r_max <= 0.0:
            continue
        hnr_values.append(10.0 * np.log10(r_max / (1.0 - r_max)))

    if not hnr_values:
        return {'hnr_mean': 0.0, 'hnr_var': 0.0}

    return {
        'hnr_mean': _safe_float(np.mean(hnr_values)),
        'hnr_var': _safe_float(np.var(hnr_values)),
    }


# ── Shimmer + Energy (single RMS call) ───────────────────

def extract_shimmer_and_energy(y: np.ndarray, sr: int,
                                frame_length: int = 2048,
                                hop_length: int = 512) -> dict:
    """Extract shimmer + energy stats from a single RMS pass."""
    rms = librosa.feature.rms(y=y, frame_length=frame_length,
                               hop_length=hop_length)[0]
    result = {
        'energy_mean': _safe_float(np.mean(rms)),
        'energy_var': _safe_float(np.var(rms)),
        'energy_range': _safe_float(np.ptp(rms)) if len(rms) > 0 else 0.0,
    }

    if len(rms) >= 2:
        diffs = np.abs(np.diff(rms))
        mean_amp = np.mean(rms)
        result['shimmer_local'] = _safe_float(np.mean(diffs) / (mean_amp + 1e-10))
        result['shimmer_abs'] = _safe_float(np.mean(diffs))
    else:
        result['shimmer_local'] = 0.0
        result['shimmer_abs'] = 0.0

    return result


# ── Spectral Centroid ─────────────────────────────────────

def extract_spectral_centroid(y: np.ndarray, sr: int) -> dict:
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    return {
        'spectral_centroid_mean': _safe_float(np.mean(cent)),
        'spectral_centroid_var': _safe_float(np.var(cent)),
    }


# ── Speech Rate ──────────────────────────────────────────

def estimate_speech_rate(y: np.ndarray, sr: int) -> dict:
    """Estimate speech rate using RMS peak events as a fast syllable proxy."""
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    duration = len(y) / sr
    if len(rms) >= 3:
        threshold = np.percentile(rms, 70)
        peaks = (
            (rms[1:-1] > rms[:-2]) &
            (rms[1:-1] >= rms[2:]) &
            (rms[1:-1] > threshold)
        )
        syllable_count = int(np.sum(peaks))
    else:
        syllable_count = 0
    rate = syllable_count / duration if duration > 0 else 0.0
    return {
        'speech_rate': _safe_float(rate),
        'syllable_count': syllable_count,
        'duration_sec': _safe_float(duration),
    }


# ── Full Pipeline ────────────────────────────────────────

def extract_all_acoustic_features(y: np.ndarray, sr: int) -> dict:
    """Run all acoustic feature extractors and merge results."""
    features = {}
    features.update(extract_mfccs(y, sr))
    features.update(extract_pitch_and_jitter(y, sr))       # single pYIN call
    features.update(extract_hnr(y, sr))
    features.update(extract_shimmer_and_energy(y, sr))      # single RMS call
    features.update(extract_spectral_centroid(y, sr))
    features.update(estimate_speech_rate(y, sr))
    return features
