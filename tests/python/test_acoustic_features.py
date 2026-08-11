"""
Direct correctness tests for acoustic feature extractors that need to respond correctly
to actual signal properties, not just avoid crashing -- HNR in particular, since the entire
point of adding it is that it should track periodicity/noise the way Praat's implementation
does (it uses the same Boersma autocorrelation method).
"""

import numpy as np

from services.acoustic_features import extract_all_acoustic_features, extract_hnr


def _sine(freq, sr, duration=1.0, amplitude=0.5):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestHNR:
    def test_pure_tone_has_high_hnr(self):
        """A clean, sustained tone is maximally periodic -- HNR should read clearly positive."""
        sr = 16000
        tone = _sine(150, sr)
        assert extract_hnr(tone, sr)['hnr_mean'] > 10

    def test_white_noise_has_low_hnr(self):
        """Pure noise has no periodicity at all -- HNR should read clearly negative."""
        sr = 16000
        rng = np.random.default_rng(0)
        noise = (0.5 * rng.standard_normal(sr)).astype(np.float32)
        assert extract_hnr(noise, sr)['hnr_mean'] < 0

    def test_adding_noise_to_a_tone_lowers_hnr(self):
        """The core validation: HNR must respond proportionally to actual noise level."""
        sr = 16000
        rng = np.random.default_rng(1)
        tone = _sine(150, sr)
        noisy = (tone + 0.15 * rng.standard_normal(sr)).astype(np.float32)
        assert extract_hnr(noisy, sr)['hnr_mean'] < extract_hnr(tone, sr)['hnr_mean']

    def test_silence_degrades_safely(self):
        sr = 16000
        silence = np.zeros(sr, dtype=np.float32)
        assert extract_hnr(silence, sr) == {'hnr_mean': 0.0, 'hnr_var': 0.0}

    def test_hnr_is_present_in_the_full_acoustic_pipeline(self):
        sr = 16000
        tone = _sine(150, sr)
        result = extract_all_acoustic_features(tone, sr)
        assert 'hnr_mean' in result
        assert 'hnr_var' in result
