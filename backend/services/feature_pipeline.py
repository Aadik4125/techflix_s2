"""
CogniVara - Shared audio -> feature extraction glue.
Used by both the authenticated /api/upload route and the stateless /api/demo/* routes,
so this logic exists exactly once.
"""

from __future__ import annotations

import io
import re
from typing import Any

import librosa
from fastapi import HTTPException

from config import FAST_ANALYSIS_MODE
from services.acoustic_features import extract_all_acoustic_features
from services.linguistic_features import extract_linguistic_features
from services.preprocessing import preprocess_audio
from services.temporal_features import extract_temporal_features


def _fast_linguistic_fallback(text: str) -> dict[str, float]:
    words = re.findall(r"[a-zA-Z']+", text.lower() if text else '')
    word_count = len(words)
    unique_count = len(set(words))
    filler_set = {'um', 'uh', 'like', 'actually', 'basically', 'so', 'well', 'right'}
    filler_count = sum(1 for w in words if w in filler_set)
    lexical_diversity = (unique_count / word_count) if word_count else 0.0
    filler_ratio = (filler_count / word_count) if word_count else 0.0
    sentence_count = max(1, text.count('.') + text.count('!') + text.count('?')) if text else 0
    sentence_length_mean = (word_count / sentence_count) if sentence_count else 0.0
    return {
        'sentence_length_mean': round(float(sentence_length_mean), 4),
        'lexical_diversity': round(float(lexical_diversity), 4),
        'avg_word_length': round(float(sum(len(w) for w in words) / word_count), 4) if word_count else 0.0,
        'filler_ratio': round(float(filler_ratio), 4),
        'content_word_ratio': round(float(max(0.0, 1.0 - filler_ratio)), 4),
        'syntactic_complexity': round(float(min(3.0, sentence_count / 3.0)), 4),
        'vocabulary_richness': round(float((word_count ** (unique_count ** -0.172)) if unique_count > 0 else 0.0), 4),
        'word_count': word_count,
        'sentence_count': sentence_count,
    }


def extract_session_features(audio_bytes: bytes, transcript: str) -> dict[str, Any]:
    """
    Decode audio and run the full acoustic/temporal/linguistic extraction pipeline.
    Raises HTTPException(400) if the audio can't be decoded.
    """
    if FAST_ANALYSIS_MODE:
        preprocessing = {
            'duration_sec': 0.0,
            'speech_duration_sec': 0.0,
            'speech_ratio': 0.0,
            'num_segments': 0,
        }
        acoustic: dict[str, Any] = {}
        temporal: dict[str, Any] = {}
        linguistic = _fast_linguistic_fallback(transcript) if transcript.strip() else {}
        return {
            'preprocessing': preprocessing,
            'acoustic': acoustic,
            'temporal': temporal,
            'linguistic': linguistic,
        }

    try:
        y, sr_loaded = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        sr = int(sr_loaded)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'Failed to decode audio: {str(exc)}')

    preprocess_result = preprocess_audio(y, sr)

    acoustic = extract_all_acoustic_features(preprocess_result['y_speech'], sr)

    temporal = extract_temporal_features(
        preprocess_result['y_clean'], sr, preprocess_result['intervals']
    )
    temporal.update(
        {
            'speech_ratio': round(float(preprocess_result['speech_ratio']), 4),
            'speech_duration_sec': round(float(preprocess_result['speech_duration_sec']), 4),
            'speech_segment_count': int(preprocess_result['num_segments']),
        }
    )

    linguistic = extract_linguistic_features(transcript) if transcript.strip() else {}

    return {
        'preprocessing': {
            'duration_sec': preprocess_result['duration_sec'],
            'speech_duration_sec': preprocess_result['speech_duration_sec'],
            'speech_ratio': preprocess_result['speech_ratio'],
            'num_segments': preprocess_result['num_segments'],
        },
        'acoustic': acoustic,
        'temporal': temporal,
        'linguistic': linguistic,
    }
