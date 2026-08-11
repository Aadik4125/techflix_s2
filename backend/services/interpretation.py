"""
CogniVara - Plain-language interpretation
Translates domain/feature-level scoring signals into non-technical, symptom-oriented
language for the end user, and suggests what kind of doctor to start with when a pattern
has been sustained across multiple check-ins — never from a single session, since a single
off day is expected noise, not a pattern (see the acoustic/temporal/linguistic domain split
in services/csi.py, which this builds directly on).

None of this is a diagnosis, and none of it names a specific illness — the underlying
research shows these signals overlap heavily across conditions (the same pause/pitch/rate
changes show up in depression, Parkinson's, and cognitive decline alike), so the honest and
defensible claim is "this domain has been trending differently," not "you may have X".
"""

from __future__ import annotations

DOMAIN_PLAIN_NAMES = {
    'acoustic': 'how steady and clear your voice sounds',
    'temporal': 'your pacing and pauses when speaking',
    'linguistic': 'your word choice and sentence patterns',
}

# Every suggestion leads with "your regular doctor" — these domains are not disease-specific,
# so a specialist is framed as something your doctor may refer you to, never a direct pointer.
DOMAIN_DOCTOR_SUGGESTIONS = {
    'acoustic': (
        'your primary care doctor — they may bring in an ENT (ear, nose & throat) specialist '
        'or a neurologist to take a closer look at voice and speech-motor changes'
    ),
    'temporal': (
        'your primary care doctor — pacing and pause changes can have several different causes, '
        "so they're best placed to decide whether a neurologist or a mental health professional "
        'is the right next step'
    ),
    'linguistic': (
        'your primary care doctor — they may refer you to a neurologist or a memory/cognitive '
        'specialist to look at word-finding and language changes'
    ),
}

# Grouped so several related raw feature names collapse into one recognizable, plain phrase.
# Order matters: the first matching group wins for a given flagged feature.
_FEATURE_PHRASE_GROUPS: list[tuple[tuple[str, ...], str]] = [
    (
        ('jitter_local', 'shimmer_local', 'hnr_mean', 'spectral_centroid_mean', 'spectral_centroid_var', 'mfcc_variance_avg'),
        'your voice has sounded less steady or clear than usual',
    ),
    (
        ('pitch_var', 'pitch_range'),
        'your voice has sounded flatter, with less natural rise and fall in pitch',
    ),
    (
        ('pause_variability', 'mean_pause_duration', 'max_pause_duration', 'pause_count', 'response_latency'),
        "you've been pausing more, or taking longer to respond",
    ),
    (
        ('speech_rate', 'rhythm_consistency', 'speed_variability'),
        'your speaking pace has been less consistent',
    ),
    (
        ('lexical_diversity', 'vocabulary_richness', 'content_word_ratio', 'avg_word_length'),
        "you've been using a narrower range of words",
    ),
    (
        ('filler_ratio', 'sentence_length_mean', 'syntactic_complexity'),
        'your sentences have sounded shorter or more hesitant',
    ),
]


def _plain_phrases_for(flagged_features: list[str], limit: int = 2) -> list[str]:
    phrases: list[str] = []
    for keys, phrase in _FEATURE_PHRASE_GROUPS:
        if phrase in phrases:
            continue
        if any(f in keys for f in flagged_features):
            phrases.append(phrase)
        if len(phrases) >= limit:
            break
    return phrases


def build_user_message(
    *,
    csi_score: int,
    confidence: float,
    flagged_features: list[str] | None,
    domain_streaks: dict[str, int] | None,
    sustained_threshold: int = 3,
) -> dict:
    """
    Returns a fully non-technical message for the end user:
      { 'tier', 'headline', 'detail', 'doctor_suggestion' }
    'doctor_suggestion' is only ever populated for the 'sustained' tier.
    """
    flagged_features = flagged_features or []
    domain_streaks = domain_streaks or {}

    if confidence < 0.45:
        return {
            'tier': 'insufficient_data',
            'headline': "We don't have enough check-ins yet to spot a reliable pattern.",
            'detail': 'Keep checking in regularly — this becomes more accurate the more you use it.',
            'doctor_suggestion': None,
        }

    sustained_domains = sorted(
        (d for d, streak in domain_streaks.items() if streak >= sustained_threshold and d in DOMAIN_PLAIN_NAMES),
        key=lambda d: domain_streaks[d],
        reverse=True,
    )

    if sustained_domains:
        primary_domain = sustained_domains[0]
        phrases = _plain_phrases_for(flagged_features)
        phrase_text = f' — for example, {phrases[0]}' if phrases else ''
        other_domains = [DOMAIN_PLAIN_NAMES[d] for d in sustained_domains[1:]]
        also_text = f' We also noticed something similar in {", ".join(other_domains)}.' if other_domains else ''
        return {
            'tier': 'sustained',
            'headline': (
                f"Over your last few check-ins, we've consistently noticed changes in "
                f'{DOMAIN_PLAIN_NAMES[primary_domain]}{phrase_text}.{also_text}'
            ),
            'detail': (
                "This isn't a diagnosis — just a pattern in your own recordings over time. "
                'Patterns like this are worth mentioning to a doctor.'
            ),
            'doctor_suggestion': DOMAIN_DOCTOR_SUGGESTIONS[primary_domain],
        }

    if flagged_features and csi_score < 60:
        phrases = _plain_phrases_for(flagged_features)
        phrase_text = phrases[0] if phrases else 'a few patterns in your speech'
        return {
            'tier': 'one_off',
            'headline': f'Today looked a little different from your usual pattern ({phrase_text}).',
            'detail': "That's normal and often just means an off day. We'll keep watching for a pattern.",
            'doctor_suggestion': None,
        }

    return {
        'tier': 'stable',
        'headline': 'Your check-ins have been consistent with your usual pattern.',
        'detail': 'Nothing to flag right now — keep going.',
        'doctor_suggestion': None,
    }
