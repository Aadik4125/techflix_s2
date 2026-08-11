"""
Plain-language interpretation: a doctor suggestion must only ever appear for a SUSTAINED
pattern (several consecutive check-ins), never a single dip -- and nothing in the output may
leak a raw internal feature/domain name or name a specific illness.
"""

from services.interpretation import (
    DOMAIN_DOCTOR_SUGGESTIONS,
    DOMAIN_PLAIN_NAMES,
    build_user_message,
)


class TestTiers:
    def test_low_confidence_is_insufficient_data(self):
        result = build_user_message(csi_score=50, confidence=0.2, flagged_features=[], domain_streaks={})
        assert result['tier'] == 'insufficient_data'
        assert result['doctor_suggestion'] is None

    def test_stable_pattern_has_no_doctor_suggestion(self):
        result = build_user_message(
            csi_score=85, confidence=0.9, flagged_features=[], domain_streaks={'acoustic': 0}
        )
        assert result['tier'] == 'stable'
        assert result['doctor_suggestion'] is None

    def test_single_flagged_session_is_one_off_not_sustained(self):
        """The whole point of streak-gating: one bad day must never trigger a doctor suggestion."""
        result = build_user_message(
            csi_score=50, confidence=0.9, flagged_features=['jitter_local'], domain_streaks={'acoustic': 1}
        )
        assert result['tier'] == 'one_off'
        assert result['doctor_suggestion'] is None

    def test_streak_below_threshold_does_not_trigger_sustained(self):
        result = build_user_message(
            csi_score=40, confidence=0.9, flagged_features=['jitter_local'], domain_streaks={'acoustic': 2}
        )
        assert result['tier'] != 'sustained'
        assert result['doctor_suggestion'] is None

    def test_sustained_streak_triggers_doctor_suggestion(self):
        result = build_user_message(
            csi_score=40, confidence=0.9, flagged_features=['jitter_local'], domain_streaks={'acoustic': 3}
        )
        assert result['tier'] == 'sustained'
        assert result['doctor_suggestion'] == DOMAIN_DOCTOR_SUGGESTIONS['acoustic']

    def test_sustained_picks_the_longest_streak_as_primary_domain(self):
        result = build_user_message(
            csi_score=30,
            confidence=0.9,
            flagged_features=['lexical_diversity', 'jitter_local'],
            domain_streaks={'acoustic': 3, 'linguistic': 5},
        )
        assert result['doctor_suggestion'] == DOMAIN_DOCTOR_SUGGESTIONS['linguistic']
        assert 'word choice' in result['headline']


class TestNoJargonLeaks:
    RAW_NAMES = [
        'acoustic', 'temporal', 'linguistic', 'z_score', 'csi',
        'jitter_local', 'shimmer_local', 'hnr_mean', 'pitch_var', 'pause_variability',
    ]

    def test_sustained_message_has_no_raw_feature_or_domain_names(self):
        result = build_user_message(
            csi_score=35,
            confidence=0.9,
            flagged_features=['jitter_local', 'hnr_mean', 'pitch_var'],
            domain_streaks={'acoustic': 4},
        )
        combined = f"{result['headline']} {result['detail']} {result['doctor_suggestion']}"
        for raw in self.RAW_NAMES:
            assert raw not in combined, f'raw internal name "{raw}" leaked into user-facing text'

    def test_doctor_suggestions_never_name_a_specific_illness(self):
        illness_words = [
            'alzheimer', 'dementia', 'parkinson', 'depression', 'anxiety', 'schizophrenia', 'bipolar',
        ]
        for domain, suggestion in DOMAIN_DOCTOR_SUGGESTIONS.items():
            lowered = suggestion.lower()
            for word in illness_words:
                assert word not in lowered, f'{domain} doctor suggestion names a specific illness: {word}'

    def test_every_plain_domain_has_a_doctor_suggestion(self):
        assert set(DOMAIN_PLAIN_NAMES.keys()) == set(DOMAIN_DOCTOR_SUGGESTIONS.keys())
