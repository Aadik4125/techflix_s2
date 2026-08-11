"""
CogniVara - Transactional email via Resend.
Best-effort only: a failure here must never break signup or any other flow.
"""

from __future__ import annotations

import logging

import requests

from config import FRONTEND_BASE_URL, RESEND_API_KEY, RESEND_FROM_EMAIL

logger = logging.getLogger(__name__)

RESEND_API_URL = 'https://api.resend.com/emails'


def send_verification_email(to_email: str, token: str) -> bool:
    if not RESEND_API_KEY:
        logger.warning('RESEND_API_KEY not configured; skipping verification email to %s', to_email)
        return False

    verify_link = f'{FRONTEND_BASE_URL.rstrip("/")}/?verify_email={token}'

    try:
        resp = requests.post(
            RESEND_API_URL,
            headers={
                'Authorization': f'Bearer {RESEND_API_KEY}',
                'Content-Type': 'application/json',
            },
            json={
                'from': RESEND_FROM_EMAIL,
                'to': [to_email],
                'subject': 'Verify your CogniVara email',
                'html': (
                    '<p>Welcome to CogniVara.</p>'
                    f'<p><a href="{verify_link}">Click here to verify your email address</a>.</p>'
                    '<p>If you didn\'t sign up for CogniVara, you can ignore this email.</p>'
                ),
            },
            timeout=10,
        )
        if resp.status_code >= 400:
            logger.warning('Resend API returned %s: %s', resp.status_code, resp.text[:500])
            return False
        return True
    except requests.RequestException as exc:
        logger.warning('Failed to send verification email to %s: %s', to_email, exc)
        return False
