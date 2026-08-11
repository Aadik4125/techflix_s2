"""
CogniVara - Audio storage
Saves raw recordings to S3-compatible object storage when configured, so they survive
redeploys on a host with an ephemeral filesystem (e.g. Render's free tier). Falls back to
local disk when not configured, matching every other optional integration in this codebase.
"""

from __future__ import annotations

import logging
import os

from config import (
    AUDIO_UPLOAD_DIR,
    S3_ACCESS_KEY_ID,
    S3_BUCKET,
    S3_ENDPOINT_URL,
    S3_REGION,
    S3_SECRET_ACCESS_KEY,
)

logger = logging.getLogger('cognivara.audio_storage')

S3_CONFIGURED = bool(S3_BUCKET and S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY)

_client = None


def _get_client():
    global _client
    if _client is None:
        import boto3

        _client = boto3.client(
            's3',
            region_name=S3_REGION,
            endpoint_url=S3_ENDPOINT_URL or None,
            aws_access_key_id=S3_ACCESS_KEY_ID,
            aws_secret_access_key=S3_SECRET_ACCESS_KEY,
        )
    return _client


def save_audio(audio_bytes: bytes, filename: str) -> str:
    """
    Persist a recording and return an identifier for where it landed: an 's3://bucket/key'
    URI when object storage is configured, otherwise a local filesystem path.

    Never raises for a storage failure — a check-in must not fail just because the raw-audio
    backup failed. Falls back to local disk and logs a warning instead, same as every other
    best-effort integration in this codebase (HF/Groq/Resend).
    """
    if S3_CONFIGURED:
        try:
            client = _get_client()
            client.put_object(Bucket=S3_BUCKET, Key=filename, Body=audio_bytes)
            return f's3://{S3_BUCKET}/{filename}'
        except Exception:
            logger.exception('Failed to upload audio to S3, falling back to local disk: %s', filename)

    filepath = os.path.join(AUDIO_UPLOAD_DIR, filename)
    with open(filepath, 'wb') as file_obj:
        file_obj.write(audio_bytes)
    return filepath
