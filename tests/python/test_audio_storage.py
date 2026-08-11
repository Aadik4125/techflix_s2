"""
Audio storage: raw recordings must survive a redeploy when S3-compatible storage is
configured, and must never fail a check-in (fall back to local disk instead) when it isn't
configured or when the upload itself fails. Uses moto to mock S3 so the actual boto3 call
shape is exercised for real, not just asserted to "look right".
"""

import moto

from services import audio_storage


def test_save_audio_uses_local_disk_when_unconfigured(tmp_path, monkeypatch):
    monkeypatch.setattr(audio_storage, 'S3_CONFIGURED', False)
    monkeypatch.setattr(audio_storage, 'AUDIO_UPLOAD_DIR', str(tmp_path))

    result = audio_storage.save_audio(b'local disk test bytes', 'session.wav')

    assert result == str(tmp_path / 'session.wav')
    with open(result, 'rb') as f:
        assert f.read() == b'local disk test bytes'


def test_save_audio_uploads_to_s3_when_configured(monkeypatch):
    monkeypatch.setattr(audio_storage, 'S3_CONFIGURED', True)
    monkeypatch.setattr(audio_storage, 'S3_BUCKET', 'cognivara-test-bucket')
    monkeypatch.setattr(audio_storage, 'S3_ACCESS_KEY_ID', 'testkey')
    monkeypatch.setattr(audio_storage, 'S3_SECRET_ACCESS_KEY', 'testsecret')
    monkeypatch.setattr(audio_storage, 'S3_REGION', 'us-east-1')
    monkeypatch.setattr(audio_storage, 'S3_ENDPOINT_URL', '')
    monkeypatch.setattr(audio_storage, '_client', None)

    with moto.mock_aws():
        import boto3

        boto3.client('s3', region_name='us-east-1').create_bucket(Bucket='cognivara-test-bucket')

        result = audio_storage.save_audio(b'real s3 round-trip test', 'session.wav')
        assert result == 's3://cognivara-test-bucket/session.wav'

        obj = boto3.client('s3', region_name='us-east-1').get_object(
            Bucket='cognivara-test-bucket', Key='session.wav'
        )
        assert obj['Body'].read() == b'real s3 round-trip test'


def test_save_audio_falls_back_to_local_disk_when_s3_call_fails(tmp_path, monkeypatch):
    """S3 configured, but the bucket doesn't exist (or credentials are bad) — a raw-audio
    backup failure must never fail the check-in itself."""
    monkeypatch.setattr(audio_storage, 'S3_CONFIGURED', True)
    monkeypatch.setattr(audio_storage, 'S3_BUCKET', 'a-bucket-that-does-not-exist')
    monkeypatch.setattr(audio_storage, 'S3_ACCESS_KEY_ID', 'testkey')
    monkeypatch.setattr(audio_storage, 'S3_SECRET_ACCESS_KEY', 'testsecret')
    monkeypatch.setattr(audio_storage, 'S3_REGION', 'us-east-1')
    monkeypatch.setattr(audio_storage, 'S3_ENDPOINT_URL', '')
    monkeypatch.setattr(audio_storage, '_client', None)
    monkeypatch.setattr(audio_storage, 'AUDIO_UPLOAD_DIR', str(tmp_path))

    with moto.mock_aws():
        # Deliberately do not create the bucket, so put_object raises NoSuchBucket.
        result = audio_storage.save_audio(b'fallback path test', 'session.wav')

    assert result == str(tmp_path / 'session.wav')
    with open(result, 'rb') as f:
        assert f.read() == b'fallback path test'
