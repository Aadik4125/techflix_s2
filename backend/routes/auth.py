from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Form, Header, HTTPException
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession

from config import AUTH_TOKEN_TTL_DAYS, EMAIL_VERIFICATION_TTL_HOURS, GOOGLE_CLIENT_ID
from database import get_db
from models.auth_token import AuthToken
from models.email_verification_token import EmailVerificationToken
from models.user import User
from services.email import send_verification_email
from services.security import generate_token, hash_password, verify_password

logger = logging.getLogger('cognivara.auth')

router = APIRouter()

EMAIL_PATTERN = re.compile(r'^[^\s@]+@[^\s@]+\.[^\s@]+$')


def _utcnow() -> datetime:
    """Naive UTC 'now', matching how DateTime columns actually round-trip through SQLite/Postgres here."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _issue_token(db: DBSession, user_id: int) -> AuthToken:
    auth_token = AuthToken(
        token=generate_token(),
        user_id=user_id,
        expires_at=_utcnow() + timedelta(days=AUTH_TOKEN_TTL_DAYS),
    )
    db.add(auth_token)
    db.commit()
    db.refresh(auth_token)
    return auth_token


def _send_verification_email(db: DBSession, user: User) -> None:
    """Best-effort: issues a verification token and emails it. Never raises."""
    verification_token = EmailVerificationToken(
        token=generate_token(),
        user_id=user.id,
        expires_at=_utcnow() + timedelta(hours=EMAIL_VERIFICATION_TTL_HOURS),
    )
    db.add(verification_token)
    db.commit()
    send_verification_email(user.email, verification_token.token)


def get_current_auth_token(
    authorization: str | None = Header(None),
    db: DBSession = Depends(get_db),
) -> AuthToken:
    if not authorization or not authorization.lower().startswith('bearer '):
        raise HTTPException(status_code=401, detail='Missing or invalid Authorization header')
    token = authorization.split(' ', 1)[1].strip()

    auth_token = db.query(AuthToken).filter(AuthToken.token == token).first()
    if auth_token is None:
        raise HTTPException(status_code=401, detail='Invalid or expired session')
    if auth_token.expires_at < _utcnow():
        db.delete(auth_token)
        db.commit()
        raise HTTPException(status_code=401, detail='Invalid or expired session')

    return auth_token


def get_current_user(
    auth_token: AuthToken = Depends(get_current_auth_token),
    db: DBSession = Depends(get_db),
) -> User:
    user = db.query(User).filter(User.id == auth_token.user_id).first()
    if user is None:
        raise HTTPException(status_code=401, detail='Invalid or expired session')
    return user


def _user_payload(user: User) -> dict:
    return {
        'user_id': user.id,
        'name': user.name,
        'email': user.email,
        'age': user.age,
        'gender': user.gender,
        'email_verified': bool(user.email_verified),
        'google_linked': bool(user.google_sub),
        'latest_csi_score': user.latest_csi_score,
        'total_sessions': user.total_sessions,
        'last_session_at': user.last_session_at.isoformat() if user.last_session_at else None,
        'created_at': user.created_at.isoformat() if user.created_at else None,
    }


@router.post('/signup')
def signup(
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    age: int | None = Form(None),
    gender: str | None = Form(None),
    db: DBSession = Depends(get_db),
):
    if len(password) < 8:
        raise HTTPException(status_code=422, detail='Password must be at least 8 characters')
    if not EMAIL_PATTERN.match(email):
        raise HTTPException(status_code=422, detail='Please provide a valid email address')

    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=409, detail='An account with this email already exists')

    user = User(
        name=name,
        email=email,
        age=age,
        gender=gender,
        password_hash=hash_password(password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Best-effort — signup succeeds regardless of whether the email actually goes out.
    _send_verification_email(db, user)

    auth_token = _issue_token(db, user.id)
    logger.info('signup succeeded for user_id=%s', user.id)

    return {
        **_user_payload(user),
        'token': auth_token.token,
        'expires_at': auth_token.expires_at.isoformat(),
    }


@router.post('/login')
def login(
    email: str = Form(...),
    password: str = Form(...),
    db: DBSession = Depends(get_db),
):
    user = db.query(User).filter(User.email == email).first()
    if user is None or not verify_password(password, user.password_hash):
        # Log the attempted email (not the password) — a spike here is a real signal
        # (credential stuffing, a broken client), but never log secrets.
        logger.warning('login failed for email=%s', email)
        raise HTTPException(status_code=401, detail='Incorrect email or password')

    auth_token = _issue_token(db, user.id)
    logger.info('login succeeded for user_id=%s', user.id)

    return {
        **_user_payload(user),
        'token': auth_token.token,
        'expires_at': auth_token.expires_at.isoformat(),
    }


class GoogleSignInRequest(BaseModel):
    id_token: str


@router.post('/google')
def google_sign_in(req: GoogleSignInRequest, db: DBSession = Depends(get_db)):
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=503, detail='Google sign-in is not configured on this server')

    try:
        claims = google_id_token.verify_oauth2_token(
            req.id_token, google_requests.Request(), GOOGLE_CLIENT_ID
        )
    except Exception as exc:
        # Covers bad signature, wrong audience, expired token, and cert-fetch failures alike —
        # all of them mean "we can't trust this credential."
        logger.warning('Google sign-in token verification failed: %s', exc)
        raise HTTPException(status_code=401, detail='Invalid Google credential')

    google_sub = claims.get('sub')
    email = claims.get('email')
    name = claims.get('name') or (email.split('@')[0] if email else 'Google User')
    if not google_sub or not email:
        raise HTTPException(status_code=401, detail='Google credential missing required claims')

    user = db.query(User).filter(User.google_sub == google_sub).first()
    if user is None:
        # Link to an existing password account with the same email rather than duplicating it.
        user = db.query(User).filter(User.email == email).first()

    if user is None:
        user = User(
            name=name,
            email=email,
            password_hash=None,
            google_sub=google_sub,
            email_verified=True,
        )
        db.add(user)
        is_new_account = True
    else:
        user.google_sub = google_sub
        user.email_verified = True  # Google just re-confirmed this email address
        is_new_account = False

    db.commit()
    db.refresh(user)

    auth_token = _issue_token(db, user.id)
    logger.info(
        'Google sign-in succeeded for user_id=%s (%s)',
        user.id, 'new account' if is_new_account else 'existing account',
    )

    return {
        **_user_payload(user),
        'token': auth_token.token,
        'expires_at': auth_token.expires_at.isoformat(),
    }


@router.post('/logout')
def logout(
    auth_token: AuthToken = Depends(get_current_auth_token),
    db: DBSession = Depends(get_db),
):
    db.delete(auth_token)
    db.commit()
    return {'status': 'ok'}


class VerifyEmailRequest(BaseModel):
    token: str


@router.post('/verify-email')
def verify_email(req: VerifyEmailRequest, db: DBSession = Depends(get_db)):
    verification_token = (
        db.query(EmailVerificationToken).filter(EmailVerificationToken.token == req.token).first()
    )
    if verification_token is None:
        raise HTTPException(status_code=400, detail='Invalid or already-used verification link')
    if verification_token.expires_at < _utcnow():
        db.delete(verification_token)
        db.commit()
        raise HTTPException(status_code=400, detail='Verification link has expired')

    user = db.query(User).filter(User.id == verification_token.user_id).first()
    if user is None:
        raise HTTPException(status_code=400, detail='Invalid verification link')

    user.email_verified = True
    db.delete(verification_token)
    db.commit()

    return {'status': 'ok', 'email_verified': True}


@router.post('/resend-verification')
def resend_verification(
    current_user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    if current_user.email_verified:
        return {'status': 'already_verified'}

    # Clear out any older outstanding tokens before issuing a fresh one.
    db.query(EmailVerificationToken).filter(
        EmailVerificationToken.user_id == current_user.id
    ).delete()
    db.commit()

    _send_verification_email(db, current_user)
    return {'status': 'sent'}


@router.get('/me')
def get_me(current_user: User = Depends(get_current_user)):
    return _user_payload(current_user)


@router.patch('/me')
def update_me(
    name: str | None = Form(None),
    age: int | None = Form(None),
    gender: str | None = Form(None),
    current_user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    if name is not None:
        current_user.name = name
    if age is not None:
        current_user.age = age
    if gender is not None:
        current_user.gender = gender

    db.commit()
    db.refresh(current_user)
    return _user_payload(current_user)
