"""보안 유틸 — 비밀번호 해시 + JWT 발급/검증.

책임: 암호학적 원시연산만 제공한다(인증 정책/RBAC 은 deps.py·service 에서).
- 비밀번호: bcrypt 직접 사용(passlib 의존성 회피로 버전 충돌 최소화).
- 토큰: PyJWT HS256.
"""
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from core.config import settings


# ----------------------------------------------------------
# 비밀번호 해시
# ----------------------------------------------------------
def hash_password(plain_password: str) -> str:
    """평문 비밀번호를 bcrypt 해시(문자열)로 변환."""
    salt = bcrypt.gensalt()
    digest = bcrypt.hashpw(plain_password.encode("utf-8"), salt)
    return digest.decode("utf-8")


def verify_password(plain_password: str, password_hash: str) -> bool:
    """평문과 저장된 해시를 비교. 잘못된 해시 포맷이면 False."""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"), password_hash.encode("utf-8")
        )
    except (ValueError, TypeError):
        return False


# ----------------------------------------------------------
# JWT
# ----------------------------------------------------------
def create_access_token(subject: str, role: str) -> str:
    """액세스 토큰 발급.

    payload: sub(사용자 id) · role(RBAC) · exp(만료).
    """
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=settings.access_token_expire_minutes)
    payload = {
        "sub": subject,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    return jwt.encode(payload, settings.jwt_signing_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict:
    """토큰 디코드/검증. 실패 시 jwt 예외를 그대로 올린다(상위에서 401 처리)."""
    return jwt.decode(token, settings.jwt_verify_key, algorithms=[settings.jwt_algorithm])
