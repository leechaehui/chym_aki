"""FastAPI 의존성 — DB 세션 / 현재 사용자 / RBAC.

책임: 인증·인가 횡단 관심사를 한 곳에 모은다.
API 레이어는 이 의존성만 선언하고 비즈니스 로직은 service 에 위임한다.
"""
from collections.abc import Callable

import jwt
from fastapi import Depends, Header
from sqlalchemy.orm import Session

from core.database import get_db
from core.exceptions import AuthError, PermissionError_
from core.security import decode_access_token
from models.user import User
from repositories.user_repository import UserRepository

__all__ = ["get_db", "get_current_user", "require_roles"]


def get_current_user(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> User:
    """Bearer 토큰 → 현재 사용자. 누락/무효/만료 시 401."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise AuthError("인증 토큰이 필요합니다.")
    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = decode_access_token(token)
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("토큰이 만료되었습니다.") from exc
    except jwt.PyJWTError as exc:
        raise AuthError("유효하지 않은 토큰입니다.") from exc

    user = UserRepository(db).get(payload.get("sub", ""))
    if not user:
        raise AuthError("계정을 찾을 수 없습니다.")
    if user.approval != "approved":
        raise AuthError("승인되지 않은 계정입니다.")
    return user


def require_roles(*roles: str) -> Callable[..., User]:
    """지정 role 만 통과시키는 의존성 팩토리(RBAC).

    예) Depends(require_roles("emergency", "admin"))
    """

    def _guard(current_user: User = Depends(get_current_user)) -> User:
        if roles and current_user.role not in roles:
            raise PermissionError_("이 작업을 수행할 권한이 없습니다.")
        return current_user

    return _guard
