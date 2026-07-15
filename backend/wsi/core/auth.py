"""WSI 서버 인증/인가 — 8010 과 동일한 앱 JWT 를 검증해 RBAC 적용.

- 병리과 화면 전용 → **pathology / admin role 만** 통과.
- 기본 RS256(비대칭): 8010 이 개인키로 서명, 여기선 **공개키로 검증만** → 8001 이 털려도 토큰 위조 불가.
  · RS256 검증은 chym_proj 에 있는 cryptography 로 직접 수행(PyJWT 불필요).
- HS256(공유 시크릿)도 폴백 지원(JWT_ALGORITHM=HS256).
- 상태 없음: 토큰 role 클레임만으로 인가(DB 불필요). 시크릿/키는 backend/.env·키파일 단일 소스, 프론트 0.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from functools import lru_cache
from pathlib import Path

from fastapi import Header

from wsi.core.errors import AuthError, ForbiddenError

_ALLOWED_ROLES = {"pathology", "admin"}
_BACKEND_DIR = Path(__file__).resolve().parents[2]          # .../backend
_DEFAULT_SECRET = "dev-only-change-me-in-production-0123456789abcdef"


def _env(key: str, default: str = "") -> str:
    """OS 환경변수 우선, 없으면 backend/.env 파싱."""
    if os.getenv(key):
        return os.environ[key]
    env = _BACKEND_DIR / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith(f"{key}="):
                return s.split("=", 1)[1].strip().strip('"').strip("'")
    return default


@lru_cache(maxsize=1)
def _algorithm() -> str:
    return _env("JWT_ALGORITHM", "RS256")


@lru_cache(maxsize=1)
def _public_key():
    """RS256 공개키 객체(검증용)."""
    from cryptography.hazmat.primitives.serialization import load_pem_public_key
    path = Path(_env("JWT_PUBLIC_KEY_PATH", "keys/jwt_public.pem"))
    if not path.is_absolute():
        path = _BACKEND_DIR / path
    return load_pem_public_key(path.read_bytes())


@lru_cache(maxsize=1)
def _hs_secret() -> str:
    return _env("JWT_SECRET", _DEFAULT_SECRET)


def _b64url(seg: str) -> bytes:
    return base64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4))


def _verify(token: str) -> dict:
    """서명·만료 검증 후 payload 반환. 알고리즘(RS256/HS256)에 맞춰 검증."""
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
    except ValueError:
        raise AuthError("토큰 형식이 올바르지 않습니다")
    signing_input = f"{header_b64}.{payload_b64}".encode()
    signature = _b64url(sig_b64)

    if _algorithm().startswith("RS"):
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        try:
            _public_key().verify(signature, signing_input, padding.PKCS1v15(), hashes.SHA256())
        except InvalidSignature:
            raise AuthError("토큰 서명 검증 실패")
    else:  # HS256 폴백
        expected = hmac.new(_hs_secret().encode(), signing_input, hashlib.sha256).digest()
        if not hmac.compare_digest(expected, signature):
            raise AuthError("토큰 서명 검증 실패")

    payload = json.loads(_b64url(payload_b64))
    exp = payload.get("exp")
    if exp and time.time() > float(exp):
        raise AuthError("토큰이 만료되었습니다")
    return payload


def require_pathology(authorization: str | None = Header(default=None)) -> dict:
    """병리과/admin 만 통과시키는 의존성. 프론트는 Bearer 토큰을 동봉해야 함."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise AuthError("인증 토큰이 필요합니다")
    payload = _verify(authorization.split(" ", 1)[1].strip())
    if payload.get("role") not in _ALLOWED_ROLES:
        raise ForbiddenError("병리과 권한이 필요합니다")
    return payload


# 읽기 전용 WSI 이미지(슬라이드/타일/썸네일/patch)는 협진 리포트를 받은 임상의도 조회 가능.
# 분석(analyze)·PACS 쓰기 등은 여전히 require_pathology.
_VIEW_ROLES = {"pathology", "admin", "nephrology", "emergency"}


def require_view(authorization: str | None = Header(default=None)) -> dict:
    """읽기 전용 WSI 이미지 조회 — 병리과/admin + 협진 임상의(신장내과/응급의학과)."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise AuthError("인증 토큰이 필요합니다")
    payload = _verify(authorization.split(" ", 1)[1].strip())
    if payload.get("role") not in _VIEW_ROLES:
        raise ForbiddenError("조회 권한이 필요합니다")
    return payload
