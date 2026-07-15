"""인증 DTO — 로그인/가입/세션 사용자/계정."""
from schemas.base import CamelModel


class LoginRequest(CamelModel):
    username: str
    password: str


class SignUpRequest(CamelModel):
    username: str
    password: str
    name: str
    role: str  # admin|emergency|nephrology|pathology
    department: str
    signature_base64: str
    employee_id: str | None = None


class SessionUser(CamelModel):
    """UI 로 노출되는 세션 사용자(비밀번호 제외)."""

    id: str
    username: str
    name: str
    role: str
    department: str
    signature_path: str | None = None
    employee_id: str | None = None


class TokenResponse(CamelModel):
    """로그인 응답 — 토큰 + 세션 사용자."""

    access_token: str
    token_type: str = "bearer"
    user: SessionUser


from pydantic import Field

class AccountOut(CamelModel):
    """관리자 화면용 계정 표현(비밀번호 해시 제외)."""

    id: str
    username: str
    name: str
    role: str
    department: str
    approval: str
    created_at: str | None = None
    last_login_at: str | None = None
    rejection_reason: str | None = None
    approved_by: str | None = None
    approved_at: str | None = None
    rejected_by: str | None = None
    rejected_at: str | None = None


class ApprovalUpdate(CamelModel):
    approval: str  # approved | rejected
    rejection_reason: str | None = Field(default=None, max_length=1000)


class ApprovalHistoryOut(CamelModel):
    id: int
    user_id: str
    admin_id: str | None
    actor_type: str
    old_status: str | None
    new_status: str
    reason: str | None
    created_at: str


class UserSignatureUpdate(CamelModel):
    signature_base64: str
