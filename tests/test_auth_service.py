"""Service — 인증(로그인/가입/승인) (작업지시서 7.1)."""
import uuid

import pytest

from core.exceptions import AuthError, ConflictError, NotFoundError
from core.query_optimizer import Page
from services.auth_service import AuthService
from fastapi import HTTPException
from tests.factories import make_user


def _uname():
    return f"svc-{uuid.uuid4().hex[:8]}"


def test_login_success_returns_user_and_token(db):
    u = make_user(db, username=_uname(), password="Secret123!", approval="approved")
    user, token = AuthService(db).login(u.username, "Secret123!")
    assert user.id == u.id
    assert token and isinstance(token, str)
    assert user.last_login_at is not None


def test_login_wrong_password_raises(db):
    u = make_user(db, username=_uname(), password="Secret123!")
    with pytest.raises(AuthError):
        AuthService(db).login(u.username, "wrong")


def test_login_unknown_user_raises(db):
    with pytest.raises(AuthError):
        AuthService(db).login("nobody-here", "x")


def test_pending_account_cannot_login(db):
    u = make_user(db, username=_uname(), password="Secret123!", approval="pending")
    with pytest.raises(HTTPException) as exc:
        AuthService(db).login(u.username, "Secret123!")
    assert exc.value.status_code == 403


def test_signup_creates_pending_account(db):
    name = _uname()
    user = AuthService(db).sign_up(
        username=name, password="Secret123!", name="신규", role="nephrology", department="신장내과", signature_base64="ZHVtbXk="
    )
    assert user.approval == "pending"
    # pending 이므로 로그인 차단.
    with pytest.raises(HTTPException) as exc:
        AuthService(db).login(name, "Secret123!")
    assert exc.value.status_code == 403


def test_signup_duplicate_username_raises_conflict(db):
    u = make_user(db, username=_uname())
    with pytest.raises(ConflictError):
        AuthService(db).sign_up(
            username=u.username, password="x", name="dup", role="nephrology", department="d", signature_base64="ZHVtbXk="
        )


def test_set_approval_flow(db):
    u = make_user(db, username=_uname(), password="Secret123!", approval="pending")
    AuthService(db).set_approval(u.id, "approved", admin_id="admin-1")
    # 승인 후 로그인 성공.
    user, token = AuthService(db).login(u.username, "Secret123!")
    assert token


def test_set_approval_invalid_value_raises(db):
    u = make_user(db, approval="pending")
    with pytest.raises(ConflictError):
        AuthService(db).set_approval(u.id, "maybe", admin_id="admin-1")


def test_set_approval_unknown_user_raises_not_found(db):
    with pytest.raises(NotFoundError):
        AuthService(db).set_approval("u-does-not-exist", "approved", admin_id="admin-1")


def test_list_accounts_returns_users(db):
    make_user(db, username=_uname())
    accounts = AuthService(db).list_accounts(Page.of(100, 0))
    assert len(accounts) >= 1
