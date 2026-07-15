"""인증 API (얇은 레이어 — 검증/위임만)."""
from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from core.deps import get_current_user, get_db, require_roles
from core.query_optimizer import Page
from models.user import User
from schemas.auth import (
    AccountOut,
    ApprovalUpdate,
    LoginRequest,
    SessionUser,
    SignUpRequest,
    TokenResponse,
    UserSignatureUpdate,
)
from services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


def _account_out(user: User) -> AccountOut:
    return AccountOut(
        id=user.id,
        username=user.username,
        name=user.name,
        role=user.role,
        department=user.department,
        approval=user.approval,
        created_at=user.created_at.isoformat() if user.created_at else None,
        last_login_at=user.last_login_at.isoformat() if user.last_login_at else None,
    )


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, request: Request, db: Session = Depends(get_db)):
    client_ip = request.client.host if request.client else None
    user, token = AuthService(db).login(body.username, body.password, client_ip)
    return TokenResponse(
        access_token=token,
        user=SessionUser.model_validate(user),
    )


@router.post("/signup", status_code=201)
def sign_up(body: SignUpRequest, db: Session = Depends(get_db)):
    AuthService(db).sign_up(
        username=body.username,
        password=body.password,
        name=body.name,
        role=body.role,
        department=body.department,
        signature_base64=body.signature_base64,
    )
    return {"message": "가입 요청이 접수되었습니다. 관리자 승인 후 로그인할 수 있습니다."}


@router.get("/me", response_model=SessionUser)
def me(current_user: User = Depends(get_current_user)):
    return SessionUser.model_validate(current_user)


@router.post("/me/signature", response_model=SessionUser)
def update_signature(
    body: UserSignatureUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    import base64
    import os
    
    signatures_dir = os.path.join(os.getcwd(), "uploads", "signatures")
    os.makedirs(signatures_dir, exist_ok=True)
    
    signature_b64 = body.signature_base64
    if "," in signature_b64:
        signature_b64 = signature_b64.split(",")[1]
        
    image_data = base64.b64decode(signature_b64)
    filename = f"sig_user_{current_user.id}.png"
    filepath = os.path.join(signatures_dir, filename)
    
    with open(filepath, "wb") as f:
        f.write(image_data)
        
    current_user.signature_path = f"/uploads/signatures/{filename}"
    db.commit()
    
    return SessionUser.model_validate(current_user)


@router.get("/accounts", response_model=list[AccountOut])
def list_accounts(
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db),
    _: User = Depends(require_roles("admin")),
):
    accounts = AuthService(db).list_accounts(Page.of(limit, offset))
    return [_account_out(a) for a in accounts]


@router.patch("/accounts/{user_id}/approval", response_model=AccountOut)
def set_approval(
    user_id: str,
    body: ApprovalUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_roles("admin")),
):
    user = AuthService(db).set_approval(user_id, body.approval, admin.id, body.rejection_reason)
    return _account_out(user)


from schemas.auth import ApprovalHistoryOut

@router.get("/accounts/{user_id}/approval_history", response_model=list[ApprovalHistoryOut])
def get_approval_history(
    user_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(require_roles("admin")),
):
    histories = AuthService(db).get_approval_history(user_id)
    return [
        ApprovalHistoryOut(
            id=h.id,
            user_id=h.user_id,
            admin_id=h.admin_id,
            actor_type=h.actor_type,
            old_status=h.old_status,
            new_status=h.new_status,
            reason=h.reason,
            created_at=h.created_at.isoformat()
        )
        for h in histories
    ]
