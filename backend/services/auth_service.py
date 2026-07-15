"""인증 서비스.

책임: 로그인/가입/승인 비즈니스 규칙 + 토큰 발급.
- 비밀번호는 해시 비교(평문 저장 없음).
- 가입은 pending 상태로 생성(관리자 승인 필요).
- 상태변경(가입/승인/로그인시각)은 트랜잭션 + 감사로그.
"""
from sqlalchemy.orm import Session

import json
from fastapi import HTTPException
from core.exceptions import AuthError, ConflictError, NotFoundError, ValidationFailedError
from core.query_optimizer import Page
from core.security import create_access_token, hash_password, verify_password
from models.base import new_id, utcnow
from models.user import User
from models.approval_history import ApprovalHistory
from repositories.user_repository import UserRepository
from services.audit_service import AuditService


class AuthService:
    def __init__(self, db: Session):
        self.db = db
        self.users = UserRepository(db)
        self.audit = AuditService(db)

    def login(self, username: str, password: str, client_ip: str | None = None) -> tuple[User, str]:
        """로그인 — 승인 계정 + 비밀번호 일치 시 토큰 발급. last_login_at 갱신."""
        user = self.users.get_by_username(username)
        if not user or not verify_password(password, user.password_hash):
            raise AuthError("아이디 또는 비밀번호가 올바르지 않습니다.")
        
        if user.approval == "pending":
            raise HTTPException(status_code=403, detail={"code": "ACCOUNT_PENDING"})
            
        if user.approval == "rejected":
            # 5분 단위 Rate Limiting (동일 User ID)
            five_mins_ago = utcnow().timestamp() - 300
            # audit.record 는 직접 db insert.
            # 하지만 간단하게 그냥 기록 남기자
            # 중복 방지를 위해 가장 최근 기록 확인
            # 간단히 여기서 AuditLog 생성
            from models.audit import AuditLog
            from sqlalchemy import select
            
            stmt = select(AuditLog).where(
                AuditLog.action == "LOGIN_REJECTED_ACCOUNT",
                AuditLog.target_id == user.id
            ).order_by(AuditLog.created_at.desc()).limit(1)
            
            last_log = self.db.execute(stmt).scalar_one_or_none()
            if not last_log or last_log.created_at.timestamp() < five_mins_ago:
                self.audit.record(
                    user_id=user.id, action="LOGIN_REJECTED_ACCOUNT", target_type="user", target_id=user.id,
                    payload={"client_ip": client_ip}
                )
                self.db.commit()

            raise HTTPException(
                status_code=403,
                detail={"code": "ACCOUNT_REJECTED", "reason": user.rejection_reason}
            )

        user.last_login_at = utcnow()
        token = create_access_token(subject=user.id, role=user.role)
        self.audit.record(
            user_id=user.id, action="login", target_type="user", target_id=user.id
        )
        self.db.commit()
        return user, token

    def sign_up(
        self, *, username: str, password: str, name: str, role: str, department: str, signature_base64: str
    ) -> User:
        """가입 — 중복 아이디 차단, pending 상태 생성."""
        if self.users.exists_username(username):
            raise ConflictError("이미 사용 중인 아이디입니다.")
        user_id = new_id("u")
        
        import base64
        import os
        
        signatures_dir = os.path.join(os.getcwd(), "uploads", "signatures")
        os.makedirs(signatures_dir, exist_ok=True)
        
        sig_b64 = signature_base64
        if "," in sig_b64:
            sig_b64 = sig_b64.split(",")[1]
            
        try:
            image_data = base64.b64decode(sig_b64)
            filename = f"sig_user_{user_id}.png"
            filepath = os.path.join(signatures_dir, filename)
            
            with open(filepath, "wb") as f:
                f.write(image_data)
                
            signature_path = f"/uploads/signatures/{filename}"
        except Exception as e:
            raise ValidationFailedError("서명 이미지 저장 중 오류가 발생했습니다.")
            
        user = User(
            id=user_id,
            username=username,
            password_hash=hash_password(password),
            name=name,
            role=role,
            department=department,
            approval="pending",
            signature_path=signature_path,
        )
        self.users.add(user)
        self.audit.record(
            user_id=None, action="signup", target_type="user", target_id=user.id
        )
        self.db.commit()
        return user

    def list_accounts(self, page: Page) -> list[User]:
        return self.users.list_ordered()

    def get_approval_history(self, user_id: str) -> list[ApprovalHistory]:
        from sqlalchemy import select
        stmt = select(ApprovalHistory).where(ApprovalHistory.user_id == user_id).order_by(ApprovalHistory.created_at.desc())
        return list(self.db.scalars(stmt))

    def set_approval(self, user_id: str, approval: str, admin_id: str, rejection_reason: str | None = None) -> User:
        """관리자 승인/거부 — 상태 전이 + 감사로그 + 히스토리 보존."""
        if approval not in ("approved", "rejected", "pending"):
            raise ConflictError("approval 은 approved | rejected | pending 만 허용됩니다.")
        
        user = self.users.get(user_id)
        if not user:
            raise NotFoundError("계정을 찾을 수 없습니다.")

        if approval == "rejected":
            if not rejection_reason or not rejection_reason.strip():
                raise ValidationFailedError("거부 사유를 입력해주세요.")
            user.rejection_reason = rejection_reason.strip()
            user.rejected_by = admin_id
            user.rejected_at = utcnow()
        elif approval == "approved":
            user.rejection_reason = None
            user.approved_by = admin_id
            user.approved_at = utcnow()

        old_status = user.approval
        user.approval = approval
        
        # ApprovalHistory 추가
        history = ApprovalHistory(
            user_id=user.id,
            admin_id=admin_id,
            actor_type="ADMIN",
            old_status=old_status,
            new_status=approval,
            reason=user.rejection_reason,
            created_at=utcnow()
        )
        self.db.add(history)

        self.audit.record(
            user_id=admin_id,
            action="ADMIN_APPROVAL_CHANGE",
            target_type="user",
            target_id=user.id,
            payload={
                "old_status": old_status,
                "new_status": approval,
                "reason": user.rejection_reason
            }
        )
        self.db.commit()
        return user

    def get(self, user_id: str) -> User | None:
        return self.users.get(user_id)
