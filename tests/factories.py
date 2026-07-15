"""테스트용 엔티티 팩토리 — 고유 ID 로 격리된 행을 생성/커밋한다.

서비스가 내부에서 commit 하므로, 각 테스트는 공유 시드 데이터를 변형하지 않고
여기서 만든 throwaway 엔티티 위에서만 상태전이를 검증한다.
"""
from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from core.security import hash_password
from models.base import utcnow
from models.bed import Bed
from models.consultation import Consultation
from models.notification import Notification
from models.patient import Patient
from models.user import User


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def make_patient(db: Session, **over) -> Patient:
    p = Patient(
        id=_uid("p"),
        mrn=_uid("MRN").upper(),
        name=over.get("name", "테스트환자"),
        sex=over.get("sex", "M"),
        age=over.get("age", 60),
        diagnosis=over.get("diagnosis", "테스트 진단"),
        admitted_at=utcnow().isoformat(),
        attending=over.get("attending", "테스트의"),
        room=over.get("room", "TEST-1"),
        ai_risk_score=over.get("ai_risk_score", 0),
    )
    db.add(p)
    db.commit()
    return p


def make_bed(db: Session, *, state: str = "available", **over) -> Bed:
    b = Bed(
        id=_uid("bed"),
        zone=over.get("zone", "ward"),
        label=over.get("label", _uid("L")[:8]),
        state=state,
        patient_id=over.get("patient_id"),
        patient_name=over.get("patient_name"),
    )
    db.add(b)
    db.commit()
    return b


def make_user(db: Session, *, approval: str = "approved", **over) -> User:
    u = User(
        id=_uid("u"),
        username=over.get("username", _uid("user")),
        password_hash=hash_password(over.get("password", "Passw0rd!")),
        name=over.get("name", "테스트유저"),
        role=over.get("role", "nephrology"),
        department=over.get("department", "신장내과"),
        approval=approval,
    )
    db.add(u)
    db.commit()
    return u


def make_notification(db: Session, *, department: str = "nephrology", **over) -> Notification:
    n = Notification(
        id=_uid("noti"),
        department=department,
        severity=over.get("severity", "INFO"),
        title=over.get("title", "테스트 알림"),
        message=over.get("message", "메시지"),
        link=over.get("link"),
        tone=over.get("tone"),
        read=over.get("read", False),
    )
    db.add(n)
    db.commit()
    return n


def make_consultation(db: Session, *, status: str = "requested", **over) -> Consultation:
    c = Consultation(
        id=_uid("c"),
        kind=over.get("kind", "pathology"),
        patient_mrn=over.get("patient_mrn", _uid("MRN").upper()),
        patient_name=over.get("patient_name", "협진환자"),
        diagnosis=over.get("diagnosis", "IgA 신병증 의증"),
        key_labs=over.get("key_labs", "Cr 2.1"),
        reason=over.get("reason", "조직검사 판독 요청"),
        urgency=over.get("urgency", "routine"),
        status=status,
        requested_by=over.get("requested_by", "neph_hong"),
        requested_at=utcnow(),
        bed_label=over.get("bed_label"),
    )
    db.add(c)
    db.commit()
    return c
