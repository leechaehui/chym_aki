"""초기 시드 데이터.

프론트엔드 mock 과 동일한 계정/환자/병상/응급환자를 적재한다.
→ 프론트가 mock 서비스를 실제 API 로 교체해도 화면이 그대로 동작한다.
멱등: 데이터가 이미 있으면 재적재하지 않는다.
"""
from sqlalchemy import select

from core.database import SessionLocal
from core.security import hash_password
from db.seed_clinical import seed_clinical
from models.bed import Bed
from models.patient import (
    Patient,
    PatientLab,
    PatientTrendPoint,
    PatientUrinePoint,
)
from models.user import User


def _lab(seq, key, label, value, unit, low, high):
    flag = "normal"
    if high is not None and value > high:
        flag = "high"
    elif low is not None and value < low:
        flag = "low"
    return PatientLab(
        seq=seq, key=key, label=label, value=value, unit=unit,
        ref_low=low, ref_high=high, flag=flag,
    )


# 데모 계정 — 프론트 mockUsers 와 동일(비밀번호는 해시 저장).
USERS = [
    ("u-admin", "admin", "Admin2026!", "박지영", "admin", "시스템관리팀", "approved"),
    ("u-neph", "neph_hong", "Neph2026!", "홍민준", "nephrology", "신장내과", "approved"),
    ("u-path", "path_lee", "Path2026!", "이수진", "pathology", "병리과", "approved"),
    ("u-pend-1", "neph_park", "Temp2026!", "박하은", "nephrology", "신장내과", "pending"),
    ("u-rej-1", "test_user", "Temp2026!", "테스트", "pathology", "병리과", "rejected"),
]


def _patients() -> list[Patient]:
    p1 = Patient(
        id="p-001", mrn="AKI-100231", name="정민호", sex="M", age=64,
        diagnosis="급성 신손상 (AKI stage 2)", admitted_at="2026-06-09T10:20:00",
        attending="홍민준", room="본관 7동 712호", ai_risk_score=78,
        labs=[
            _lab(0, "cr", "Creatinine", 2.8, "mg/dL", 0.7, 1.3),
            _lab(1, "egfr", "eGFR", 26, "mL/min", 60, None),
            _lab(2, "bun", "BUN", 48, "mg/dL", 8, 20),
            _lab(3, "na", "Na", 134, "mmol/L", 135, 145),
            _lab(4, "k", "K", 5.4, "mmol/L", 3.5, 5.1),
            _lab(5, "hco3", "HCO3", 18, "mmol/L", 22, 29),
            _lab(6, "alb", "Albumin", 3.2, "g/dL", 3.5, 5.2),
            _lab(7, "upcr", "UPCR", 2.1, "g/g", 0, 0.2),
        ],
        trend=[
            PatientTrendPoint(date="2026-06-09", creatinine=1.6, egfr=48, bun=28),
            PatientTrendPoint(date="2026-06-10", creatinine=2.0, egfr=38, bun=34),
            PatientTrendPoint(date="2026-06-11", creatinine=2.4, egfr=31, bun=41),
            PatientTrendPoint(date="2026-06-12", creatinine=2.7, egfr=28, bun=45),
            PatientTrendPoint(date="2026-06-13", creatinine=2.8, egfr=26, bun=48),
        ],
        urine_output=[
            PatientUrinePoint(date="2026-06-09", value=0.9),
            PatientUrinePoint(date="2026-06-10", value=0.7),
            PatientUrinePoint(date="2026-06-11", value=0.5),
            PatientUrinePoint(date="2026-06-12", value=0.4),
            PatientUrinePoint(date="2026-06-13", value=0.35),
        ],
    )
    p2 = Patient(
        id="p-002", mrn="AKI-100244", name="한서영", sex="F", age=52,
        diagnosis="사구체신염 의증, 단백뇨", admitted_at="2026-06-11T08:00:00",
        attending="홍민준", room="본관 7동 715호", ai_risk_score=54,
        labs=[
            _lab(0, "cr", "Creatinine", 1.7, "mg/dL", 0.6, 1.1),
            _lab(1, "egfr", "eGFR", 42, "mL/min", 60, None),
            _lab(2, "bun", "BUN", 26, "mg/dL", 8, 20),
            _lab(3, "na", "Na", 138, "mmol/L", 135, 145),
            _lab(4, "k", "K", 4.2, "mmol/L", 3.5, 5.1),
            _lab(5, "hco3", "HCO3", 23, "mmol/L", 22, 29),
            _lab(6, "alb", "Albumin", 2.8, "g/dL", 3.5, 5.2),
            _lab(7, "upcr", "UPCR", 3.6, "g/g", 0, 0.2),
        ],
        trend=[
            PatientTrendPoint(date="2026-06-11", creatinine=1.5, egfr=48, bun=22),
            PatientTrendPoint(date="2026-06-12", creatinine=1.6, egfr=45, bun=24),
            PatientTrendPoint(date="2026-06-13", creatinine=1.7, egfr=42, bun=26),
        ],
        urine_output=[
            PatientUrinePoint(date="2026-06-11", value=1.1),
            PatientUrinePoint(date="2026-06-12", value=1.0),
            PatientUrinePoint(date="2026-06-13", value=0.95),
        ],
    )
    p3 = Patient(
        id="p-003", mrn="AKI-100258", name="오태윤", sex="M", age=71,
        diagnosis="만성 신질환 급성 악화 (CKD on AKI)", admitted_at="2026-06-07T15:45:00",
        attending="홍민준", room="본관 7동 720호", ai_risk_score=91,
        labs=[
            _lab(0, "cr", "Creatinine", 4.3, "mg/dL", 0.7, 1.3),
            _lab(1, "egfr", "eGFR", 13, "mL/min", 60, None),
            _lab(2, "bun", "BUN", 62, "mg/dL", 8, 20),
            _lab(3, "na", "Na", 131, "mmol/L", 135, 145),
            _lab(4, "k", "K", 6.2, "mmol/L", 3.5, 5.1),
            _lab(5, "hco3", "HCO3", 16, "mmol/L", 22, 29),
            _lab(6, "alb", "Albumin", 3.0, "g/dL", 3.5, 5.2),
            _lab(7, "upcr", "UPCR", 1.4, "g/g", 0, 0.2),
        ],
        trend=[
            PatientTrendPoint(date="2026-06-07", creatinine=3.1, egfr=22, bun=50),
            PatientTrendPoint(date="2026-06-09", creatinine=3.4, egfr=20, bun=55),
            PatientTrendPoint(date="2026-06-11", creatinine=3.7, egfr=18, bun=59),
            PatientTrendPoint(date="2026-06-13", creatinine=4.3, egfr=13, bun=62),
        ],
        urine_output=[
            PatientUrinePoint(date="2026-06-07", value=0.6),
            PatientUrinePoint(date="2026-06-09", value=0.45),
            PatientUrinePoint(date="2026-06-11", value=0.3),
            PatientUrinePoint(date="2026-06-13", value=0.2),
        ],
    )
    p4 = Patient(
        id="p-004", mrn="AKI-100269", name="배은경", sex="F", age=47,
        diagnosis="조영제 유발 신병증, 회복기", admitted_at="2026-06-05T09:30:00",
        attending="홍민준", room="본관 7동 709호", ai_risk_score=22,
        labs=[
            _lab(0, "cr", "Creatinine", 1.1, "mg/dL", 0.6, 1.1),
            _lab(1, "egfr", "eGFR", 68, "mL/min", 60, None),
            _lab(2, "bun", "BUN", 18, "mg/dL", 8, 20),
            _lab(3, "na", "Na", 140, "mmol/L", 135, 145),
            _lab(4, "k", "K", 4.0, "mmol/L", 3.5, 5.1),
            _lab(5, "hco3", "HCO3", 24, "mmol/L", 22, 29),
            _lab(6, "alb", "Albumin", 3.8, "g/dL", 3.5, 5.2),
            _lab(7, "upcr", "UPCR", 0.3, "g/g", 0, 0.2),
        ],
        trend=[
            PatientTrendPoint(date="2026-06-05", creatinine=1.9, egfr=38, bun=30),
            PatientTrendPoint(date="2026-06-08", creatinine=1.5, egfr=50, bun=24),
            PatientTrendPoint(date="2026-06-11", creatinine=1.3, egfr=60, bun=20),
            PatientTrendPoint(date="2026-06-13", creatinine=1.1, egfr=68, bun=18),
        ],
        urine_output=[
            PatientUrinePoint(date="2026-06-05", value=0.8),
            PatientUrinePoint(date="2026-06-08", value=1.2),
            PatientUrinePoint(date="2026-06-11", value=1.4),
            PatientUrinePoint(date="2026-06-13", value=1.5),
        ],
    )
    p5 = Patient(
        id="p-005", mrn="AKI-100277", name="신재원", sex="M", age=58,
        diagnosis="패혈증 동반 급성 신손상 의증", admitted_at="2026-06-15T22:10:00",
        attending="홍민준", room="본관 7동 718호", ai_risk_score=81,
        labs=[
            _lab(0, "cr", "Creatinine", 1.6, "mg/dL", 0.7, 1.3),
            _lab(1, "egfr", "eGFR", 47, "mL/min", 60, None),
            _lab(2, "bun", "BUN", 30, "mg/dL", 8, 20),
            _lab(3, "na", "Na", 137, "mmol/L", 135, 145),
            _lab(4, "k", "K", 4.8, "mmol/L", 3.5, 5.1),
            _lab(5, "hco3", "HCO3", 20, "mmol/L", 22, 29),
            _lab(6, "alb", "Albumin", 3.1, "g/dL", 3.5, 5.2),
            _lab(7, "upcr", "UPCR", 1.0, "g/g", 0, 0.2),
        ],
        trend=[
            PatientTrendPoint(date="2026-06-15", creatinine=1.1, egfr=72, bun=20),
            PatientTrendPoint(date="2026-06-16", creatinine=1.6, egfr=47, bun=30),
        ],
        urine_output=[
            PatientUrinePoint(date="2026-06-15", value=0.8),
            PatientUrinePoint(date="2026-06-16", value=0.45),
        ],
    )
    return [p1, p2, p3, p4, p5]


def _beds() -> list[Bed]:
    """프론트 mockBeds 와 동일한 구역/상태/이름 구성."""
    spec = [
        # (zone, prefix, [states], [occupied names])
        ("er", "ER",
         ["occupied", "occupied", "occupied", "available", "occupied", "cleaning", "available", "occupied"],
         ["정민호", "강지호", "윤소민", None, "서준영", None, None, "노현서"]),
        ("icu", "ICU",
         ["occupied", "occupied", "occupied", "occupied", "occupied", "reserved"],
         ["오태윤", "임도현", "조은우", "백유진", "남지원", None]),
        ("ward", "W7",
         ["occupied", "occupied", "available", "occupied", "available", "available", "occupied", "reserved"],
         ["한서영", "정민호", None, "오태윤", None, None, "문세아", None]),
        ("isolation", "ISO",
         ["occupied", "available", "available", "cleaning"],
         ["배승민", None, None, None]),
    ]
    beds: list[Bed] = []
    for zone, prefix, states, names in spec:
        for i, state in enumerate(states):
            beds.append(
                Bed(
                    id=f"{prefix}-{i + 1}",
                    zone=zone,
                    label=f"{prefix}-{str(i + 1).zfill(2)}",
                    state=state,
                    patient_name=names[i] if state == "occupied" else None,
                )
            )
    return beds




def seed() -> None:
    """멱등 시드. users 테이블이 비어있을 때만 전체 적재."""
    db = SessionLocal()
    try:
        already = db.execute(select(User.id).limit(1)).first()
        if already:
            return

        for uid, username, pw, name, role, dept, approval in USERS:
            user = User(
                id=uid, username=username, password_hash=hash_password(pw),
                name=name, role=role, department=dept, approval=approval,
            )
            db.add(user)
            
            # 초기 데이터용 승인/거부 이력 추가
            if approval in ("approved", "rejected"):
                from models.approval_history import ApprovalHistory
                from models.base import utcnow
                
                history = ApprovalHistory(
                    user_id=uid,
                    admin_id="u-admin",
                    actor_type="ADMIN",
                    old_status="pending",
                    new_status=approval,
                    reason="테스트 계정 초기 세팅" if approval == "rejected" else None,
                    created_at=utcnow()
                )
                db.add(history)
        for p in _patients():
            db.add(p)
        for b in _beds():
            db.add(b)
        seed_clinical(db)

        db.commit()
        print("[seed] 초기 데이터 적재 완료 (users/patients/beds/emergency/pathology/bed_detail/notification)")
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    from core.database import init_db

    init_db()
    seed()
