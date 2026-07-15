"""병상 서비스 — 병상 트랜잭션의 오케스트레이터.

책임 분리
- 이 서비스: 트랜잭션 경계 관리 + 병상 잠금/상태변경 + 입원/감사/타임라인 조율.
- AdmissionService: 입원 행 생성/종료(단일 책임).
- TimelineService/AuditService: 부가 기록.

BED ASSIGN 트랜잭션:
  1. SELECT bed FOR UPDATE   (비관적 잠금)
  2. 상태(available) 확인
  3. 환자 확정(기존 or 신규 등록)
  4. 입원 중복 체크 + 입원 생성
  5. bed 상태 occupied 변경
  6. BED_CHANGE 타임라인 + 감사로그
  7. COMMIT

BED RELEASE 트랜잭션:
  1. bed lock → 2. 활성 입원 종료 → 3. bed 상태 cleaning + 환자 해제
  4. 타임라인 + 감사로그 → 5. COMMIT
"""
from sqlalchemy.orm import Session

from core.exceptions import ConflictError, NotFoundError
from core.query_optimizer import Page
from models.base import new_id
from models.bed import Bed
from models.patient import Patient
from repositories.bed_repository import BedRepository
from repositories.patient_repository import PatientRepository
from services.admission_service import AdmissionService
from services.audit_service import AuditService
from services.timeline_service import TimelineService


# 구역별 표시 라벨(집계/타임라인 메시지에 사용).
ZONE_LABEL = {"er": "응급실", "icu": "ICU", "ward": "일반병동", "isolation": "격리병동"}


class BedService:
    def __init__(self, db: Session):
        self.db = db
        self.beds = BedRepository(db)
        self.patients = PatientRepository(db)
        self.admissions = AdmissionService(db)
        self.timeline = TimelineService(db)
        self.audit = AuditService(db)

    # ---------------- 조회 ----------------
    def list_beds(self) -> list[Bed]:
        return self.beds.list_ordered()

    def summarize(self, beds: list[Bed]) -> list[dict]:
        """구역별 총/사용중/잔여 집계(KPI). 메모리 내 1패스 집계(추가 쿼리 없음)."""
        agg: dict[str, dict] = {}
        for bed in beds:
            z = agg.setdefault(
                bed.zone, {"zone": bed.zone, "total": 0, "occupied": 0, "available": 0}
            )
            z["total"] += 1
            if bed.state == "occupied":
                z["occupied"] += 1
            elif bed.state == "available":
                z["available"] += 1
        return list(agg.values())


    # ---------------- 병상 배정 트랜잭션 ----------------
    def assign_bed(
        self,
        *,
        bed_id: str,
        actor_id: str,
        actor_name: str,
        patient_id: str | None = None,
        patient_name: str | None = None,
        sex: str | None = None,
        age: int | None = None,
        diagnosis: str | None = None,
    ) -> tuple[Bed, "object"]:
        """병상 배정 — 단일 트랜잭션. 실패 시 전체 롤백."""
        try:
            # 1) 병상 비관적 잠금
            bed = self.beds.get_for_update(bed_id)
            if not bed:
                raise NotFoundError("병상을 찾을 수 없습니다.")
            # 2) 상태 확인 — available 만 배정 가능
            if bed.state != "available":
                raise ConflictError(
                    f"배정 불가 상태입니다(현재: {bed.state})."
                )

            # 3) 환자 확정 — 기존 환자 또는 신규 즉석 등록(응급 입실)
            patient = self._resolve_patient(
                patient_id, patient_name, sex, age, diagnosis
            )

            # 4) 입원 중복 체크 + 입원 생성(AdmissionService)
            admission = self.admissions.open(patient.id, bed.id)

            # 5) 병상 상태 변경
            bed.state = "occupied"
            bed.patient_id = patient.id
            bed.patient_name = patient.name
            self.db.flush()

            # 6) 타임라인(BED_CHANGE) + 감사로그
            self.timeline.add_event(
                patient_id=patient.id,
                event_type="BED_CHANGE",
                title=f"{ZONE_LABEL.get(bed.zone, bed.zone)} {bed.label} 병상 배정",
                severity="INFO",
                description=f"{patient.name} 환자 입실",
                source="BED_SYSTEM",
                actor=actor_name,
                payload={"bedId": bed.id, "admissionId": admission.id},
            )
            self.audit.record(
                user_id=actor_id,
                action="bed_assign",
                target_type="bed",
                target_id=bed.id,
                payload={"patientId": patient.id, "admissionId": admission.id},
            )

            # 7) COMMIT
            self.db.commit()
            self.db.refresh(bed)
            return bed, admission
        except Exception:
            self.db.rollback()
            raise

    # ---------------- 병상 해제 트랜잭션 ----------------
    def release_bed(
        self, *, bed_id: str, actor_id: str, actor_name: str
    ) -> tuple[Bed, "object"]:
        """병상 해제 — 입원 종료 + 병상 정리중 전환. 단일 트랜잭션."""
        try:
            # 1) 병상 잠금
            bed = self.beds.get_for_update(bed_id)
            if not bed:
                raise NotFoundError("병상을 찾을 수 없습니다.")
            if bed.state != "occupied":
                raise ConflictError(
                    f"사용중 병상만 해제할 수 있습니다(현재: {bed.state})."
                )

            released_patient_id = bed.patient_id

            # 2) 활성 입원 종료
            admission = self.admissions.close_by_bed(bed.id)

            # 3) 병상 상태 CLEANING + 환자 해제
            bed.state = "cleaning"
            bed.patient_id = None
            bed.patient_name = None
            self.db.flush()

            # 4) 타임라인 + 감사로그
            if released_patient_id:
                self.timeline.add_event(
                    patient_id=released_patient_id,
                    event_type="BED_CHANGE",
                    title=f"{ZONE_LABEL.get(bed.zone, bed.zone)} {bed.label} 병상 해제",
                    severity="INFO",
                    description="퇴실/전동 — 병상 정리중",
                    source="BED_SYSTEM",
                    actor=actor_name,
                    payload={"bedId": bed.id},
                )
            self.audit.record(
                user_id=actor_id,
                action="bed_release",
                target_type="bed",
                target_id=bed.id,
                payload={
                    "patientId": released_patient_id,
                    "admissionId": admission.id if admission else None,
                },
            )

            # 5) COMMIT
            self.db.commit()
            self.db.refresh(bed)
            return bed, admission
        except Exception:
            self.db.rollback()
            raise

    # ---------------- 내부 헬퍼 ----------------
    def _resolve_patient(
        self,
        patient_id: str | None,
        patient_name: str | None,
        sex: str | None,
        age: int | None,
        diagnosis: str | None,
    ) -> Patient:
        """기존 환자 조회 또는 신규 환자 즉석 등록(응급 입실)."""
        if patient_id:
            patient = self.patients.get(patient_id)
            if not patient:
                raise NotFoundError("환자를 찾을 수 없습니다.")
            return patient

        if not patient_name:
            raise ConflictError("patient_id 또는 신규 환자 정보(patient_name)가 필요합니다.")

        # 신규 환자 최소 정보 생성 — MRN 은 자동 부여.
        from models.base import utcnow

        patient = Patient(
            id=new_id("p"),
            mrn=f"ER-{new_id('')[1:9].upper()}",
            name=patient_name,
            sex=(sex or "M"),
            age=age or 0,
            diagnosis=diagnosis or "응급 입실 — 진단 미정",
            admitted_at=utcnow().isoformat(),
            attending="미지정",
            room="응급실",
            ai_risk_score=0,
        )
        return self.patients.add(patient)
