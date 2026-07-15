"""협진 서비스.

책임: 협진 생성/상태전이(접수·회신)와 타임라인 단계 기록.
- 모든 상태변경은 트랜잭션 + 감사로그.
- 협진 단계 변화는 환자 타임라인(CONSULTATION 이벤트)으로도 흘려보낸다(EMR 통합).
"""
import json

from sqlalchemy import text
from sqlalchemy.orm import Session

from core.event_bus import ALERT_EVENT, event_bus
from core.exceptions import NotFoundError
from core.query_optimizer import Page
from models.base import new_id, utcnow
from models.consultation import ConsultEvent, Consultation
from repositories.consultation_repository import ConsultationRepository
from repositories.patient_repository import PatientRepository
from services.audit_service import AuditService
from services.timeline_service import TimelineService

# 협진 종류별 요청/회신 부서 라벨(타임라인 actor 표기).
REQUESTER_DEPT = {"pathology": "신장내과", "nephrology": "응급의학과"}
RESPONDER_DEPT = {"pathology": "병리과", "nephrology": "신장내과"}
# 회신 완료 알림을 받을 '요청 부서' 역할(role) — WS/notification 라우팅 기준.
REQUESTER_ROLE = {"pathology": "nephrology", "nephrology": "emergency"}


class ConsultationService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = ConsultationRepository(db)
        self.patients = PatientRepository(db)
        self.audit = AuditService(db)
        self.timeline = TimelineService(db)

    def list(self, page: Page, kind: str | None = None) -> list[Consultation]:
        return self.repo.list_with_timeline(page, kind)

    def get(self, consult_id: str) -> Consultation:
        consult = self.repo.get_with_timeline(consult_id)
        if not consult:
            raise NotFoundError("협진을 찾을 수 없습니다.")
        return consult

    def canonical_patient_name(self, mrn: str) -> str | None:
        """MRN 으로 현재 환자 명단(chym.patients)의 표시명을 조회.

        협진에는 요청 시점의 이름이 '냉동'되어 저장된다. 이후 환자 명단의 표시명이
        바뀌면(재-Setup·이름 정정 등) 병리 판독목록이 대시보드/모니터링과 어긋난다.
        읽기 시점에 명단의 현재 이름으로 해소해 전 화면 표시명을 일치시킨다.
        명단에 없는(시드/큐레이션) 협진은 None → 저장된 이름을 그대로 쓴다."""
        p = self.patients.get_by_mrn(mrn)
        return p.name if p else None

    def request(self, *, actor_id: str, data: dict) -> Consultation:
        """협진 요청 생성 — requested 상태 + 첫 타임라인 단계."""
        try:
            kind = data.get("kind", "pathology")
            now = utcnow().isoformat()
            consult = Consultation(
                id=new_id("c"),
                kind=kind,
                patient_mrn=data["patient_mrn"],
                patient_name=data["patient_name"],
                diagnosis=data["diagnosis"],
                key_labs=data.get("key_labs", ""),
                reason=data.get("reason", ""),
                urgency=data.get("urgency", "routine"),
                status="requested",
                requested_by=data["requested_by"],
                requested_by_user_id=actor_id,   # 정밀 타겟팅용 — 회신 알림을 요청자 계정에만.
                requested_at=utcnow(),
                bed_label=data.get("bed_label"),
            )
            label = "응급 협진 요청" if kind == "nephrology" else "협진 요청"
            consult.timeline.append(
                ConsultEvent(
                    id=new_id("t"),
                    stage="requested",
                    label=label,
                    at=now,
                    actor=f"{data['requested_by']} ({REQUESTER_DEPT[kind]})",
                )
            )
            self.repo.add(consult)
            self._mirror_to_patient_timeline(
                consult, title=label, severity=self._severity(consult.urgency)
            )
            self.audit.record(
                user_id=actor_id,
                action="consult_request",
                target_type="consultation",
                target_id=consult.id,
                payload={"kind": kind, "urgency": consult.urgency},
            )
            self.db.commit()
            return self.get(consult.id)
        except Exception:
            self.db.rollback()
            raise

    def accept(self, consult_id: str, *, actor_id: str, actor: str) -> Consultation:
        """협진 접수 — requested → in_progress."""
        try:
            consult = self.get(consult_id)
            if consult.status != "requested":
                return consult
            now = utcnow().isoformat()
            consult.status = "in_progress"
            consult.timeline.append(
                ConsultEvent(
                    id=new_id("t"),
                    stage="received",
                    label=f"{RESPONDER_DEPT[consult.kind]} 접수",
                    at=now,
                    actor=f"{actor} ({RESPONDER_DEPT[consult.kind]})",
                )
            )
            self.db.flush()
            self.audit.record(
                user_id=actor_id,
                action="consult_accept",
                target_type="consultation",
                target_id=consult.id,
            )
            self.db.commit()
            return self.get(consult.id)
        except Exception:
            self.db.rollback()
            raise

    def reply(self, consult_id: str, *, actor_id: str, reply: dict) -> Consultation:
        """협진 회신 — status replied + reply 저장 + 타임라인."""
        try:
            consult = self.get(consult_id)
            now = utcnow().isoformat()
            reply_payload = {**reply, "repliedAt": now}
            consult.status = "replied"
            consult.reply_json = json.dumps(reply_payload, ensure_ascii=False)
            consult.timeline.append(
                ConsultEvent(
                    id=new_id("t"),
                    stage="replied",
                    label="협진 회신",
                    at=now,
                    actor=f"{reply['author']} ({RESPONDER_DEPT[consult.kind]})",
                )
            )
            self.db.flush()
            self._mirror_to_patient_timeline(
                consult, title="협진 회신 도착", severity="ACTION_REQUIRED"
            )
            self.audit.record(
                user_id=actor_id,
                action="consult_reply",
                target_type="consultation",
                target_id=consult.id,
            )
            self.db.commit()
            # 회신(=병리 판독 완료) 을 요청한 부서(신장내과)에 자동 통지.
            self._notify_requester(consult_id)
            return self.get(consult.id)
        except Exception:
            self.db.rollback()
            raise

    def _notify_requester(self, consult_id: str) -> None:
        """협진 회신 완료 → 요청 부서에 통지: chym.notifications 영속 + ALERT_EVENT WS 라이브 푸시.

        '요청한 사람'(consult.requested_by)을 문구에 명시해, 요청한 신장내과 전문의가
        자기 요청 건임을 바로 알 수 있게 한다. 라우팅 자체는 부서(role) 단위."""
        consult = self.repo.get_with_timeline(consult_id)
        if consult is None:
            return
        dept = REQUESTER_ROLE.get(consult.kind)
        if not dept:
            return
        name = self.canonical_patient_name(consult.patient_mrn) or consult.patient_name
        link = f"/nephrology?patient={consult.patient_mrn}"
        title = f"병리 판독 완료 · {name}"
        message = (
            f"{consult.requested_by}님이 요청한 {name}({consult.patient_mrn}) "
            f"병리 판독이 완료되었습니다. 소견·진단을 확인하세요."
        )
        # 요청자 계정을 알면 그 계정에만 정밀 타겟팅, 모르면(구건) 부서 전체로 폴백.
        target_user_id = consult.requested_by_user_id
        try:
            self.db.execute(text("""
                INSERT INTO chym.notifications (id, department, target_user_id, severity, title, message, link, read)
                VALUES (:id, :dept, :target, 'ACTION_REQUIRED', :title, :message, :link, false)
            """), {"id": new_id("noti-consult"), "dept": dept, "target": target_user_id,
                   "title": title, "message": message, "link": link})
            self.db.commit()
        except Exception:
            self.db.rollback()  # 통지 실패가 회신 자체를 되돌리지 않게 격리
            return
        event_bus.publish(ALERT_EVENT, {
            "eventType": ALERT_EVENT, "eventId": new_id("alertevt"), "alertId": None,
            "patientId": consult.patient_mrn, "type": "CONSULT_REPLY", "severity": "info",
            "priority": 30, "dedupKey": f"consult-reply-{consult.id}",
            "title": title, "message": message, "link": link,
            "department": dept, "targetUserId": target_user_id, "status": "active", "createdAt": "",
        })

    # ---------------- 내부 ----------------
    def _severity(self, urgency: str) -> str:
        return {
            "routine": "INFO",
            "urgent": "WARNING",
            "emergency": "CRITICAL",
        }.get(urgency, "INFO")

    def _mirror_to_patient_timeline(
        self, consult: Consultation, *, title: str, severity: str
    ) -> None:
        """협진 이벤트를 환자 타임라인(CONSULTATION)으로 미러링.

        MRN 으로 환자를 찾을 수 있을 때만 기록(외부 환자는 스킵).
        """
        patient = self.patients.get_by_mrn(consult.patient_mrn)
        if not patient:
            return
        self.timeline.add_event(
            patient_id=patient.id,
            event_type="CONSULTATION",
            title=title,
            severity=severity,
            description=f"{consult.diagnosis} — {consult.reason}"[:200],
            source="CONSULTATION",
            actor=consult.requested_by,
            payload={"consultId": consult.id, "kind": consult.kind},
        )
