"""AI Draft 오케스트레이터 (작업지시서 1·8).

파이프라인:
  Audio → STT → Transcript → SOAP → Problem List → CDSS Risk → Timeline Event

이 서비스는 '조율'만 한다(SRP). 실제 로직은 전용 서비스에 위임:
  SttService / SoapService / ProblemListService / CdssRiskService
  + validator(soap/ap_evidence/risk) 로 안전성 검증 후에만 영속화.

안전장치
- SOAP/CDSS 검증 실패 시 ValidationFailedError → 결과를 저장하지 않는다(hallucination 차단).
- 위험 HIGH → AI_ALERT(CRITICAL) 타임라인 + nephrology trigger(6.3).
"""
import json

from sqlalchemy.orm import Session

from ai_draft.aki_model import get_aki_predictor
from core.exceptions import NotFoundError, ValidationFailedError
from core.query_optimizer import Page
from models.ai_draft import AiDraftNote
from models.base import new_id
from nlp.factory import create_nlp_strategy
from repositories.ai_draft_repository import AiDraftRepository
from repositories.patient_repository import PatientRepository
from services.audit_service import AuditService
from services.cdss_risk_service import CdssRiskService
from services.nephrology_service import build_features_from_patient
from services.problem_list_service import ProblemListService
from services.soap_service import SoapService
from services.stt_service import SttService
from services.timeline_service import TimelineService
from validator.soap_validator import run_all as run_soap_validation


class AiDraftService:
    def __init__(self, db: Session):
        self.db = db
        self.patients = PatientRepository(db)
        self.notes = AiDraftRepository(db)
        self.audit = AuditService(db)
        self.timeline = TimelineService(db)
        self.nlp = create_nlp_strategy()
        self.predictor = get_aki_predictor()
        self.soap_service = SoapService()
        self.problem_service = ProblemListService()
        self.cdss = CdssRiskService()

    # ---------------- STT ----------------
    def transcribe(self, audio_bytes: bytes | None, fallback_text: str = "") -> dict:
        """오디오 → 전사(STT 서비스 위임)."""
        return SttService().transcribe(audio_bytes, fallback_text)

    # ---------------- 파이프라인(순수 계산, 영속화 없음) ----------------
    def run_pipeline(self, patient, transcript: str) -> dict:
        """전사 → SOAP → Problem List → CDSS Risk → 검증. 결과 dict 반환(저장 X)."""
        symptoms = self.nlp.extract_symptoms(transcript)
        aki_result = self.predictor.predict(build_features_from_patient(patient))
        soap = self.soap_service.generate(patient, transcript, symptoms, aki_result)
        problems = self.problem_service.derive(soap)
        risk = self.cdss.score(patient, soap, problems, aki_result)
        validation = run_soap_validation(soap, risk)
        return {
            "transcript": transcript,
            "symptoms": symptoms,
            "soap": soap,
            "problem_list": problems,
            "risk": risk,
            "validation": validation,
        }

    # ---------------- 생성 + 영속화(트랜잭션) ----------------
    def generate_draft(
        self,
        *,
        patient_id: str,
        transcript: str,
        actor_id: str,
        audio_bytes: bytes | None = None,
    ) -> AiDraftNote:
        """전체 파이프라인 실행 → 검증 통과 시 저장 + 타임라인 이벤트(단일 트랜잭션)."""
        try:
            patient = self.patients.get_with_details(patient_id)
            if not patient:
                raise NotFoundError("환자를 찾을 수 없습니다.")

            # 1) STT (오디오가 있으면 전사로 대체)
            if audio_bytes is not None:
                transcript = SttService().transcribe(audio_bytes, transcript)["transcript"]

            # 2~6) 파이프라인
            result = self.run_pipeline(patient, transcript)

            # 7) 안전 검증 — 실패 시 저장 차단
            if not result["validation"]["passed"]:
                raise ValidationFailedError(
                    "SOAP/CDSS 검증 실패 — 초안을 저장하지 않습니다.",
                    report=result["validation"],
                )

            # 8) 영속화(구조화 결과를 draft_text 에 저장)
            note = AiDraftNote(
                id=new_id("draft"),
                patient_id=patient.id,
                transcript=transcript,
                symptoms_json=json.dumps(result["symptoms"], ensure_ascii=False),
                draft_text=json.dumps(
                    {
                        "soap": result["soap"],
                        "problem_list": result["problem_list"],
                        "risk": result["risk"],
                        "validation": result["validation"],
                    },
                    ensure_ascii=False,
                ),
                status="draft",
            )
            self.notes.add(note)

            # 9) 타임라인 이벤트(ai_draft_note) — EMR 통합
            risk = result["risk"]
            self.timeline.add_event(
                patient_id=patient.id,
                event_type="ai_draft_note",
                title=f"AI 진료초안 생성 (위험 {risk['tier']})",
                severity="INFO",
                description=f"위험점수 {risk['risk_score']} · 문제 {len(result['problem_list'])}건",
                source="AI_SYSTEM",
                actor="AI Draft",
                payload={"draftId": note.id, "riskScore": risk["risk_score"], "tier": risk["tier"]},
            )

            # 10) HIGH → AI_ALERT(CRITICAL) + nephrology trigger
            if risk["nephrology_trigger"]:
                self.timeline.add_event(
                    patient_id=patient.id,
                    event_type="AI_ALERT",
                    title=f"CDSS 고위험 경보 (risk {risk['risk_score']})",
                    severity="CRITICAL",
                    description=risk["explanation"][:200],
                    source="AI_SYSTEM",
                    actor="CDSS",
                    payload={"riskScore": risk["risk_score"], "nephrologyTrigger": True},
                )
                patient.ai_risk_score = int(round(risk["risk_score"] * 100))

            # 11) 감사로그 + COMMIT
            self.audit.record(
                user_id=actor_id,
                action="ai_draft_generate",
                target_type="ai_draft_note",
                target_id=note.id,
                payload={"patientId": patient.id, "riskTier": risk["tier"]},
            )
            self.db.commit()
            self.db.refresh(note)
            return note
        except Exception:
            self.db.rollback()
            raise

    def list_for_patient(self, patient_id: str, page: Page) -> list[AiDraftNote]:
        return self.notes.list_for_patient(patient_id, page)

    @staticmethod
    def to_out(note: AiDraftNote) -> dict:
        """ORM 노트 → API 출력 dict(구조화 결과 역직렬화)."""
        draft = json.loads(note.draft_text or "{}")
        return {
            "id": note.id,
            "patient_id": note.patient_id,
            "transcript": note.transcript,
            "symptoms": json.loads(note.symptoms_json or "{}"),
            "soap": draft.get("soap", {}),
            "problem_list": draft.get("problem_list", []),
            "risk": draft.get("risk", {}),
            "validation": draft.get("validation", {}),
            "status": note.status,
            "created_at": note.created_at.isoformat() if note.created_at else None,
        }
