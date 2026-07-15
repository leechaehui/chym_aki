"""신장내과 서비스 — AKI 분석 + 타임라인(읽기 전용) 소비.

책임
- AKI 예측(예측기 Factory 사용) — 원시 피처 또는 환자 데이터 기반.
- 환자 분석 시 ai_risk_score 갱신 + 고위험이면 AI_ALERT 타임라인 기록(트랜잭션).
- 타임라인은 '읽기'만 한다(AKI 도메인의 쓰기는 AI_ALERT 자동기록에 한정).
"""
from sqlalchemy.orm import Session

from ai_draft.aki_model import get_aki_predictor
from core.exceptions import NotFoundError
from core.query_optimizer import Page
from models.patient import Patient
from repositories.patient_repository import PatientRepository
from services.audit_service import AuditService
from services.timeline_service import TimelineService


def _lab(patient: Patient, key: str):
    for lab in patient.labs:
        if lab.key == key:
            return lab.value
    return None


def build_features_from_patient(patient: Patient) -> dict:
    """환자 검사/추이/소변량 → AKI 피처 dict 로 변환.

    baseline 은 추이의 첫 creatinine, max 는 마지막값으로 근사한다.
    """
    trend = list(patient.trend)
    urine = list(patient.urine_output)

    cr_now = _lab(patient, "cr")
    egfr = _lab(patient, "egfr")
    k = _lab(patient, "k")
    bun = _lab(patient, "bun")
    hco3 = _lab(patient, "hco3")

    baseline = trend[0].creatinine if trend else cr_now
    cr_max = max([t.creatinine for t in trend] + ([cr_now] if cr_now else [])) if (
        trend or cr_now
    ) else None
    delta = (cr_max - baseline) if (cr_max and baseline) else None
    uo = urine[-1].value if urine else None

    return {
        "creatinine_max": cr_max,
        "creatinine_min": baseline,
        "baseline_creatinine": baseline,
        "creatinine_delta": delta,
        "egfr": egfr,
        "bun_max": bun,
        "potassium_max": k,
        "bicarbonate_min": hco3,
        "urine_ml_kg_hr": uo,
        "oliguria_flag": 1 if (uo is not None and uo < 0.5) else 0,
        "age": patient.age,
    }


class NephrologyService:
    def __init__(self, db: Session):
        self.db = db
        self.patients = PatientRepository(db)
        self.predictor = get_aki_predictor()
        self.timeline = TimelineService(db)
        self.audit = AuditService(db)

    def analyze_features(self, features: dict) -> dict:
        """원시 피처 기반 AKI 추론(영속화 없음 — 순수 분석).

        하이브리드 예측기가 피처 커버리지로 모델/규칙을 라우팅한다.
        """
        return self.predictor.predict(features)

    def analyze_vector(self, features: dict) -> dict:
        """표준화된 48h 전체 피처 벡터(연구/배치)에 학습 모델을 직접 적용.

        모델 미탑재(규칙 기반만 가능) 환경에서는 일반 라우팅으로 폴백한다.
        """
        if hasattr(self.predictor, "predict_with_model"):
            return self.predictor.predict_with_model(features)
        return self.predictor.predict(features)

    def analyze_patient(self, patient_id: str, *, actor_id: str) -> dict:
        """환자 기반 AKI 분석 + ai_risk_score 갱신 + 고위험 알림(트랜잭션)."""
        try:
            patient = self.patients.get_with_details(patient_id)
            if not patient:
                raise NotFoundError("환자를 찾을 수 없습니다.")

            features = build_features_from_patient(patient)
            result = self.predictor.predict(features)

            # ai_risk_score 갱신
            patient.ai_risk_score = result["risk_score"]
            self.db.flush()

            # 고위험(Stage2+3)이면 AI_ALERT 타임라인 기록(EMR 통합)
            if result["stage_num"] >= 3:
                self.timeline.add_event(
                    patient_id=patient.id,
                    event_type="AI_ALERT",
                    title=f"AKI 고위험 경보 ({result['stage']})",
                    severity="CRITICAL",
                    description="; ".join(result["rationale"][:3]),
                    source="AI_SYSTEM",
                    actor="AKI Model",
                    payload={"riskScore": result["risk_score"]},
                )
            self.audit.record(
                user_id=actor_id,
                action="aki_analyze",
                target_type="patient",
                target_id=patient.id,
                payload={"riskScore": result["risk_score"], "source": result["source"]},
            )
            self.db.commit()
            return result
        except Exception:
            self.db.rollback()
            raise

    def read_timeline(
        self,
        patient_id: str,
        page: Page,
        severity: str | None = None,
        event_type: str | None = None,
    ):
        """환자 타임라인 읽기(읽기 전용) — 신장내과 상태 조회."""
        if not self.patients.get(patient_id):
            raise NotFoundError("환자를 찾을 수 없습니다.")
        return self.timeline.list_for_patient(patient_id, page, severity, event_type)
