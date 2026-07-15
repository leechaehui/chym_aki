"""신장내과 API — AKI 분석 + 타임라인(읽기 전용)."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.serializers import timeline_to_out
from core.deps import get_current_user, get_db, require_roles
from core.query_optimizer import Page
from models.user import User
from schemas.nephrology import (
    AiDraftResultOut,
    AkiAnalysisResult,
    AkiFeatureInput,
    CareunitStatOut,
    IcuAkiPatientOut,
    IcuMonitorSummaryOut,
    IcuPatientSearchOut,
    IcuPatientSummaryOut,
    PatientShapOut,
    PatientTrendsOut,
    PatientValidationOut,
    RiskHistoryOut,
    RrtAssessmentOut,
    VoiceDraftRequest,
)
from schemas.timeline import TimelineEventOut
from services.icu_draft_service import IcuDraftService
from services.icu_monitor_service import IcuMonitorService, is_available, mark_patient_viewed
from services.icu_patient_detail_service import IcuPatientDetailService
from services.nephrology_service import NephrologyService

router = APIRouter(prefix="/nephrology", tags=["nephrology"])


@router.post("/aki/analyze", response_model=AkiAnalysisResult)
def analyze_features(
    body: AkiFeatureInput,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """원시 피처 기반 AKI 추론(영속화 없음)."""
    result = NephrologyService(db).analyze_features(body.model_dump())
    return AkiAnalysisResult(**result)


@router.post("/aki/predict-vector", response_model=AkiAnalysisResult)
def predict_vector(
    features: dict,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """표준화된 48h 전체 피처 벡터(35개)에 학습 모델을 직접 적용(연구/배치).

    예) MIMIC 코호트 행. 모델 미탑재 환경에서는 규칙 기반으로 폴백.
    """
    result = NephrologyService(db).analyze_vector(features)
    return AkiAnalysisResult(**result)


@router.post("/aki/analyze/{patient_id}", response_model=AkiAnalysisResult)
def analyze_patient(
    patient_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("nephrology", "admin")),
):
    """환자 기반 AKI 분석 — ai_risk_score 갱신 + 고위험 시 AI_ALERT 기록."""
    result = NephrologyService(db).analyze_patient(patient_id, actor_id=user.id)
    return AkiAnalysisResult(**result)


@router.get("/icu/aki-monitor", response_model=list[IcuAkiPatientOut])
def icu_aki_monitor(
    limit: int = 30,
    offset: int = 0,
    careunit: str | None = None,
    min_risk: int = 0,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("nephrology", "admin")),
):
    """실제 MIMIC-IV ICU 코호트에 학습 모델을 적용한 AKI 위험 모니터(위험순).

    신장내과 데모 환자(rule-based)와 달리, 표준화 48h 전체 피처가 있어 학습 모델이 유효하다.
    """
    if not is_available():
        return []
    return [IcuAkiPatientOut(**p) for p in
            IcuMonitorService().list_patients(limit=limit, offset=offset,
                                              careunit=careunit, min_risk=min_risk)]


@router.get("/icu/summary", response_model=IcuMonitorSummaryOut)
def icu_summary(
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("nephrology", "admin")),
):
    """ICU 코호트 AKI 위험 집계(고위험/중등도/안정)."""
    if not is_available():
        return IcuMonitorSummaryOut(total=0, high=0, moderate=0, low=0, n_careunits=0)
    return IcuMonitorSummaryOut(**IcuMonitorService().summary())


def _resolve_stay_id(db: Session, id_val: int) -> int:
    """id_val 이 stay_id 일 수도 있고, 데모로 전달된 subject_id 일 수도 있습니다.
    1. 만약 id_val 이 이미 cohort.stay_id 에 존재하는 진짜 stay_id 라면 그대로 리턴합니다.
    2. 그렇지 않다면 cohort.subject_id 에서 찾아 stay_id 로 치환해 줍니다.
    """
    # 진짜 stay_id 인지 먼저 검증
    exists_stay = db.execute(
        text("SELECT 1 FROM chym.cohort WHERE stay_id = :val LIMIT 1"),
        {"val": id_val}
    ).scalar()
    if exists_stay:
        return id_val

    # subject_id 이면 stay_id 로 치환
    row = db.execute(
        text("SELECT stay_id FROM chym.cohort WHERE subject_id = :val LIMIT 1"),
        {"val": id_val}
    ).first()
    if row:
        return int(row[0])
    return id_val


@router.get("/icu/careunits", response_model=list[CareunitStatOut])
def icu_careunits(
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("nephrology", "admin")),
):
    """ICU 병동(careunit)별 환자수/고위험수."""
    if not is_available():
        return []
    return [CareunitStatOut(**c) for c in IcuMonitorService().careunits()]


@router.get("/icu/patients/search", response_model=list[IcuPatientSearchOut])
def icu_patient_search(
    q: str,
    limit: int = 10,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("nephrology", "admin")),
):
    """ICU 코호트에서 stay_id · subject_id · 병동으로 환자 검색(자동완성·위험순)."""
    if not is_available():
        return []
    return [IcuPatientSearchOut(**p) for p in IcuPatientDetailService().search(q, limit=limit)]


@router.get("/icu/patients/{stay_id}/summary", response_model=IcuPatientSummaryOut)
def icu_patient_summary(
    stay_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("nephrology", "admin")),
):
    """Quick View — 환자 기본/입원/AKI 상태/신장 기능 요약."""
    stay_id = _resolve_stay_id(db, stay_id)
    summary = IcuPatientDetailService().summary(stay_id) if is_available() else None
    if summary is None:
        raise HTTPException(status_code=404, detail="patient not found in ICU cohort")
    mark_patient_viewed(stay_id)
    return IcuPatientSummaryOut(**summary)


@router.get("/icu/patients/{stay_id}/trends", response_model=PatientTrendsOut)
def icu_patient_trends(
    stay_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("nephrology", "admin")),
):
    """환자별 Cr · eGFR · 소변량 추세 그래프 데이터."""
    stay_id = _resolve_stay_id(db, stay_id)
    if not is_available():
        return PatientTrendsOut(stay_id=stay_id, creatinine=[], egfr=[], urine_output=[])
    return PatientTrendsOut(**IcuPatientDetailService().trends(stay_id))


@router.get("/icu/patients/{stay_id}/shap", response_model=PatientShapOut)
def icu_patient_shap(
    stay_id: int,
    limit: int = 8,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("nephrology", "admin")),
):
    """환자별 상위 기여 피처(SHAP-style) — 환자마다 다르게 계산."""
    stay_id = _resolve_stay_id(db, stay_id)
    if not is_available():
        return PatientShapOut(stay_id=stay_id, features=[])
    return PatientShapOut(**IcuPatientDetailService().shap(stay_id, limit=limit))


@router.get("/icu/patients/{stay_id}/validation", response_model=PatientValidationOut)
def icu_patient_validation(
    stay_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("nephrology", "admin")),
):
    """환자별 예측 vs 실제 + 모델 신뢰도 곡선(ECE 대체)."""
    stay_id = _resolve_stay_id(db, stay_id)
    validation = IcuPatientDetailService().validation(stay_id) if is_available() else None
    if validation is None:
        raise HTTPException(status_code=404, detail="patient not found in ICU cohort")
    return PatientValidationOut(**validation)


@router.get("/icu/patients/{stay_id}/risk-history", response_model=RiskHistoryOut)
def icu_patient_risk_history(
    stay_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("nephrology", "admin")),
):
    """위험도 추세 — Cr 기반 KDIGO 추정선 + 실제 모델 예측 1점(정직성 구분)."""
    stay_id = _resolve_stay_id(db, stay_id)
    if not is_available():
        return RiskHistoryOut(stay_id=stay_id, trajectory=[], model_point=None,
                               disclaimer="코호트 미가용")
    return RiskHistoryOut(**IcuPatientDetailService().risk_history(stay_id))


@router.get("/icu/patients/{stay_id}/rrt-assessment", response_model=RrtAssessmentOut)
def icu_patient_rrt_assessment(
    stay_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("nephrology", "admin")),
):
    """RRT 트리거(임상 의사결정 보조) — KDIGO+ICU 규칙 기반 단계 분류(예측 아님)."""
    stay_id = _resolve_stay_id(db, stay_id)
    assessment = IcuPatientDetailService().rrt_assessment(stay_id) if is_available() else None
    if assessment is None:
        raise HTTPException(status_code=404, detail="patient not found or no creatinine series")
    return RrtAssessmentOut(**assessment)


@router.post("/icu/patients/{stay_id}/draft", response_model=AiDraftResultOut)
def icu_patient_draft(
    stay_id: int,
    body: VoiceDraftRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("nephrology", "admin")),
):
    """실제 ICU stay 기반 AI 진료 초안(SOAP) — 전사 → 근거기반 SOAP·Problem·CDSS·검증.

    데모 환자 mock 없이 실데이터로 동일 파이프라인을 돌린다(영속화 없음).
    """
    stay_id = _resolve_stay_id(db, stay_id)
    if not is_available():
      raise HTTPException(status_code=503, detail="ICU cohort/model not available")
    result = IcuDraftService().draft_for_stay(stay_id, body.transcript or "")
    if result is None:
      raise HTTPException(status_code=404, detail="patient not found in ICU cohort")
    return AiDraftResultOut(**result)


@router.get("/icu/model-metrics")
def icu_model_metrics(
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("nephrology", "admin")),
):
    """모델 성능 지표(테스트셋 운영 threshold 기준 정확도/Precision/위음성률 + 검증 리포트).

    행마다 동일한 모델 단위 상수이므로 목록 위 성능 패널·검증 리포트 모달에서 사용한다.
    """
    from services.model_metrics_service import available as metrics_available
    from services.model_metrics_service import model_metrics

    if not metrics_available():
        return {}
    return model_metrics()


@router.get("/timeline/{patient_id}", response_model=list[TimelineEventOut])
def read_timeline(
    patient_id: str,
    severity: str | None = None,
    event_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """신장내과 환자 상태 조회(읽기 전용) — AKI 도메인은 타임라인을 읽기만 한다."""
    events = NephrologyService(db).read_timeline(
        patient_id, Page.of(limit, offset), severity, event_type
    )
    return [timeline_to_out(e) for e in events]
