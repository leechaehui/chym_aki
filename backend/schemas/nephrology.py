"""신장내과 DTO — AKI 분석 + 음성 기반 AI 진료 초안(SOAP)."""
from schemas.base import CamelModel


# ----------------------------------------------------------
# AKI 분석
# ----------------------------------------------------------
class AkiFeatureInput(CamelModel):
    """AKI 추론 입력 피처(48h 피처셋 부분집합).

    모든 필드 optional — 누락 시 모델/룰이 보수적 기본값으로 처리한다.
    핵심 KDIGO 인자(creatinine, urine, oliguria) 위주.
    """
    creatinine_max: float | None = None
    creatinine_min: float | None = None
    creatinine_delta: float | None = None
    baseline_creatinine: float | None = None
    egfr: float | None = None
    bun_max: float | None = None
    potassium_max: float | None = None
    bicarbonate_min: float | None = None
    urine_ml_kg_hr: float | None = None
    urine_output_6h: float | None = None
    oliguria_flag: int | None = None
    lactate_max: float | None = None
    map_min: float | None = None
    vasopressor_flag: int | None = None
    age: int | None = None


class AkiAnalysisResult(CamelModel):
    """AKI 추론 결과 — 3-class(Non-AKI/Stage1/Stage2+3) + 위험점수."""

    risk: str  # high | moderate | low
    risk_label: str
    stage: str  # 예: "AKI Stage 2"
    stage_num: int  # 0|1|2|3
    risk_score: int  # 0–100
    p_non_aki: float
    p_stage1: float
    p_stage2_plus: float
    rationale: list[str]  # 판정 근거(설명가능성)
    source: str  # "model" | "rule-based"


# ----------------------------------------------------------
# ICU AKI 모니터 (실제 MIMIC-IV ICU 코호트 + 학습 모델)
# ----------------------------------------------------------
class IcuAkiPatientOut(CamelModel):
    """ICU 입실 환자별 학습 모델 AKI 예측 + 정답(검증용)."""

    stay_id: int
    subject_id: int
    age: int | None = None
    gender: str
    careunit: str
    icu_los_hours: float | None = None
    predict_at: str | None = None  # AKI 발생 위험 예측 시각(icu_intime+48h, HH:MM)
    risk_score: int
    risk: str
    risk_label: str
    stage: str
    stage_num: int
    p_non_aki: float
    p_stage1: float
    p_stage2_plus: float
    is_new_patient: bool = False
    source: str  # "model"
    actual_label: int   # 실제 AKI 발생(0/1) — 모델 정확도 대조용
    actual_stage: int   # 실제 AKI stage(0–3)


class IcuMonitorSummaryOut(CamelModel):
    total: int
    high: int
    moderate: int
    low: int
    # AI 위험점수 밴드 집계(기본 0 — summary 폴백 호환).
    risk_high: int = 0
    risk_moderate: int = 0
    risk_low: int = 0
    n_careunits: int


# ----------------------------------------------------------
# ICU 환자 상세 (검색 · Quick View · Trend · SHAP · Validation · Risk History)
# ----------------------------------------------------------
class IcuPatientSearchOut(CamelModel):
    """환자 검색 결과 한 건(자동완성·빠른 진입용)."""

    stay_id: int
    subject_id: int
    age: int | None = None
    gender: str
    careunit: str
    risk_score: int
    risk: str
    stage: str


class IcuPatientSummaryOut(CamelModel):
    """Quick View — 기본 정보 · 입원 · AKI 상태 · 신장 기능(결측은 None)."""

    stay_id: int
    subject_id: int
    age: int | None = None
    gender: str
    weight_kg: float | None = None
    height_cm: float | None = None
    bmi: float | None = None
    careunit: str
    bed: str | None = None
    admission_type: str | None = None
    icu_los_hours: float | None = None
    elapsed_text: str | None = None
    predict_at: str | None = None
    stage: str
    stage_num: int
    risk_score: int
    risk: str
    p_stage2_plus: float
    creatinine: float | None = None
    egfr: float | None = None
    bun: float | None = None
    baseline_creatinine: float | None = None
    urine_rate_ml_kg_h: float | None = None


class TrendPointOut(CamelModel):
    """추세 그래프 한 점 — 입실 후 경과시간(hours) 대비 값."""

    hours: float
    value: float


class PatientTrendsOut(CamelModel):
    """환자별 Cr · eGFR · 소변량 시계열."""

    stay_id: int
    creatinine: list[TrendPointOut]
    egfr: list[TrendPointOut]
    urine_output: list[TrendPointOut]


class ShapFeatureOut(CamelModel):
    """환자별 피처 기여도 한 건(부호 = 위험 ↑/↓)."""

    feature: str
    label: str
    contribution: float
    scaled_value: float
    value: float | None = None
    unit: str = ""
    shap_value: float | None = None


class PatientShapOut(CamelModel):
    stay_id: int
    features: list[ShapFeatureOut]
    shap_features: list[ShapFeatureOut] = []


class CalibrationBinOut(CamelModel):
    """신뢰도 곡선 구간 — 예측확률 평균 vs 실제 발생률."""

    predicted_mean: float
    observed_rate: float
    count: int


class CalibrationOut(CamelModel):
    bins: list[CalibrationBinOut]
    ece: float
    n: int


class PatientValidationOut(CamelModel):
    """환자별 예측 vs 실제 + 모델 신뢰도 곡선."""

    stay_id: int
    predicted_stage: str
    predicted_stage_num: int
    p_aki: float
    p_stage2_plus: float
    risk_score: int
    actual_label: int
    actual_stage: int
    correct: bool
    calibration: CalibrationOut


class RiskTrajectoryPointOut(CamelModel):
    """Cr 기반 KDIGO 위험 추정 한 점(실제 모델 출력 아님)."""

    hours: float
    risk_score: int
    kdigo_stage: int
    source: str  # "cr_kdigo"


class RiskModelPointOut(CamelModel):
    """실제 모델 위험 출력 1점(입실+48h)."""

    hours: float
    risk_score: int
    source: str  # "model"


class RiskHistoryOut(CamelModel):
    stay_id: int
    baseline_creatinine: float | None = None
    trajectory: list[RiskTrajectoryPointOut]
    model_point: RiskModelPointOut | None = None
    disclaimer: str


class RrtAssessmentOut(CamelModel):
    """RRT 트리거 — 임상 의사결정 보조(예측 아님). KDIGO+ICU 규칙 기반 단계 분류."""

    stay_id: int
    rrt_level: str          # LEVEL_0 | DISCUSSION | CONSIDERATION | URGENT (enum name)
    label: str
    confidence: float
    key_drivers: list[str]
    trend_signal: str       # stable | worsening | rapidly deteriorating
    clinical_action: str
    warning_note: str
    unavailable_criteria: list[str]  # 정직성: 평가 못한 기준


class CareunitStatOut(CamelModel):
    careunit: str
    n: int
    high: int


# ----------------------------------------------------------
# 음성 기반 AI 진료 초안 (근거 기반 SOAP + Problem List + CDSS)
# ----------------------------------------------------------
class VoiceDraftRequest(CamelModel):
    """음성/대화 기반 초안 생성 요청.

    transcript 직접 입력(passthrough) 또는 audio 업로드 후 STT(별도 엔드포인트).
    """
    patient_id: str
    transcript: str = ""


class AiDraftResultOut(CamelModel):
    """AI Draft 파이프라인 결과(SOAP/Problem List/CDSS Risk/Validation).

    soap/risk/validation 은 중첩 구조(S/O/A/P, breakdown, evidence)를 그대로 노출하기 위해
    dict 로 전달한다(키 의미가 표준 SOAP 라벨이라 camelCase 변환을 피한다).
    """
    id: str
    patient_id: str
    transcript: str
    symptoms: dict
    soap: dict
    problem_list: list[dict]
    risk: dict
    validation: dict
    status: str
    created_at: str | None = None


# 하위호환 별칭(기존 import 보호).
AiDraftNoteOut = AiDraftResultOut


class TranscriptionResult(CamelModel):
    """STT 결과."""

    transcript: str
    engine: str  # whisper | passthrough
