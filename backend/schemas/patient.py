"""환자/검사 DTO — 프론트 Patient/LabValue/Trend/UrineOutput 형태에 1:1 대응."""
from schemas.base import CamelModel


class LabValueOut(CamelModel):
    key: str
    label: str
    value: float
    unit: str
    ref_low: float | None = None
    ref_high: float | None = None
    flag: str


class LabTrendPointOut(CamelModel):
    date: str
    creatinine: float
    egfr: float
    bun: float


class UrineOutputPointOut(CamelModel):
    date: str
    value: float


class PatientOut(CamelModel):
    id: str
    mrn: str
    name: str
    sex: str
    age: int
    diagnosis: str
    admitted_at: str
    attending: str
    room: str
    ai_risk_score: int
    # 모델의 실제 예측 stage 문자열("Non-AKI"/"AKI Stage 1"/"AKI Stage 2-3") — 있으면 프론트가
    # 위험 등급 뱃지를 점수 대신 이 값으로 판정해 다른 화면(ICU AKI 모니터링)과 등급이 일치하게 한다.
    predicted_stage: str | None = None
    labs: list[LabValueOut] = []
    trend: list[LabTrendPointOut] = []
    urine_output: list[UrineOutputPointOut] = []


class PatientSummaryOut(CamelModel):
    """목록용 경량 환자 DTO(검사 배열 제외 — SELECT * 회피)."""

    id: str
    mrn: str
    name: str
    sex: str
    age: int
    diagnosis: str
    admitted_at: str
    attending: str
    room: str
    ai_risk_score: int
