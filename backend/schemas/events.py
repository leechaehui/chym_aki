"""이벤트 / Alert 스키마 (지시서 §3, §8).

CamelModel 상속 → 입력은 snake/camel 모두 허용, 출력은 camelCase(프론트 직결).
이벤트 dict 는 버스에서 평범한 dict 로 흐르고, 이 스키마는 API 경계(검증/직렬화)에서만 쓴다.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from schemas.base import CamelModel

# --- 액션/심각도 리터럴 ---
AuditAction = Literal["VIEWED", "DISMISSED", "ACKNOWLEDGED", "ESCALATED"]
Severity = Literal["info", "warning", "critical"]


# ----------------------------------------------------------
# 3.1 LAB_EVENT (인제스트 입력)
# ----------------------------------------------------------
class LabData(CamelModel):
    creatinine: float | None = None
    egfr: float | None = None
    urine_output: float | None = None  # mL/kg/h
    # 선택적 임상 컨텍스트(safety/guardrail 용)
    symptom: str | None = None          # 예: "소변량 감소"
    dehydration: bool | None = None     # FP 가드: 탈수
    contrast_exposure: bool | None = None  # FP 가드: 조영제 노출
    diuretics: bool | None = None       # FP 가드: 이뇨제(UO 해석 왜곡)
    sex: str | None = None              # baseline population prior(M/F)
    ckd_risk: bool | None = None        # baseline 상향(CKD 위험군)


class LabEvent(CamelModel):
    event_type: Literal["LAB_EVENT"] = "LAB_EVENT"
    patient_id: str
    subject_id: int | None = None
    data: LabData
    # 선택: 직접 baseline/이전 Cr 추이를 함께 전달(없으면 엔진이 추정)
    prior_creatinines: list[float] = Field(default_factory=list)
    age: int | None = None
    timestamp: int | None = None


# ----------------------------------------------------------
# 3.3 ALERT_EVENT / 8. alerts 레코드 출력
# ----------------------------------------------------------
class AlertOut(CamelModel):
    id: str
    patient_id: str
    type: str
    severity: Severity
    priority: int
    aki_stage: str | None = None
    aki_score: float | None = None
    subject_id: int | None = None
    dedup_key: str
    title: str
    message: str
    source_event_id: str | None = None
    department: str
    status: str
    created_at: str


# ----------------------------------------------------------
# 8. audit 액션 입력
# ----------------------------------------------------------
class AlertAuditInput(CamelModel):
    action: AuditAction
    role: str | None = None


# ----------------------------------------------------------
# event_log 트레이스 출력
# ----------------------------------------------------------
class EventLogOut(CamelModel):
    id: str
    event_type: str
    patient_id: str | None = None
    payload: dict[str, Any] | None = None
    created_at: str


# 인제스트 응답(생성된 alert 요약)
class IngestResult(CamelModel):
    accepted: int
    alerts: list[AlertOut]
