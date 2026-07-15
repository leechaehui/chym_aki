"""협진 DTO — 프론트 Consult/ConsultEvent/ConsultReply 대응."""
from typing import Any

from schemas.base import CamelModel


class ConsultEventOut(CamelModel):
    id: str
    stage: str
    label: str
    at: str
    actor: str


class ConsultReplyOut(CamelModel):
    findings: str
    diagnosis: str
    recommendation: str
    author: str
    replied_at: str
    # 판독의 전자 서명 이미지 경로(공개 /uploads) + stain별 AI 분석 요약(ROI·heatmap·신뢰도).
    signature_path: str | None = None
    analysis: list[Any] | None = None


class ConsultOut(CamelModel):
    id: str
    kind: str
    patient_mrn: str
    patient_name: str
    diagnosis: str
    key_labs: str
    reason: str
    urgency: str
    status: str
    requested_by: str
    requested_at: str
    bed_label: str | None = None
    timeline: list[ConsultEventOut] = []
    reply: ConsultReplyOut | None = None


class ConsultRequestInput(CamelModel):
    kind: str = "pathology"
    patient_mrn: str
    patient_name: str
    diagnosis: str
    key_labs: str = ""
    reason: str = ""
    urgency: str = "routine"
    requested_by: str
    bed_label: str | None = None


class ConsultAcceptInput(CamelModel):
    actor: str


class ConsultReplyInput(CamelModel):
    findings: str
    diagnosis: str
    recommendation: str
    author: str
    signature_path: str | None = None
    analysis: list[Any] | None = None
