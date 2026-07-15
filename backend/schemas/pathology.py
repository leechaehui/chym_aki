"""병리 DTO — 프론트 PathologyResult/DetectedLayer/QuantMetric/PathologyReport 대응."""
from schemas.base import CamelModel


class DetectedLayerOut(CamelModel):
    key: str
    label: str
    color: str
    count: int | None = None
    visible: bool


class QuantMetricOut(CamelModel):
    key: str
    label: str
    value: float | None = None
    unit: str


class PathologyReportOut(CamelModel):
    findings: str
    diagnosis: str
    status: str  # draft | final
    updated_at: str | None = None


class PathologyResultOut(CamelModel):
    consult_id: str
    stain: str
    image_url: str | None = None
    layers: list[DetectedLayerOut] = []
    metrics: list[QuantMetricOut] = []
    report: PathologyReportOut


class PathologyReportIn(CamelModel):
    findings: str
    diagnosis: str
    status: str  # draft | final
