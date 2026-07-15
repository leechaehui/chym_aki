"""병상/응급환자 DTO — 프론트 Bed/BedSummary/EmergencyPatient 대응."""
from schemas.base import CamelModel


class BedOut(CamelModel):
    id: str
    zone: str
    label: str
    state: str
    patient_name: str | None = None


class BedSummaryOut(CamelModel):
    """구역별 집계(KPI/보드 헤더)."""

    zone: str
    total: int
    occupied: int
    available: int


