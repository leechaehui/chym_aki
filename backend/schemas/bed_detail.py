"""병상 상세/예약 DTO — 프론트 BedPatientDetail/Medication/BedLab/BedReservation 대응."""
from schemas.base import CamelModel


class MedicationOut(CamelModel):
    name: str
    dose: str
    route: str
    status: str


class BedLabOut(CamelModel):
    label: str
    value: str
    unit: str
    flag: str  # normal|high|low


class VitalEntryOut(CamelModel):
    label: str
    value: str


class BedPatientDetailOut(CamelModel):
    diagnosis: str
    attending: str
    admitted_at: str
    aki_risk: bool
    aki_stage: str | None = None
    medications: list[MedicationOut] = []
    labs: list[BedLabOut] = []
    recent_inputs: list[VitalEntryOut] | None = None
    treatment_report: str | None = None
    treatment_items: list[str] | None = None


