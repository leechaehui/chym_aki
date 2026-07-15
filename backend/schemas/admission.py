"""입원/병상 트랜잭션 DTO — 배정/해제 요청 및 입원 결과."""
from schemas.base import CamelModel


class BedAssignRequest(CamelModel):
    """병상 배정 요청. 기존 환자 배정(patient_id) 또는 신규 등록(이름/성별/나이) 모두 지원."""

    patient_id: str | None = None
    # 신규 환자 즉석 등록(응급 입실)용 — patient_id 미지정 시 사용.
    patient_name: str | None = None
    sex: str | None = None
    age: int | None = None
    diagnosis: str | None = None


class AdmissionOut(CamelModel):
    id: str
    patient_id: str
    bed_id: str
    status: str
    admitted_at: str | None = None
    discharged_at: str | None = None


class BedActionResult(CamelModel):
    """배정/해제 결과 — 갱신된 병상 + 관련 입원 기록."""

    bed: "BedOut"
    admission: AdmissionOut | None = None


from schemas.bed import BedOut  # noqa: E402  (전방 참조 해소)

BedActionResult.model_rebuild()
