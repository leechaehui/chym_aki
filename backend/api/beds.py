"""병상 API (응급의학과). 배정/해제는 emergency/admin 권한 필요(RBAC)."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from core.deps import get_current_user, get_db, require_roles
from core.query_optimizer import Page
from models.user import User
from schemas.admission import AdmissionOut, BedActionResult, BedAssignRequest
from schemas.bed import BedOut, BedSummaryOut
from schemas.bed_detail import BedPatientDetailOut
from services.bed_detail_service import BedDetailService
from services.bed_service import BedService

router = APIRouter(prefix="/beds", tags=["beds"])


def _admission_out(adm) -> AdmissionOut | None:
    if not adm:
        return None
    return AdmissionOut(
        id=adm.id,
        patient_id=adm.patient_id,
        bed_id=adm.bed_id,
        status=adm.status,
        admitted_at=adm.admitted_at.isoformat() if adm.admitted_at else None,
        discharged_at=adm.discharged_at.isoformat() if adm.discharged_at else None,
    )


@router.get("", response_model=list[BedOut])
def list_beds(db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    return [BedOut.model_validate(b) for b in BedService(db).list_beds()]


@router.get("/summary", response_model=list[BedSummaryOut])
def bed_summary(db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    service = BedService(db)
    return [BedSummaryOut(**s) for s in service.summarize(service.list_beds())]




@router.get("/details", response_model=dict[str, BedPatientDetailOut])
def list_bed_details(db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    """사용중 병상별 입실 상세(처방·검사·AKI 처치권고). bedId → 상세."""
    return BedDetailService(db).list_details()




@router.post("/{bed_id}/assign", response_model=BedActionResult)
def assign_bed(
    bed_id: str,
    body: BedAssignRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("admin")),
):
    bed, admission = BedService(db).assign_bed(
        bed_id=bed_id,
        actor_id=user.id,
        actor_name=user.name,
        patient_id=body.patient_id,
        patient_name=body.patient_name,
        sex=body.sex,
        age=body.age,
        diagnosis=body.diagnosis,
    )
    return BedActionResult(
        bed=BedOut.model_validate(bed), admission=_admission_out(admission)
    )


@router.post("/{bed_id}/release", response_model=BedActionResult)
def release_bed(
    bed_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("admin")),
):
    bed, admission = BedService(db).release_bed(
        bed_id=bed_id, actor_id=user.id, actor_name=user.name
    )
    return BedActionResult(
        bed=BedOut.model_validate(bed), admission=_admission_out(admission)
    )
