"""협진 API."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.serializers import consult_to_out
from core.deps import get_current_user, get_db
from core.query_optimizer import Page
from models.user import User
from schemas.consultation import (
    ConsultAcceptInput,
    ConsultOut,
    ConsultReplyInput,
    ConsultRequestInput,
)
from services.consultation_service import ConsultationService

router = APIRouter(prefix="/consultations", tags=["consultation"])


@router.get("", response_model=list[ConsultOut])
def list_consults(
    kind: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    svc = ConsultationService(db)
    consults = svc.list(Page.of(limit, offset), kind)
    return [consult_to_out(c, patient_name=svc.canonical_patient_name(c.patient_mrn)) for c in consults]


@router.get("/{consult_id}", response_model=ConsultOut)
def get_consult(
    consult_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    svc = ConsultationService(db)
    consult = svc.get(consult_id)
    return consult_to_out(consult, patient_name=svc.canonical_patient_name(consult.patient_mrn))


@router.post("", response_model=ConsultOut, status_code=201)
def request_consult(
    body: ConsultRequestInput,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    consult = ConsultationService(db).request(
        actor_id=user.id, data=body.model_dump()
    )
    return consult_to_out(consult)


@router.post("/{consult_id}/accept", response_model=ConsultOut)
def accept_consult(
    consult_id: str,
    body: ConsultAcceptInput,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    svc = ConsultationService(db)
    consult = svc.accept(consult_id, actor_id=user.id, actor=body.actor)
    return consult_to_out(consult, patient_name=svc.canonical_patient_name(consult.patient_mrn))


@router.post("/{consult_id}/reply", response_model=ConsultOut)
def reply_consult(
    consult_id: str,
    body: ConsultReplyInput,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    svc = ConsultationService(db)
    consult = svc.reply(consult_id, actor_id=user.id, reply=body.model_dump())
    return consult_to_out(consult, patient_name=svc.canonical_patient_name(consult.patient_mrn))
