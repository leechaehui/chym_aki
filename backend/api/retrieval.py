"""Retrieval CDSS API — Clinical Concept → Pathology Reference Retrieval.

Evidence Display 전용. 진단/treatment 권고 생성 경로 없음.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.deps import get_current_user, get_db
from models.user import User
import models.retrieval as R
from retrieval.engine.calibration import Calibrator
from retrieval.ood.detector import OodDetector
from retrieval.service import RetrievalService
from schemas.retrieval import RetrievalQueryIn, RetrievalResultOut

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.post("/query", response_model=RetrievalResultOut)
def query(
    body: RetrievalQueryIn,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """임상 concept → OOD 게이트 → 규칙 prefilter → metadata 코사인 → Top-K + 대표 WSI."""
    return RetrievalService(
        db, ood_detector=OodDetector.load(db), calibrator=Calibrator.load(db),
    ).query(body)


@router.get("/prototypes/{prototype_id}")
def get_prototype(
    prototype_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """프로토타입 상세(라벨·멤버수·검증자·대표 WSI)."""
    p = db.get(R.Prototype, prototype_id)
    if p is None:
        raise HTTPException(404, "프로토타입을 찾을 수 없습니다.")
    rep = (db.query(R.WsiMetadata)
           .filter(R.WsiMetadata.prototype_id == p.id,
                   R.WsiMetadata.is_representative.is_(True)).first())
    return {
        "prototypeId": p.id, "label": p.label, "nMembers": p.n_members,
        "isRare": p.is_rare, "kdigoBand": p.kdigo_band, "etiologyHint": p.etiology_hint,
        "egfrMean": p.egfr_mean, "validatedBy": p.validated_by,
        "representativeWsi": ({"slideId": rep.slide_id, "stain": rep.stain} if rep else None),
        "memberPatientIds": [m.patient_id for m in p.members],
    }


@router.get("/prototypes")
def list_prototypes(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """아틀라스 전체(대시보드/디버깅용)."""
    rows = db.query(R.Prototype).order_by(R.Prototype.n_members.desc()).all()
    return [{"prototypeId": p.id, "label": p.label, "nMembers": p.n_members,
             "isRare": p.is_rare, "kdigoBand": p.kdigo_band,
             "etiologyHint": p.etiology_hint, "egfrMean": p.egfr_mean} for p in rows]
