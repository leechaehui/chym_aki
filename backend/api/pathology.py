"""병리 API — WSI 분석 결과 조회 + 병리 보고서(소견/진단) 저장."""
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from core.deps import get_current_user, get_db
from models.user import User
from schemas.pathology import PathologyReportIn, PathologyResultOut
from services.pathology_service import PathologyService

router = APIRouter(prefix="/pathology", tags=["pathology"])


@router.get("", response_model=list[PathologyResultOut])
def list_results(db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    return PathologyService(db).list()


@router.get("/by-consult/{consult_id}", response_model=PathologyResultOut)
def get_by_consult(
    consult_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    return PathologyService(db).get_by_consult(consult_id)


@router.put("/by-consult/{consult_id}/report", response_model=PathologyResultOut)
def save_report(
    consult_id: str,
    body: PathologyReportIn,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    return PathologyService(db).save_report(consult_id, body)


@router.get("/wsi-mapping/{subject_id}")
def get_wsi_mapping(
    subject_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """환자(subject_id=patientMrn) ↔ WSI 슬라이드(8001, PACS case_id) 매핑 —
    chym.phase2_wsi_mapping 조회. 8001은 DB를 모른 채(PACS 전용) 유지하고,
    8010이 매핑만 조회해서 프론트에 slide_id 목록을 준다(architecture: wsi/README.md 원칙 유지)."""
    rows = db.execute(
        text("select slide_id, stain_type from chym.phase2_wsi_mapping where subject_id = :sid"),
        {"sid": subject_id},
    ).fetchall()
    return [{"slide_id": r[0], "stain": r[1]} for r in rows]
