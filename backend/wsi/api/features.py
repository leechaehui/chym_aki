"""피처 추출 Controller — has_features=False 슬라이드의 온디맨드 패치추출 트리거/폴링."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from wsi.core.auth import require_pathology
from wsi.infra.engine_factory import get_feature_extraction_service
from wsi.schemas.wsi import WsiStain

router = APIRouter(prefix="/wsi", tags=["wsi-features"], dependencies=[Depends(require_pathology)])


@router.post("/extract-features/{slide_id}")
def extract_features(slide_id: str, stain: WsiStain = Query(...), case_code: str | None = Query(None)):
    """패치추출+인코딩 시작(백그라운드). 이미 진행 중/완료면 현재 상태만 반환(멱등)."""
    return get_feature_extraction_service().start(slide_id, stain, case_code)


@router.get("/extract-status/{slide_id}")
def extract_status(slide_id: str, stain: WsiStain = Query(...)):
    """진행률 폴링 — {status: not_started|downloading|extracting|encoding|ready|error, ...}."""
    return get_feature_extraction_service().status(slide_id, stain)
