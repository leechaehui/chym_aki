"""분석 Controller — /analyze, /result. HTTP ↔ 유스케이스 변환만."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from wsi.core.auth import require_pathology
from wsi.infra.engine_factory import get_abmil_service, get_analysis_service
from wsi.schemas.wsi import AnalyzeRequest, WsiAnalysisResult, WsiStain

# 병리과 화면 전용 — pathology/admin 만 접근.
router = APIRouter(tags=["wsi"], dependencies=[Depends(require_pathology)])

# stain 별 추론 엔진 분리 — PAS 는 Pathology_model(CdssEngine, 멀티스테인),
# HE/MT 는 aki_wsi ABMIL(단일스테인 회귀). 둘 다 8001 하나로 통합.
_ABMIL_STAINS = {"HE", "MT"}


@router.post("/analyze", response_model=WsiAnalysisResult)
def analyze(req: AnalyzeRequest):
    if req.stain in _ABMIL_STAINS:
        return get_abmil_service().analyze(
            slide_id=req.slide_id, stain=req.stain, use_cache=req.use_cache)
    return get_analysis_service().analyze(
        slide_id=req.slide_id, stain=req.stain, use_cache=req.use_cache)


@router.get("/result/{stain}/{slide_id}", response_model=WsiAnalysisResult)
def get_result(stain: WsiStain, slide_id: str):
    if stain in _ABMIL_STAINS:
        return get_abmil_service().get_result(stain=stain, slide_id=slide_id)
    return get_analysis_service().get_result(stain=stain, slide_id=slide_id)
