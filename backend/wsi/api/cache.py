"""슬라이드 준비(캐시) Controller — 뷰어 진입 전 SVS 가용성 확인/워밍.

프론트(wsiService.cacheRoute)는 8001 을 우선 호출하고 미구현(404)이면 8010 으로 폴백하므로,
이 라우트가 존재하는 순간 자동으로 8001(추론서버)로 전환된다.
병리과 화면 전용 — pathology/admin 만 접근(라우터 전체 게이트).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from wsi.core.auth import require_pathology
from wsi.infra.engine_factory import get_cache_service
from wsi.schemas.wsi import WsiCacheStatus, WsiStain

router = APIRouter(prefix="/wsi", tags=["wsi-cache"],
                   dependencies=[Depends(require_pathology)])


@router.post("/prepare/{slide_id}", response_model=WsiCacheStatus,
             response_model_exclude_none=True)
def prepare(slide_id: str, stain: WsiStain | None = Query(None)):
    """슬라이드 준비 시작 — 로컬 SVS 면 즉시 ready, 부재/손상이면 error."""
    return get_cache_service().prepare(slide_id, stain)


@router.get("/cache-status/{slide_id}", response_model=WsiCacheStatus,
            response_model_exclude_none=True)
def cache_status(slide_id: str, stain: WsiStain | None = Query(None)):
    """준비 상태 폴링용 — 현재 가용성(ready/error)."""
    return get_cache_service().status(slide_id, stain)
