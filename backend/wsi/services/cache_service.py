"""슬라이드 준비(캐시) 유스케이스 — 뷰어 진입 전 SVS 로컬 가용성 확인 + 핸들 워밍.

manifest 슬라이드의 원본 SVS 는 로컬(svs_root)에 존재하므로 '다운로드'는 없다.
  prepare = SVS 해석(존재 확인) + DeepZoom 핸들 워밍(첫 타일 가속) → ready.
SVS 부재/손상은 error(터미널) 로 보고한다 — 프론트 폴링이 ready/error 에서만 멈추므로
not_started 를 돌려주면 무한 폴링이 된다(그래서 반환하지 않는다).

PACS 케이스의 원격 다운로드/캐시는 별 경로(PacsDicomTileSource._ensure_downloaded)로 이미 분리.
DIP: 포트(SlideRepository/TileSource)에만 의존 — 데이터 출처/타일러 교체에 영향 없음.
"""
from __future__ import annotations

import logging
from typing import get_args

from wsi.core.errors import TileUnavailableError
from wsi.domain.ports import SlideRepository, TileSource
from wsi.schemas.wsi import WsiCacheStatus, WsiStain

log = logging.getLogger("wsi.cache")

# 프론트가 prepare 를 stain 없이 호출하므로(slide_id 만), 미지정 시 전 stain 을 훑는다.
_STAINS: tuple[str, ...] = tuple(get_args(WsiStain))  # ("HE", "MT")


class SlideCacheService:
    """뷰어용 슬라이드 준비 — 로컬 SVS 확인/워밍만(원격 다운로드 없음)."""

    def __init__(self, *, repo: SlideRepository, tiles: TileSource):
        self._repo = repo
        self._tiles = tiles

    def _resolve_svs(self, slide_id: str, stain: str | None):
        """slide_id(+선택 stain)로 로컬 SVS 경로 해석. 없으면 None."""
        for s in ((stain,) if stain else _STAINS):
            svs = self._repo.svs_path(slide_id, s)
            if svs is not None:
                return svs
        return None

    def prepare(self, slide_id: str, stain: str | None = None) -> WsiCacheStatus:
        svs = self._resolve_svs(slide_id, stain)
        if svs is None:
            return WsiCacheStatus(status="error",
                                  message="로컬에 SVS 파일이 없습니다(스테이징 대상 아님).")
        try:
            # openslide + DeepZoom 핸들 워밍(메타데이터만 읽어 빠름). 이후 첫 타일 가속 + 손상 조기검출.
            self._tiles.dzi_descriptor(svs)
        except TileUnavailableError as e:
            log.warning("prepare 워밍 실패 slide=%s stain=%s: %s", slide_id, stain, e)
            return WsiCacheStatus(status="error", message="SVS 를 열 수 없습니다(파일 손상 가능).")
        log.info("prepare ready slide=%s stain=%s svs=%s", slide_id, stain, svs.name)
        return WsiCacheStatus(status="ready")

    def status(self, slide_id: str, stain: str | None = None) -> WsiCacheStatus:
        svs = self._resolve_svs(slide_id, stain)
        if svs is None:
            return WsiCacheStatus(status="error", message="로컬에 SVS 파일이 없습니다.")
        return WsiCacheStatus(status="ready")
