"""Factory / 합성 루트 — 객체 생성·의존성 배선을 한 곳에 격리.

api/service 는 '무엇을 쓸지'만 알고 '어떻게 만들지'는 모른다(생성 로직 은닉).
지연 싱글톤(lru_cache): 무거운 엔진(torch)은 첫 /analyze 때 로드 → 서버 기동/목록·타일은
엔진 없이도 동작(장애 격리). 구현 교체는 이 파일만 수정(OCP).
"""
from __future__ import annotations

from functools import lru_cache

from wsi.core.config import get_settings
from wsi.domain.ports import InferenceEngine, ResultCache, SlideRepository, TileSource
from wsi.infra.cdss_engine_adapter import CdssEngineAdapter
from wsi.infra.manifest_slide_repo import ManifestSlideRepository
from wsi.infra.result_cache import FileResultCache
from wsi.infra.tile_source import OpenSlideTileSource
from wsi.services.analysis_service import AnalysisService


@lru_cache(maxsize=1)
def get_repository() -> SlideRepository:
    return ManifestSlideRepository(get_settings())


@lru_cache(maxsize=1)
def get_tile_source() -> TileSource:
    return OpenSlideTileSource()


@lru_cache(maxsize=1)
def get_cache() -> ResultCache:
    return FileResultCache(get_settings())


@lru_cache(maxsize=1)
def get_engine() -> InferenceEngine:
    """무거운 추론 엔진(5-seed ensemble) — 첫 호출 시 1회만 로드."""
    return CdssEngineAdapter(get_settings())


@lru_cache(maxsize=1)
def get_analysis_service() -> AnalysisService:
    return AnalysisService(repo=get_repository(), cache=get_cache(), engine_provider=get_engine)


@lru_cache(maxsize=1)
def get_abmil_service():
    """HE/MT 전용 ABMIL 분석 서비스(aki_wsi 모델) — CdssEngine(PAS)과 별도 파이프라인."""
    from wsi.infra.abmil_engine import AbmilAnalysisService
    return AbmilAnalysisService(
        settings=get_settings(), repo=get_repository(), cache=get_cache(),
    )


@lru_cache(maxsize=1)
def get_feature_extraction_service():
    """피처(.pt) 없는 신규 PACS 슬라이드용 온디맨드 패치추출+인코딩 서비스."""
    from wsi.infra.feature_extraction import FeatureExtractionService
    return FeatureExtractionService(
        settings=get_settings(), repo=get_repository(),
        pacs_tiles_provider=get_pacs_tile_source,
    )


@lru_cache(maxsize=1)
def get_cache_service():
    """슬라이드 준비(로컬 SVS 확인/워밍) — torch 비의존(목록/타일 계층과 동일하게 가벼움)."""
    from wsi.services.cache_service import SlideCacheService
    return SlideCacheService(repo=get_repository(), tiles=get_tile_source())


# ── PACS (선택) — 지연 생성. 자격증명 없으면 첫 호출 시 PacsDisabledError. ──
@lru_cache(maxsize=1)
def get_pacs_gateway():
    from wsi.pacs.gateway import PacsGateway
    return PacsGateway(get_settings())


@lru_cache(maxsize=1)
def get_pacs_repository():
    from wsi.pacs.repository import PacsCaseRepository
    return PacsCaseRepository(get_pacs_gateway())


@lru_cache(maxsize=1)
def get_pacs_tile_source():
    from wsi.pacs.tile_source import PacsDicomTileSource
    return PacsDicomTileSource(get_settings(), get_pacs_gateway())


@lru_cache(maxsize=1)
def get_job_store():
    from wsi.pacs.jobs import JobStore
    return JobStore()
