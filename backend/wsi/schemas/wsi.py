"""WSI API 계약(DTO) — 프론트 src/types/wsi.ts 와 1:1 대응.

이 모듈은 '경계의 언어'다. 도메인/엔진의 내부 표현(Banff 등급·CdssEngine 리포트)이
바뀌어도 이 계약은 mapper 를 통해서만 변한다 → 프론트와의 결합을 한 곳에 격리.
필드명은 프론트 camelCase(updatedAt 등) 그대로 맞춘다.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# PAS 추가: ctranspath 인코더는 stain-agnostic, ordinal 모델은 STAIN_KEEP(HE/PAS/MT/Silver)로 학습됨.
WsiStain = Literal["HE", "MT", "PAS"]


class WsiSlide(BaseModel):
    slide_id: str
    case_code: str            # 화면 표시용 케이스 코드(manifest 는 patient_id 와 동일)
    stain: WsiStain
    has_features: bool        # 임베딩 보유 → /analyze 가능
    cached: bool              # 로컬 SVS 보유(manifest) 또는 PACS DICOM 캐시 보유 → 뷰어 즉시 가능(미보유 시 프론트 '↓' 표시)
    is_pacs: bool = False     # PACS 출처 → pacsDziUrl 사용(로컬 SVS 불필요). PACS-only 라 통상 True.
    description: str | None = None


class WsiSlideList(BaseModel):
    stain: WsiStain
    slides: list[WsiSlide]
    total: int


class WsiCacheStatus(BaseModel):
    """슬라이드 준비(캐시) 상태 — 프론트 CacheStatus 유니온과 대응.

    manifest 슬라이드의 SVS 는 로컬에 있어 ready/error 만 사용한다.
    downloading/not_started 필드는 PACS 등 원격 소스 확장용(현재 미사용) — 응답에서 None 은 제외.
    """
    status: Literal["ready", "downloading", "not_started", "error"]
    downloaded_mb: float | None = None
    step: str | None = None
    message: str | None = None


class WsiMetric(BaseModel):
    key: str          # fibrosisRatio | atrophyRatio | inflammation
    label: str        # 한글 표시명
    value: float      # 표시값(% 환산)
    unit: str
    raw: float        # 원시 Banff 등급(0-3)
    # descriptor별 보정 신뢰도(0~1) — CDSS(PAS) 의 calibratedConfidence.
    # ABMIL(HE/MT) 은 회귀라 신뢰도 개념이 없어 None.
    confidence: float | None = None


class WsiLayer(BaseModel):
    key: str
    label: str
    color: str        # "r,g,b"
    count: int | None
    visible: bool


class WsiReport(BaseModel):
    findings: str
    diagnosis: str    # 비진단 신뢰도 해석(확정판정 아님)
    status: str       # ALLOW | ABSTAIN
    updatedAt: str | None
    # 슬라이드 단위 불확실성(CORAL margin, 0~1·낮을수록 확신) — CDSS(PAS) 전용, ABMIL 은 None.
    uncertainty: float | None = None


class WsiHeatmapCell(BaseModel):
    patch_idx: int
    weight: float


class WsiAttnOverlay(BaseModel):
    cx: float                    # 슬라이드 너비 기준 정규화 중심 x(0~1)
    cy: float
    r: float
    weight: float                # attention 가중치(0~1)
    contrib: dict[str, float]    # task별 패치 기여(정규화)
    px: int = 0                  # 원본 슬라이드 level 0 기준 좌상단 x 픽셀 좌표
    py: int = 0                  # 원본 슬라이드 level 0 기준 좌상단 y 픽셀 좌표
    psize: int = 0               # 원본 슬라이드 level 0 기준 패치 픽셀 크기


class WsiTissue(BaseModel):
    id: int
    x: int
    y: int
    w: int
    h: int
    area: int


class WsiAnalysisResult(BaseModel):
    stain: WsiStain
    slide_id: str
    model_label: str
    metrics: list[WsiMetric]
    layers: list[WsiLayer]
    report: WsiReport
    heatmap: list[WsiHeatmapCell]
    attn_overlays: list[WsiAttnOverlay]
    n_patches: int
    tissues: list[WsiTissue] = []
    slide_w: int = 0
    slide_h: int = 0



class AnalyzeRequest(BaseModel):
    slide_id: str
    stain: WsiStain
    use_cache: bool = True
