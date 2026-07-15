"""도메인 포트(추상) — 의존성 역전(DIP)의 경계.

서비스/도메인은 '구현'이 아니라 '이 추상'에만 의존한다. 구현체(infra)는 factory 가 주입.
인터페이스 분리(ISP): 거대한 단일 인터페이스 대신 역할별로 쪼갠다.
  - SlideRepository  : 데이터 접근(임베딩 bag·좌표·SVS 위치)
  - InferenceEngine  : 모델 추론(QC·라우팅·ensemble·attention)
  - TileSource       : WSI 타일(DZI/썸네일) 생성
이 셋은 서로를 모른다 → 한 축의 변경이 다른 축으로 번지지 않음(failure/변경 격리).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from wsi.schemas.wsi import WsiSlide


@dataclass(frozen=True)
class StainCoords:
    """한 stain 의 패치 공간좌표 — 오버레이 정규화용(없으면 None).
    환자당 물리 SVS 가 여러 개일 수 있어 각 patch 에 slide_file(파일명)을 함께 들고 다니고,
    화면에 실제로 띄운 파일(displayed_slide)과 다른 patch 는 mapper 가 걸러낸다(가짜 좌표 방지)."""
    coords: list[tuple[float, float, float, str]]  # (tile_x, tile_y, out_size, slide_file)
    slide_w: int
    slide_h: int
    displayed_slide: str | None


@dataclass(frozen=True)
class EngineOutput:
    """추론 1회의 도메인 결과 — 임상 리포트 + 설명가능성(패치 attention)."""
    report: dict                              # CdssEngine 임상 리포트(비진단)
    n_patches: int
    stain_ids: list[str]                      # 모델 행 순서의 stain 라벨(len N)
    task_attn: dict[str, list[float]]         # task -> 패치별 attention(len N, ensemble 평균)


@runtime_checkable
class SlideRepository(Protocol):
    def list_slides(self, stain: str) -> list[WsiSlide]: ...
    def load_bag(self, slide_id: str) -> dict[str, np.ndarray]: ...
    def patch_coords(self, slide_id: str) -> dict[str, StainCoords]: ...
    def svs_path(self, slide_id: str, stain: str) -> Path | None: ...


@runtime_checkable
class InferenceEngine(Protocol):
    @property
    def model_label(self) -> str: ...
    @property
    def variant_info(self) -> dict: ...   # {requested, resolved, status, reason} — 캐시/A/B 실행기준
    def analyze(self, bag: dict[str, np.ndarray], *, slide_id: str) -> EngineOutput: ...


@runtime_checkable
class TileSource(Protocol):
    def dzi_descriptor(self, svs: Path) -> str: ...                 # DZI XML
    def dzi_tile(self, svs: Path, level: int, col: int, row: int) -> bytes: ...
    def thumbnail(self, svs: Path, size: int) -> bytes: ...


@runtime_checkable
class ResultCache(Protocol):
    """분석 결과 캐시 — 서비스의 상태를 외부화(stateless 서비스). 구현은 파일/DB/object storage 교체 가능."""
    def get(self, stain: str, slide_id: str, version: str = "") -> dict | None: ...
    def set(self, stain: str, slide_id: str, payload: dict, version: str = "") -> None: ...
