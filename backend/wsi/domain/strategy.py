"""디스크립터 매핑 정책 — 모델 task(Banff 등급) → 프론트 지표/레이어 변환 규칙.

OCP: 새 디스크립터가 생기면 if 분기를 늘리지 말고 DESCRIPTORS 테이블에 한 줄 추가한다.
값/색/라벨/등급환산이 한 곳(데이터)로 모여 mapper 는 규칙을 '실행'만 한다.
색·라벨은 프론트(WSIViewer LAYER_COLORS/TARGET_LABEL)와 정확히 일치시킨다.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DescriptorSpec:
    engine_task: str               # CdssEngine ORD 키
    metric_key: str                # 프론트 지표/레이어 키
    label: str                     # 한글 표시명(프론트 TARGET_LABEL)
    color: str                     # "r,g,b" (프론트 LAYER_COLORS)
    banff_edges: tuple[int, int, int]   # 등급 경계(%) → 표시값 환산용


# 모델이 실제 예측하는 3개 디스크립터만 노출(tubularInjury·artHyalinosis 는 미예측 → 계약 제외).
DESCRIPTORS: tuple[DescriptorSpec, ...] = (
    DescriptorSpec("fibrosis",     "fibrosisRatio", "간질 섬유화", "96,165,250", (5, 25, 50)),
    DescriptorSpec("atrophy",      "atrophyRatio",  "세뇨관 위축", "251,191,36", (5, 25, 50)),
    DescriptorSpec("inflammation", "inflammation",  "간질 염증",   "74,222,128", (10, 25, 50)),
)


def grade_to_percent(grade: int, edges: tuple[int, int, int]) -> float:
    """Banff 등급(0-3) → 대표 표시값(%). 각 등급 구간의 중앙값, 최고등급은 상한 가산."""
    e1, e2, e3 = edges
    table = {0: e1 / 2.0, 1: (e1 + e2) / 2.0, 2: (e2 + e3) / 2.0, 3: min(100.0, e3 * 1.3)}
    return round(table.get(int(grade), 0.0), 1)
