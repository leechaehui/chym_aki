"""
CDSS 경로 단일 소스 로더 — 절대경로 하드코딩 제거용.

원칙: 코드는 ROOT를 박지 말고 이 모듈로 경로를 받는다.
    from mil.cdss_paths import CFG, data_root, p
    emb = p("embeddings")              # config.paths.embeddings
    root = data_root()                 # 마이그레이션 상태에 따른 데이터 루트

마이그레이션 완료(config.migration_status=="done") 전엔 data_root=config.json[data_root](현재 트리),
완료 후엔 cdss_core(기본: data/cdss_core, 리포 루트 기준)를 반환 → 코드 변경 없이 전환.
"""
import json
from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent          # Pathology_model
_REPO = _PKG.parent                                     # 리포 루트(C:/team/chym_aki)
_CFG_PATH = _PKG / "config.json"
CFG = json.loads(_CFG_PATH.read_text(encoding="utf-8"))


def _resolve(val: str) -> Path:
    """상대경로는 리포 루트 기준으로 해석(머신 비의존). 절대경로(예: 원본 SVS on D:)는 그대로 사용."""
    pth = Path(val)
    return pth if pth.is_absolute() else (_REPO / pth)


def data_root() -> Path:
    key = "cdss_core" if CFG.get("migration_status") == "done" else "data_root"
    return _resolve(CFG[key])


def p(name: str) -> Path:
    """config.paths[name] 절대경로 반환(상대경로는 리포 루트 기준)."""
    return _resolve(CFG["paths"][name])


def cohort_raw(cohort: str) -> Path:
    return _resolve(CFG["paths"]["data_raw"]) / cohort
