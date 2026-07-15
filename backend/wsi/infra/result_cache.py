"""ResultCache 구현 — 분석 결과를 파일(JSON)로 외부화.

서비스를 stateless 로 유지(상태는 여기). 캐시 손상/부재는 None 반환 → 서비스가 재계산(fallback).
저장소가 바뀌어도(DB·object storage) 이 구현만 교체(DIP).
"""
from __future__ import annotations

import json

from wsi.core.config import WsiSettings


class FileResultCache:
    def __init__(self, settings: WsiSettings):
        self._dir = settings.cache_dir

    def _path(self, stain: str, slide_id: str, version: str = ""):
        safe = slide_id.replace("/", "_").replace("\\", "_")
        vsfx = f"__{version}" if version else ""   # versioned: model_variant__expl__norm (A/B 오염 방지)
        return self._dir / f"{stain}__{safe}{vsfx}.json"

    def get(self, stain: str, slide_id: str, version: str = "") -> dict | None:
        p = self._path(stain, slide_id, version)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None                       # 손상 캐시 → 무시하고 재계산 유도

    def set(self, stain: str, slide_id: str, payload: dict, version: str = "") -> None:
        self._path(stain, slide_id, version).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
