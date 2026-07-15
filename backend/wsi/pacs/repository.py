"""PACS 슬라이드 목록 — 게이트웨이의 케이스를 뷰어용 슬라이드로 매핑.

WSI(병리 슬라이드)는 modality=SM(Slide Microscopy). 분석(임베딩)은 PACS 범위 밖 → 목록/뷰어만 제공.
"""
from __future__ import annotations

from wsi.pacs.gateway import PacsGateway


class PacsCaseRepository:
    def __init__(self, gateway: PacsGateway):
        self._gw = gateway

    def list_wsi_cases(self) -> list[dict]:
        """WSI(SM) 케이스 목록 — 프론트 PACS 브라우저용 요약."""
        rows = []
        for c in self._gw.list_cases(modality="SM"):
            rows.append({
                "case_id": c.get("id"),
                "case_code": c.get("case_code"),
                "project_code": c.get("project_code"),
                "study_uid": c.get("study_uid"),
                "description": c.get("description"),
                "created_at": c.get("created_at"),
            })
        return rows

    def manifest(self, case_id: str) -> dict:
        return self._gw.get_manifest(case_id)
