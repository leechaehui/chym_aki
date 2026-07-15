"""병리 서비스 — WSI 분석 결과 조회(읽기) + 병리 보고서(소견/진단) 저장(쓰기)."""
import json
from datetime import datetime

from sqlalchemy.orm import Session

from core.exceptions import NotFoundError
from models.pathology import PathologyResult
from repositories.pathology_repository import PathologyRepository
from schemas.pathology import (
    DetectedLayerOut,
    PathologyReportIn,
    PathologyReportOut,
    PathologyResultOut,
    QuantMetricOut,
)


def _to_out(row: PathologyResult) -> PathologyResultOut:
    return PathologyResultOut(
        consult_id=row.consult_id,
        stain=row.stain,
        image_url=row.image_url,
        layers=[DetectedLayerOut(**layer) for layer in json.loads(row.layers_json or "[]")],
        metrics=[QuantMetricOut(**m) for m in json.loads(row.metrics_json or "[]")],
        report=PathologyReportOut(
            findings=row.report_findings,
            diagnosis=row.report_diagnosis,
            status=row.report_status,
            updated_at=row.report_updated_at,
        ),
    )


class PathologyService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = PathologyRepository(db)

    def list(self) -> list[PathologyResultOut]:
        return [_to_out(r) for r in self.repo.list_all_ordered()]

    def get_by_consult(self, consult_id: str) -> PathologyResultOut:
        row = self.repo.get_by_consult(consult_id)
        if not row:
            raise NotFoundError("해당 협진의 병리 결과를 찾을 수 없습니다.")
        return _to_out(row)

    def save_report(self, consult_id: str, data: PathologyReportIn) -> PathologyResultOut:
        """'임시 저장'/'판독 완료' — 소견/진단/상태를 저장(없으면 새로 생성)."""
        try:
            row = self.repo.get_or_create(consult_id)
            row.report_findings = data.findings
            row.report_diagnosis = data.diagnosis
            row.report_status = data.status
            row.report_updated_at = datetime.now().isoformat(timespec="seconds")
            self.db.commit()
            return _to_out(row)
        except Exception:
            self.db.rollback()
            raise
