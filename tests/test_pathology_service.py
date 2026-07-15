"""Service — 병리 결과 조회(읽기) (작업지시서 7.1)."""
import uuid

import pytest

from core.exceptions import NotFoundError
from models.pathology import PathologyResult
from services.pathology_service import PathologyService


def _make_result(db, **over) -> PathologyResult:
    row = PathologyResult(
        consult_id=over.get("consult_id", f"c-{uuid.uuid4().hex[:8]}"),
        stain=over.get("stain", "PAS"),
        image_url=over.get("image_url", "http://example/wsi.svs"),
        layers_json="[]",
        metrics_json="[]",
        report_findings=over.get("report_findings", "사구체 경화 소견"),
        report_diagnosis=over.get("report_diagnosis", "IgA 신병증"),
        report_status=over.get("report_status", "final"),
        report_updated_at=None,
    )
    db.add(row)
    db.commit()
    return row


def test_get_by_consult_returns_result(db):
    row = _make_result(db)
    out = PathologyService(db).get_by_consult(row.consult_id)
    assert out.consult_id == row.consult_id
    assert out.report.diagnosis == "IgA 신병증"


def test_get_by_consult_unknown_raises_not_found(db):
    with pytest.raises(NotFoundError):
        PathologyService(db).get_by_consult("c-does-not-exist")


def test_list_includes_created(db):
    row = _make_result(db)
    results = PathologyService(db).list()
    assert any(r.consult_id == row.consult_id for r in results)
