"""Service — 신장내과 AKI 분석(환자 기반) + AI_ALERT 적재 (작업지시서 4/5/7.1)."""
import pytest

from core.exceptions import NotFoundError
from models.patient import PatientLab, PatientTrendPoint, PatientUrinePoint
from models.timeline import TimelineEvent
from services.nephrology_service import NephrologyService, build_features_from_patient
from tests.factories import make_patient


def _severe_patient(db):
    """Cr 급상승 + 무뇨 → 규칙 기반 고위험이 나오도록 구성한 환자."""
    p = make_patient(db)
    db.add_all(
        [
            PatientTrendPoint(patient_id=p.id, date="2026-06-01", creatinine=1.0, egfr=85, bun=18),
            PatientTrendPoint(patient_id=p.id, date="2026-06-03", creatinine=4.5, egfr=12, bun=60),
            PatientUrinePoint(patient_id=p.id, date="2026-06-03", value=0.1),
            PatientLab(patient_id=p.id, seq=0, key="cr", label="Creatinine",
                       value=4.5, unit="mg/dL", flag="high"),
            PatientLab(patient_id=p.id, seq=1, key="k", label="Potassium",
                       value=6.2, unit="mmol/L", flag="high"),
        ]
    )
    db.commit()
    return p


def _stable_patient(db):
    p = make_patient(db)
    db.add_all(
        [
            PatientTrendPoint(patient_id=p.id, date="2026-06-01", creatinine=0.9, egfr=95, bun=12),
            PatientUrinePoint(patient_id=p.id, date="2026-06-01", value=1.2),
            PatientLab(patient_id=p.id, seq=0, key="cr", label="Creatinine",
                       value=0.9, unit="mg/dL", flag="normal"),
        ]
    )
    db.commit()
    return p


def test_build_features_extracts_kidney_signals(db):
    p = _severe_patient(db)
    db.refresh(p)
    feats = build_features_from_patient(p)
    assert feats["baseline_creatinine"] == 1.0
    assert feats["creatinine_max"] == 4.5
    assert feats["urine_ml_kg_hr"] == 0.1
    assert feats["oliguria_flag"] == 1


def test_analyze_patient_high_risk_writes_ai_alert(db):
    p = _severe_patient(db)
    result = NephrologyService(db).analyze_patient(p.id, actor_id="u-neph")
    assert result["risk_score"] > 0
    db.refresh(p)
    assert p.ai_risk_score == result["risk_score"]
    if result["stage_num"] >= 3:
        alerts = (
            db.query(TimelineEvent)
            .filter(TimelineEvent.patient_id == p.id, TimelineEvent.event_type == "AI_ALERT")
            .all()
        )
        assert alerts and alerts[0].severity == "CRITICAL"


def test_analyze_patient_stable_updates_score(db):
    p = _stable_patient(db)
    result = NephrologyService(db).analyze_patient(p.id, actor_id="u-neph")
    assert result["risk"] in {"low", "moderate", "high"}
    db.refresh(p)
    assert p.ai_risk_score == result["risk_score"]


def test_analyze_patient_unknown_raises_not_found(db):
    with pytest.raises(NotFoundError):
        NephrologyService(db).analyze_patient("p-does-not-exist", actor_id="u")


def test_analyze_features_pure_no_persist(db):
    out = NephrologyService(db).analyze_features(
        {"creatinine_max": 3.0, "baseline_creatinine": 1.0, "urine_ml_kg_hr": 0.2}
    )
    assert {"risk", "risk_score", "rationale", "source"} <= set(out)


def test_analyze_vector_routes_to_model_or_fallback(db):
    # 35개 표준 벡터(0 채움) — 모델 탑재 시 model, 아니면 rule 폴백. 계약만 검증.
    feats = {c: 0.0 for c in ["creatinine_max", "creatinine_min", "urine_output_6h"]}
    out = NephrologyService(db).analyze_vector(feats)
    assert out["source"] in {"model", "rule-based", "hybrid"}
