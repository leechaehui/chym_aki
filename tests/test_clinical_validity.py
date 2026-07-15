"""Validation — 임상 타당성(단조성) 검증 (작업지시서 4.2).

배포 예측기가 Cr↑ / eGFR↓ / 소변량↓ 에 대해 위험을 비감소로 반영하는지 강제한다.
이 테스트가 깨지면 '모델이 임상 신호를 무시'한다는 회귀이므로 반드시 실패해야 한다.
"""
from ai_draft.aki_model import RuleBasedAkiPredictor
from validator.clinical import (
    check_creatinine_monotonic,
    check_egfr_monotonic,
    check_urine_monotonic,
    run_clinical_checks,
)


def test_creatinine_increases_risk():
    chk = check_creatinine_monotonic(RuleBasedAkiPredictor())
    assert chk.passed, chk.risk_scores
    # 시작보다 끝(3.5배)이 더 높아야 한다(엄격 상승 확인).
    assert chk.risk_scores[-1] > chk.risk_scores[0]


def test_egfr_decline_increases_risk():
    chk = check_egfr_monotonic(RuleBasedAkiPredictor())
    assert chk.passed, chk.risk_scores
    assert chk.risk_scores[-1] > chk.risk_scores[0]


def test_oliguria_increases_risk():
    chk = check_urine_monotonic(RuleBasedAkiPredictor())
    assert chk.passed, chk.risk_scores
    assert chk.risk_scores[-1] > chk.risk_scores[0]


def test_run_all_clinical_checks_pass():
    checks = run_clinical_checks(RuleBasedAkiPredictor())
    assert len(checks) == 3
    assert all(c.passed for c in checks)
