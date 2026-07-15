"""Unit — AKI 예측기 계약(LSP) + 하이브리드 라우팅."""
from ai_draft.aki_model import (
    HybridAkiPredictor,
    ModelAkiPredictor,
    RuleBasedAkiPredictor,
    get_aki_predictor,
)

RESULT_KEYS = {
    "risk", "risk_label", "stage", "stage_num", "risk_score",
    "p_non_aki", "p_stage1", "p_stage2_plus", "rationale", "source",
}


def test_rule_based_result_contract():
    out = RuleBasedAkiPredictor().predict({"creatinine_max": 3.0, "baseline_creatinine": 1.0})
    assert RESULT_KEYS <= set(out)
    assert 0 <= out["risk_score"] <= 100
    probs = out["p_non_aki"] + out["p_stage1"] + out["p_stage2_plus"]
    assert abs(probs - 1.0) < 1e-6


def test_rule_based_empty_input_is_conservative():
    out = RuleBasedAkiPredictor().predict({})
    assert out["risk"] == "low"
    assert out["stage_num"] == 0


def test_probabilities_normalized_for_severe_case():
    out = RuleBasedAkiPredictor().predict(
        {"creatinine_max": 4.5, "baseline_creatinine": 1.0, "urine_ml_kg_hr": 0.1}
    )
    assert out["stage_num"] >= 1
    assert out["p_stage2_plus"] > out["p_non_aki"]


def test_singleton_factory_returns_same_instance():
    assert get_aki_predictor() is get_aki_predictor()


def test_hybrid_routes_to_rule_on_low_coverage():
    """피처 커버리지가 낮으면(희소 EMR) 규칙 기반으로 라우팅(LSP 보존)."""
    rule = RuleBasedAkiPredictor()

    class _StubModel(ModelAkiPredictor):
        def __init__(self):  # 모델 로드 우회
            pass

        def predict(self, features):  # 호출되면 안 됨
            raise AssertionError("low-coverage 인데 모델이 호출됨")

    hybrid = HybridAkiPredictor(_StubModel(), rule, feature_cols=[f"f{i}" for i in range(35)])
    out = hybrid.predict({"creatinine_max": 2.0, "baseline_creatinine": 1.0})
    assert out["source"] == "rule-based"
    assert any("커버리지" in r for r in out["rationale"])
