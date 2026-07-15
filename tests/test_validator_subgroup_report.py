"""Unit — validator.subgroup / validator.report (V&V 모듈 자체 회귀 안전)."""
import numpy as np

from validator.report import ValidationReport
from validator.subgroup import (
    feature_ablation,
    missing_sensitivity,
    quantile_bands,
    subgroup_metrics,
)


def test_subgroup_metrics_splits_by_group():
    y = [0, 1, 0, 1, 0, 1, 0, 1]
    p = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]
    groups = ["A", "A", "A", "A", "B", "B", "B", "B"]
    out = subgroup_metrics(y, p, groups, min_n=2)
    names = {r.name for r in out}
    assert names == {"A", "B"}
    assert all(0.0 <= r.auroc <= 1.0 for r in out)


def test_subgroup_metrics_drops_small_groups():
    y = [0, 1, 0, 1]
    p = [0.1, 0.9, 0.2, 0.8]
    groups = ["A", "A", "B", "B"]
    out = subgroup_metrics(y, p, groups, min_n=3)  # 각 그룹 2개 < 3
    assert out == []


def test_quantile_bands_three_terciles():
    vals = list(range(30))
    bands = quantile_bands(vals)
    assert set(bands) == {"Q1(낮음)", "Q2(중간)", "Q3(높음)"}


def test_missing_sensitivity_detects_degradation():
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, n)
    # 결측 적은 그룹은 신호 강하게, 많은 그룹은 무작위로.
    miss = np.array([0] * (n // 2) + [10] * (n // 2))
    p = np.where(miss == 0, y * 0.8 + 0.1, rng.random(n))
    res = missing_sensitivity(y, p, miss)
    assert res.low_missing_auroc > res.high_missing_auroc
    assert res.delta > 0


def test_feature_ablation_reports_drop():
    rng = np.random.default_rng(1)
    n = 300
    X = rng.random((n, 3))
    y = (X[:, 0] > 0.5).astype(int)  # 0번 피처가 라벨 결정

    def predict(Xm):
        return Xm[:, 0]  # 0번 피처를 점수로

    out = feature_ablation(predict, X, y, {"f0": 0, "f1": 1}, {"중요": ["f0"], "무관": ["f1"]})
    by = {r.feature_group: r for r in out}
    assert by["중요"].auroc_drop > by["무관"].auroc_drop  # 중요 피처 제거가 더 큰 저하


def test_validation_report_markdown_and_json():
    rep = ValidationReport(title="테스트 리포트")
    rep.dataset = {"n_rows": 10}
    rep.add("metric_section", {
        "n": 10, "positives": 4, "prevalence": 0.4, "auroc": 0.9, "auprc": 0.8,
        "brier": 0.1, "ece": 0.05,
        "calibration": [{"bin": "[0.0,0.1)", "count": 5, "mean_predicted": 0.05, "observed_rate": 0.0}],
    })
    rep.add("list_section", [{"a": 1}])
    rep.known_failures.append("샘플 실패")

    md = rep.to_markdown()
    assert "# 테스트 리포트" in md
    assert "AUROC 0.9000" in md
    assert "Known failure cases" in md

    import json
    data = json.loads(rep.to_json())
    assert data["title"] == "테스트 리포트"
    assert data["known_failures"] == ["샘플 실패"]
