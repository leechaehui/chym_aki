"""Service — CDSS Risk (작업지시서 6, 10): hybrid·breakdown·재현성·tier."""
from services.cdss_risk_service import WEIGHTS, CdssRiskService


class _Lab:
    def __init__(self, key, value):
        self.key, self.value = key, value


class _Pt:
    def __init__(self, value):
        self.creatinine = value
        self.date = "d"


class _Urine:
    def __init__(self, value):
        self.value = value


class _Patient:
    def __init__(self, labs=None, trend=None, urine=None):
        self.labs = labs or []
        self.trend = trend or []
        self.urine_output = urine or []


def _aki(p1=0.2, p23=0.6, source="model"):
    return {"p_stage1": p1, "p_stage2_plus": p23, "source": source, "risk_score": 80}


def test_score_has_all_four_components():
    risk = CdssRiskService().score(_Patient(), {}, [], _aki())
    comps = {c["component"] for c in risk["breakdown"]}
    assert comps == set(WEIGHTS)  # single-variable 금지 → 4개 모두


def test_weights_sum_to_one():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_score_is_reproducible_from_breakdown():
    p = _Patient(
        labs=[_Lab("cr", 4.0), _Lab("k", 6.5)],
        trend=[_Pt(1.0), _Pt(4.0)],
        urine=[_Urine(0.2)],
    )
    risk = CdssRiskService().score(p, {}, [], _aki())
    recomputed = min(1.0, sum(c["contribution"] for c in risk["breakdown"]))
    assert abs(recomputed - risk["risk_score"]) < 1e-6


def test_high_risk_triggers_nephrology_and_modal():
    p = _Patient(
        labs=[_Lab("cr", 4.5), _Lab("k", 6.8)],
        trend=[_Pt(1.0), _Pt(4.5)],
        urine=[_Urine(0.1)],
    )
    risk = CdssRiskService().score(p, [], [], _aki(p23=0.9))
    assert risk["tier"] == "HIGH"
    assert risk["alert"] == "modal"
    assert risk["nephrology_trigger"] is True


def test_low_risk_is_log_only():
    p = _Patient(labs=[_Lab("cr", 0.9)], trend=[_Pt(0.9)], urine=[_Urine(1.5)])
    risk = CdssRiskService().score(p, [], [], _aki(p1=0.0, p23=0.0))
    assert risk["tier"] == "LOW"
    assert risk["alert"] == "log"
    assert risk["nephrology_trigger"] is False


def test_every_component_is_explained():
    risk = CdssRiskService().score(_Patient(), {}, [], _aki())
    assert all(c["explanation"] for c in risk["breakdown"])  # 설명 없는 score 금지


def test_deterministic_same_input_same_score():
    p = _Patient(labs=[_Lab("cr", 3.0)], trend=[_Pt(1.0), _Pt(3.0)], urine=[_Urine(0.4)])
    r1 = CdssRiskService().score(p, [], [], _aki())
    r2 = CdssRiskService().score(p, [], [], _aki())
    assert r1["risk_score"] == r2["risk_score"]
    assert r1["breakdown"] == r2["breakdown"]
