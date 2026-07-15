"""AKI 진단 예측기 (Strategy + Factory).

최종 모델(final_aki_model.py)의 2-stage 융합을 백엔드에서 재현한다:
  Stage1 (Logistic Regression) : Non-AKI vs AKI  → p_aki
  Stage2 (LightGBM)            : Stage1 vs Stage2+3 → p_severe
  Soft fusion:
     P(Non-AKI)  = 1 - p_aki
     P(Stage1)   = p_aki * (1 - p_severe)
     P(Stage2+3) = p_aki * p_severe

설계:
- 학습된 .pkl 번들이 있으면 ModelAkiPredictor 사용.
- 없으면(또는 의존성 미설치) RuleBasedAkiPredictor(KDIGO 휴리스틱)로 폴백 → 항상 동작.
- 두 구현 모두 동일한 결과 dict 계약을 따른다(LSP).
"""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod

from core.config import settings


# 최종 3-class 라벨(final_aki_model.py 와 동일).
LABELS = {0: "Non-AKI", 1: "AKI Stage1", 2: "AKI Stage2+3"}


def _build_result(
    p_non_aki: float,
    p_stage1: float,
    p_stage23: float,
    rationale: list[str],
    source: str,
) -> dict:
    """3-class 확률 → 표준 결과 dict(서비스/스키마 공용 계약)."""
    probs = [p_non_aki, p_stage1, p_stage23]
    pred = int(max(range(3), key=lambda i: probs[i]))

    # 화면 표기용 위험 등급/단계 매핑.
    if pred == 2:
        risk, risk_label, stage, stage_num = "high", "고위험", "AKI Stage 2-3", 3
    elif pred == 1:
        risk, risk_label, stage, stage_num = "moderate", "중등도", "AKI Stage 1", 1
    else:
        risk, risk_label, stage, stage_num = "low", "안정", "Non-AKI", 0

    # 위험 점수(0–100): AKI 확률 가중 합(Stage2+3 에 더 큰 가중).
    risk_score = int(round(min(100.0, (p_stage1 * 60.0 + p_stage23 * 100.0))))

    return {
        "risk": risk,
        "risk_label": risk_label,
        "stage": stage,
        "stage_num": stage_num,
        "risk_score": risk_score,
        "p_non_aki": round(p_non_aki, 4),
        "p_stage1": round(p_stage1, 4),
        "p_stage2_plus": round(p_stage23, 4),
        "rationale": rationale,
        "source": source,
    }


class AkiPredictor(ABC):
    source: str = "base"

    @abstractmethod
    def predict(self, features: dict) -> dict:
        raise NotImplementedError


class HybridAkiPredictor(AkiPredictor):
    """모델/규칙 라우팅 예측기.

    학습 모델은 **표준화된 48h 전체 피처 벡터**(MIMIC 코호트 행)에서만 유효하다.
    앱 환자 분석은 소수의 원시 EMR 검사값만 제공하므로 KDIGO 규칙 기반이 의학적으로 옳다.
    → 입력 피처 커버리지로 두 전략을 안전하게 라우팅한다(LSP 보존).
    """

    source = "hybrid"

    def __init__(
        self,
        model_predictor: "ModelAkiPredictor",
        rule_predictor: "RuleBasedAkiPredictor",
        feature_cols: list[str],
        min_coverage: float = 0.6,
    ):
        self.model = model_predictor
        self.rule = rule_predictor
        self.feature_cols = feature_cols
        self.min_coverage = min_coverage

    def _coverage(self, features: dict) -> float:
        if not self.feature_cols:
            return 0.0
        present = sum(
            1 for c in self.feature_cols if features.get(c) not in (None, "")
        )
        return present / len(self.feature_cols)

    def predict(self, features: dict) -> dict:
        """커버리지 충분(표준화 전체벡터) → 모델, 아니면 규칙 기반."""
        coverage = self._coverage(features)
        if coverage >= self.min_coverage:
            return self.model.predict(features)
        result = self.rule.predict(features)
        result["rationale"].append(
            f"입력 피처 커버리지 {coverage:.0%} (<{self.min_coverage:.0%}) "
            "→ 학습 모델 대신 규칙 기반(KDIGO) 사용"
        )
        return result

    def predict_with_model(self, features: dict) -> dict:
        """표준화 전체 피처 벡터에 대해 학습 모델을 강제 적용(연구/배치)."""
        return self.model.predict(features)


# ----------------------------------------------------------
# 1) 학습 모델 기반 예측기
# ----------------------------------------------------------
class ModelAkiPredictor(AkiPredictor):
    """stage1(LR)+stage2(LGBM) .pkl 번들 기반 추론."""

    source = "model"

    def __init__(self, stage1_bundle: dict, stage2_bundle: dict):
        self._lr = stage1_bundle["model"]
        self._lgbm = stage2_bundle["model"]
        # 학습 시 사용한 피처 순서. 입력 dict 를 이 순서로 정렬한다.
        self._feature_cols: list[str] = stage1_bundle.get("feature_cols", [])

    def _vectorize(self, features: dict):
        import numpy as np

        # 누락 피처는 0.0 으로 채운다(보수적). 학습 피처 순서를 엄격히 따른다.
        row = [float(features.get(col, 0.0) or 0.0) for col in self._feature_cols]
        return np.array([row], dtype=float)

    def predict(self, features: dict) -> dict:
        x = self._vectorize(features)
        p_aki = float(self._lr.predict_proba(x)[0, 1])
        p_severe = float(self._lgbm.predict_proba(x)[0, 1])

        p_non_aki = 1.0 - p_aki
        p_stage1 = p_aki * (1.0 - p_severe)
        p_stage23 = p_aki * p_severe

        rationale = [
            f"Stage1(LR) P(AKI)={p_aki:.2f}",
            f"Stage2(LGBM) P(Stage2+3 | AKI)={p_severe:.2f}",
            "학습 모델 2-stage soft fusion 결과",
        ]
        return _build_result(p_non_aki, p_stage1, p_stage23, rationale, self.source)


# ----------------------------------------------------------
# 2) 규칙 기반(KDIGO) 폴백 예측기
# ----------------------------------------------------------
class RuleBasedAkiPredictor(AkiPredictor):
    """KDIGO 기준을 단순화한 휴리스틱 예측기(모델 부재 시 폴백).

    Cr 절대값/배수, 시간당 소변량/핍뇨, eGFR, 고칼륨/산증을 종합해
    severity 점수를 만들고 3-class 확률로 변환한다.
    """

    source = "rule-based"

    def predict(self, features: dict) -> dict:
        rationale: list[str] = []
        score = 0.0  # 0(정상) → 높을수록 중증

        cr_max = _f(features.get("creatinine_max"))
        baseline = _f(features.get("baseline_creatinine")) or _f(
            features.get("creatinine_min")
        )
        delta = _f(features.get("creatinine_delta"))
        egfr = _f(features.get("egfr"))
        uo = _f(features.get("urine_ml_kg_hr"))
        oliguria = features.get("oliguria_flag")
        k_max = _f(features.get("potassium_max"))
        hco3 = _f(features.get("bicarbonate_min"))

        # --- 1) Creatinine 배수(KDIGO 핵심) ---
        ratio = None
        if cr_max and baseline and baseline > 0:
            ratio = cr_max / baseline
        if ratio is not None:
            if ratio >= 3.0:
                score += 3.0
                rationale.append(f"Cr {ratio:.1f}배 상승 (KDIGO Stage 3 기준)")
            elif ratio >= 2.0:
                score += 2.0
                rationale.append(f"Cr {ratio:.1f}배 상승 (Stage 2)")
            elif ratio >= 1.5:
                score += 1.0
                rationale.append(f"Cr {ratio:.1f}배 상승 (Stage 1)")
        if delta and delta >= 0.3:
            score += 1.0
            rationale.append(f"48h Cr 증가 {delta:.1f} mg/dL (≥0.3)")
        if cr_max and cr_max >= 4.0:
            score += 1.5
            rationale.append(f"Cr 절대값 {cr_max:.1f} mg/dL (≥4.0)")

        # --- 2) eGFR ---
        if egfr:
            if egfr < 15:
                score += 2.0
                rationale.append(f"eGFR {egfr:.0f} (<15)")
            elif egfr < 30:
                score += 1.0
                rationale.append(f"eGFR {egfr:.0f} (<30)")

        # --- 3) 소변량/핍뇨 ---
        if uo is not None:
            if uo < 0.3:
                score += 2.0
                rationale.append(f"시간당 소변량 {uo:.2f} mL/kg/h (<0.3, 무뇨)")
            elif uo < 0.5:
                score += 1.0
                rationale.append(f"시간당 소변량 {uo:.2f} mL/kg/h (<0.5, 핍뇨)")
        if oliguria in (1, True, "1"):
            score += 0.5
            rationale.append("핍뇨 플래그 양성")

        # --- 4) 합병증(고칼륨/대사성 산증) — 중증도 가중 ---
        if k_max and k_max >= 6.0:
            score += 1.0
            rationale.append(f"고칼륨혈증 K {k_max:.1f} (≥6.0)")
        if hco3 and hco3 < 18:
            score += 0.5
            rationale.append(f"대사성 산증 HCO3 {hco3:.0f} (<18)")

        if not rationale:
            rationale.append("유효 입력 부족 — 보수적으로 Non-AKI 추정")

        # severity 점수 → 3-class 확률(단조 매핑).
        p_non_aki, p_stage1, p_stage23 = _score_to_probs(score)
        return _build_result(p_non_aki, p_stage1, p_stage23, rationale, self.source)


def _f(value) -> float | None:
    """안전 float 변환(None/빈값 → None)."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_to_probs(score: float) -> tuple[float, float, float]:
    """severity 점수를 3-class 확률로 변환.

    score 0 → Non-AKI 우세, 2 부근 → Stage1, 4+ → Stage2+3 우세.
    """
    # 경계: <1.0 Non-AKI / 1.0–2.5 Stage1 / >2.5 Stage2+3 (부드러운 가중).
    import math

    def bump(center: float, width: float) -> float:
        return math.exp(-((score - center) ** 2) / (2 * width**2))

    w_non = bump(0.0, 1.2)
    w_s1 = bump(2.0, 1.2)
    w_s23 = bump(4.5, 1.6)
    total = w_non + w_s1 + w_s23
    return w_non / total, w_s1 / total, w_s23 / total


# ----------------------------------------------------------
# Factory
# ----------------------------------------------------------
_singleton: AkiPredictor | None = None


def get_aki_predictor() -> AkiPredictor:
    """AKI 예측기 싱글턴 생성.

    학습 번들 로드 성공 시 ModelAkiPredictor, 실패 시 RuleBasedAkiPredictor.
    프로세스당 1회만 모델을 로드한다.
    """
    global _singleton
    if _singleton is not None:
        return _singleton

    stage1_path = settings.aki_model_path / settings.aki_stage1_file
    stage2_path = settings.aki_model_path / settings.aki_stage2_file

    if stage1_path.exists() and stage2_path.exists():
        try:
            with open(stage1_path, "rb") as f:
                stage1_bundle = pickle.load(f)
            
            # scikit-learn 버전 호환성 패치 (multi_class 속성 유실 해결)
            lr = stage1_bundle.get("model")
            if lr and not hasattr(lr, "multi_class"):
                lr.multi_class = "auto"
                
            with open(stage2_path, "rb") as f:
                stage2_bundle = pickle.load(f)
            model_predictor = ModelAkiPredictor(stage1_bundle, stage2_bundle)
            # 모델은 표준화 전체벡터에만 유효 → 규칙 기반과 하이브리드로 묶어 라우팅.
            _singleton = HybridAkiPredictor(
                model_predictor,
                RuleBasedAkiPredictor(),
                stage1_bundle.get("feature_cols", []),
            )
            return _singleton
        except Exception:
            # 번들 포맷/의존성 문제 → 규칙 기반으로 안전 폴백.
            pass

    _singleton = RuleBasedAkiPredictor()
    return _singleton
