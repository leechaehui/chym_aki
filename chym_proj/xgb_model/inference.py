"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
xgb_model/inference.py  —  XGBoost 추론 엔진 (경로 수정)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

역할:
  학습 완료된 모델을 로드하고 단일 환자 또는 배치 환자에 대한
  AKI 발생 확률(0~1)을 예측한다.

  SCR-06(AI 위험도 점수)과 SCR-07(시계열 차트)의 ML 기여값을 제공한다.

사용 방법:
  # 단일 환자 (API 호출 시)
  engine = AKIInferenceEngine()
  result = engine.predict_single(stay_id=12345, db_session=session)

  # 배치 (스케줄러에서 전체 ICU 환자 갱신 시)
  results = engine.predict_batch_patients(stay_ids=[100, 200, 300], db_session=session)

모델 파일 경로:
  xgb_model/model/xgb_aki.json        ← 자동 감지 (절대경로)
  xgb_model/model/threshold.txt       ← 자동 감지 (절대경로)
  xgb_model/model/feature_names.csv   ← 자동 감지 (절대경로)
  xgb_model/model/label_encoders.pkl  ← 자동 감지 (절대경로)

설계:
  - AKIInferenceEngine 싱글턴 패턴: 앱 시작 시 모델을 1회 로드하고 메모리에 유지.
  - 모델 파일 없음 → ModelNotReadyError 발생 → 백엔드에서 규칙 기반 fallback 사용.
  - predict() 결과는 InferenceResult 데이터클래스로 반환 (dict 대신 타입 안전성 확보).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from preprocessing import (
    load_label_encoders,
    load_feature_names,
    preprocess_for_inference,
)

logger = logging.getLogger("aki_cdss.inference")


# ─────────────────────────────────────────────────────────────────────────────
# 예외 클래스
# ─────────────────────────────────────────────────────────────────────────────

class ModelNotReadyError(Exception):
    """모델 파일이 없거나 로드 실패 시 발생."""
    pass


class InferenceError(Exception):
    """예측 실행 중 오류 발생 시."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# 결과 데이터클래스
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    """단일 환자 추론 결과.

    SCR-06 API 응답과 SCR-07 시계열 데이터포인트에 사용된다.

    Attributes:
        stay_id:          ICU 체류 ID
        aki_probability:  AKI 발생 확률 (0.0~1.0)
        aki_probability_pct: 화면 표시용 퍼센트 (0~100)
        is_high_risk:     threshold 초과 여부 (SCR-06 빨간 배경 기준)
        threshold:        사용된 분류 임계값
        model_version:    모델 파일 경로 (모니터링·감사 추적용)
        feature_values:   예측에 사용된 피처값 (디버깅·설명 가능성)
        missing_features: 누락된 피처 수 (신뢰도 지표)
    """
    stay_id:             int
    aki_probability:     float
    aki_probability_pct: int
    is_high_risk:        bool
    threshold:           float
    model_version:       str
    feature_values:      dict = field(default_factory=dict)
    missing_features:    int  = 0


@dataclass
class BatchInferenceResult:
    """배치 추론 결과 (스케줄러용)."""
    results:        list[InferenceResult]
    n_total:        int
    n_high_risk:    int
    n_failed:       int              # 예측 실패 환자 수
    failed_stay_ids:list[int] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# 추론 엔진 (싱글턴)
# ─────────────────────────────────────────────────────────────────────────────

class AKIInferenceEngine:
    """XGBoost AKI 예측 추론 엔진.

    앱 시작 시 1회 인스턴스를 생성하고 FastAPI의 lifespan 또는 모듈 레벨에서
    싱글턴으로 사용한다.

    사용 예:
        # main.py 또는 scr06_ai_risk_score.py 상단에서
        aki_engine = AKIInferenceEngine()

        # SCR-06 API에서
        result = aki_engine.predict_single(stay_id, df_patient)
    """

    def __init__(self) -> None:
        self._model:         Optional[xgb.XGBClassifier]       = None
        self._feature_names: Optional[list[str]]               = None
        self._encoders:      Optional[dict]                     = None
        self._threshold:     float                              = 0.5

        # ── 경로 설정 (절대 경로) ──────────────────────────────────────
        # inference.py가 있는 xgb_model 폴더를 기준으로 절대 경로 설정
        base_dir = os.path.dirname(os.path.abspath(__file__))  # xgb_model/ 폴더
        model_dir = os.path.join(base_dir, "model")              # xgb_model/model/ 폴더

        # 환경변수로 재정의 가능하지만, 기본값은 절대 경로
        self._model_path:    str = os.getenv(
            "XGB_MODEL_PATH",
            os.path.join(model_dir, "xgb_aki.json")
        )
        self._threshold_path:str = os.getenv(
            "XGB_THRESHOLD_PATH",
            os.path.join(model_dir, "threshold.txt")
        )
        self._feature_path:  str = os.path.join(model_dir, "feature_names.csv")
        self._encoder_path:  str = os.path.join(model_dir, "label_encoders.pkl")
        self._is_ready:      bool = False

        logger.info(f"[추론 엔진] 모델 경로: {self._model_path}")

        # 초기화 시 자동 로드 시도
        self._try_load()

    # ── 모델 로딩 ──────────────────────────────────────────────────────────

    def _try_load(self) -> None:
        """모델·피처·인코더·임계값 파일을 로드한다.

        파일이 없으면 ModelNotReadyError 없이 조용히 실패하고 is_ready=False 유지.
        이후 predict() 호출 시 ModelNotReadyError를 발생시킨다.
        """
        try:
            self._load_model()
            self._load_artifacts()
            self._is_ready = True
            logger.info(
                f"[추론 엔진] ✅ 로드 완료 — "
                f"피처 {len(self._feature_names)}개, "
                f"임계값 {self._threshold:.3f}"
            )
        except FileNotFoundError as e:
            logger.warning(f"[추론 엔진] 모델 파일 없음 ({e}). 규칙 기반 fallback 사용.")
            self._is_ready = False

    def _load_model(self) -> None:
        """XGBoost 모델 JSON 파일을 로드한다."""
        if not os.path.exists(self._model_path):
            raise FileNotFoundError(f"모델 파일 없음: {self._model_path}")
        logger.info(f"[추론 엔진] 모델 로드 중: {self._model_path}")
        self._model = xgb.XGBClassifier()
        self._model.load_model(self._model_path)
        logger.info(f"[추론 엔진] 모델 로드 완료")

    def _load_artifacts(self) -> None:
        """피처 목록·인코더·임계값을 로드한다."""
        # 피처 목록
        self._feature_names = load_feature_names()
        if self._feature_names is None:
            raise FileNotFoundError(f"feature_names.csv 없음: {self._feature_path}")
        logger.info(f"[추론 엔진] 피처 {len(self._feature_names)}개 로드")

        # 인코더 (없으면 고정 딕셔너리 fallback 사용)
        self._encoders = load_label_encoders()
        if self._encoders is None:
            logger.warning("label_encoders.pkl 없음 — 고정 딕셔너리 사용")

        # 임계값
        if os.path.exists(self._threshold_path):
            with open(self._threshold_path) as f:
                self._threshold = float(f.read().strip())
            logger.info(f"[추론 엔진] 임계값 로드: {self._threshold:.3f}")
        else:
            logger.warning("threshold.txt 없음 — 기본값 0.5 사용")
            self._threshold = 0.5

    def reload(self) -> None:
        """모델 파일이 업데이트됐을 때 핫 리로드한다.

        모델 재학습 후 API 서버 재시작 없이 새 모델을 반영할 때 사용.
        """
        logger.info("[추론 엔진] 핫 리로드 시작 ...")
        self._is_ready = False
        self._try_load()

    @property
    def is_ready(self) -> bool:
        """모델이 예측 준비 상태인지 여부."""
        return self._is_ready

    # ── 피처 추출 (DB 조회) ────────────────────────────────────────────────

    def _fetch_patient_features_from_db(
        self,
        stay_id: int,
        db_session,         # SQLAlchemy Session 또는 execute_query 호환 객체
    ) -> Optional[pd.DataFrame]:
        """cdss_master_features에서 단일 환자의 피처 벡터를 조회한다.

        Args:
            stay_id:    ICU 체류 ID
            db_session: DB 세션 (execute_query 함수 또는 SQLAlchemy Session)

        Returns:
            1행짜리 pd.DataFrame, 환자 없으면 None
        """
        from db import execute_query  # 백엔드 공통 쿼리 헬퍼

        rows = execute_query(
            "SELECT * FROM aki_project.cdss_master_features WHERE stay_id = :sid",
            {"sid": stay_id}
        )
        if not rows:
            return None
        return pd.DataFrame(rows)

    def _fetch_batch_features_from_db(
        self,
        stay_ids: list[int],
    ) -> pd.DataFrame:
        """여러 환자의 피처 벡터를 한 번의 쿼리로 조회한다 (배치용)."""
        from db import execute_query
        ids_str = ", ".join(str(s) for s in stay_ids)
        rows = execute_query(
            f"SELECT * FROM aki_project.cdss_master_features WHERE stay_id IN ({ids_str})"
        )
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ── 예측 ──────────────────────────────────────────────────────────────

    def predict_single(
        self,
        stay_id: int,
        df_patient: Optional[pd.DataFrame] = None,
    ) -> InferenceResult:
        """단일 환자 AKI 발생 확률을 예측한다.

        SCR-06 API의 predict_aki_probability_using_xgboost_model()에서 호출.

        Args:
            stay_id:    ICU 체류 ID
            df_patient: 이미 조회된 피처 DataFrame (없으면 DB에서 자동 조회)

        Returns:
            InferenceResult

        Raises:
            ModelNotReadyError: 모델 파일이 없을 때
            InferenceError:     예측 실행 중 오류
        """
        if not self._is_ready:
            raise ModelNotReadyError(
                f"XGBoost 모델이 준비되지 않았습니다. "
                f"train.py 실행 후 {self._model_path} 파일을 확인하세요."
            )

        # 피처 DataFrame 확보
        if df_patient is None:
            df_patient = self._fetch_patient_features_from_db(stay_id, None)
        if df_patient is None or df_patient.empty:
            raise InferenceError(f"stay_id={stay_id} 피처 데이터 없음")

        # 전처리
        try:
            X = preprocess_for_inference(
                df_patient,
                feature_names=self._feature_names,
                encoders=self._encoders,
            )
        except Exception as e:
            raise InferenceError(f"전처리 실패: {e}")

        # 결측 피처 수 계산 (신뢰도 지표)
        missing_count = int(X.isna().sum().sum())

        # XGBoost 예측
        try:
            prob = float(self._model.predict_proba(X)[0, 1])
        except Exception as e:
            raise InferenceError(f"XGBoost 예측 실패: {e}")

        prob_pct    = max(0, min(100, round(prob * 100)))
        is_high     = prob >= self._threshold

        logger.debug(
            f"[추론] stay_id={stay_id}  prob={prob:.4f}  "
            f"high_risk={is_high}  missing={missing_count}"
        )

        return InferenceResult(
            stay_id             = stay_id,
            aki_probability     = prob,
            aki_probability_pct = prob_pct,
            is_high_risk        = is_high,
            threshold           = self._threshold,
            model_version       = self._model_path,
            feature_values      = X.iloc[0].to_dict(),
            missing_features    = missing_count,
        )

    def predict_batch(
        self,
        stay_ids: list[int],
    ) -> BatchInferenceResult:
        """여러 환자의 AKI 발생 확률을 배치로 예측한다.

        SCR-07 시계열 스케줄러에서 전체 ICU 환자를 1시간마다 갱신할 때 사용.
        단일 예측 반복보다 한 번의 DB 조회로 효율적으로 처리한다.

        Args:
            stay_ids: ICU 체류 ID 목록

        Returns:
            BatchInferenceResult (성공·실패 분리)
        """
        if not self._is_ready:
            raise ModelNotReadyError("XGBoost 모델이 준비되지 않았습니다.")

        # 배치 피처 조회
        df_all = self._fetch_batch_features_from_db(stay_ids)
        if df_all.empty:
            return BatchInferenceResult(
                results=[], n_total=len(stay_ids),
                n_high_risk=0, n_failed=len(stay_ids),
                failed_stay_ids=stay_ids,
            )

        # 전처리
        X_all = preprocess_for_inference(
            df_all, self._feature_names, self._encoders
        )

        # 배치 예측
        probs = self._model.predict_proba(X_all)[:, 1]

        results: list[InferenceResult] = []
        found_ids = set(df_all["stay_id"].tolist())
        failed    = [s for s in stay_ids if s not in found_ids]

        for i, row in df_all.iterrows():
            sid  = int(row["stay_id"])
            prob = float(probs[i])
            results.append(InferenceResult(
                stay_id             = sid,
                aki_probability     = prob,
                aki_probability_pct = max(0, min(100, round(prob * 100))),
                is_high_risk        = prob >= self._threshold,
                threshold           = self._threshold,
                model_version       = self._model_path,
                missing_features    = int(X_all.iloc[i].isna().sum()),
            ))

        n_high = sum(1 for r in results if r.is_high_risk)
        logger.info(
            f"[배치 추론] {len(results)}건 완료  고위험 {n_high}건  실패 {len(failed)}건"
        )

        return BatchInferenceResult(
            results         = results,
            n_total         = len(stay_ids),
            n_high_risk     = n_high,
            n_failed        = len(failed),
            failed_stay_ids = failed,
        )

    # ── 설명 가능성 ────────────────────────────────────────────────────────

    def get_shap_values_for_explanation(
        self,
        stay_id: int,
        df_patient: Optional[pd.DataFrame] = None,
    ) -> Optional[pd.DataFrame]:
        """단일 환자의 SHAP 값을 계산해 피처별 기여도를 반환한다.

        SCR-06 위험 요인 테이블에서 "왜 이 점수인가"를 설명하는 데 사용 가능.
        shap 라이브러리가 설치된 경우에만 동작한다.

        Args:
            stay_id:    ICU 체류 ID
            df_patient: 이미 조회된 피처 DataFrame

        Returns:
            SHAP 값 DataFrame (피처명·shap_value·abs_shap 컬럼), 실패 시 None
        """
        try:
            import shap
        except ImportError:
            logger.warning("shap 라이브러리 없음. pip install shap")
            return None

        if df_patient is None:
            df_patient = self._fetch_patient_features_from_db(stay_id, None)
        if df_patient is None:
            return None

        X = preprocess_for_inference(df_patient, self._feature_names, self._encoders)

        explainer  = shap.TreeExplainer(self._model)
        shap_vals  = explainer.shap_values(X)[0]   # 양성 클래스 SHAP 값

        df_shap = pd.DataFrame({
            "feature":   self._feature_names,
            "shap_value":shap_vals,
            "abs_shap":  np.abs(shap_vals),
        }).sort_values("abs_shap", ascending=False)

        return df_shap


# ─────────────────────────────────────────────────────────────────────────────
# 모듈 레벨 싱글턴
# FastAPI main.py 또는 scr06_ai_risk_score.py에서 import해서 사용
# ─────────────────────────────────────────────────────────────────────────────

# 앱 시작 시 1회 생성. 이후 모든 API가 이 인스턴스를 재사용한다.
aki_engine = AKIInferenceEngine()


# ─────────────────────────────────────────────────────────────────────────────
# CLI 테스트 (직접 실행 시)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AKI XGBoost 추론 테스트")
    parser.add_argument("--stay-id", type=int, required=True, help="ICU stay_id")
    args = parser.parse_args()

    if not aki_engine.is_ready:
        print(" 모델이 준비되지 않았습니다. train.py를 먼저 실행하세요.")
        exit(1)

    print(f"\n[추론 테스트] stay_id={args.stay_id}")
    try:
        result = aki_engine.predict_single(args.stay_id)
        print(f"  AKI 발생 확률 : {result.aki_probability:.4f} ({result.aki_probability_pct}%)")
        print(f"  고위험 여부   : {'⚠ 고위험' if result.is_high_risk else '✅ 저위험'}")
        print(f"  분류 임계값   : {result.threshold:.3f}")
        print(f"  결측 피처 수  : {result.missing_features}개")
        print(f"  모델 버전     : {result.model_version}")
    except (ModelNotReadyError, InferenceError) as e:
        print(f" 예측 실패: {e}")