"""V&V 전용 검증 레이어.

이 패키지는 작업지시서의 Validation(의료/AI 유효성) 규칙을 코드로 강제한다.
다른 레이어(api/service/repository)와 달리 '검증'만 책임지며, 어떤 도메인 상태도
변경하지 않는다(순수 함수 + 리포트 생성).

구성
- metrics      : 분류 성능 지표(AUROC/AUPRC/calibration/Brier/ECE) — 순수 numpy.
- subgroup     : subgroup analysis + missing-data sensitivity.
- clinical     : 임상 타당성 검증(Cr/eGFR/소변량/약물 단조성) — 예측기 계약 기반.
- timeseries   : 환자 타임라인 재구성 + sliding window + onset-time labeling.
- report       : 위 결과를 종합한 ValidationReport 데이터클래스.
"""
from validator.metrics import (
    BinaryMetrics,
    CalibrationBin,
    binary_metrics,
    calibration_table,
)
from validator.report import ValidationReport

__all__ = [
    "BinaryMetrics",
    "CalibrationBin",
    "binary_metrics",
    "calibration_table",
    "ValidationReport",
]
