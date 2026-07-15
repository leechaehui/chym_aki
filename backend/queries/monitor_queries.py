"""ICU 모니터 서비스 SQL 쿼리.

원본: services/icu_monitor_service.py 에 인라인으로 있던 SQL 을 추출.
─────────────────────────────────────────────────────────────
연관 백엔드     : services/icu_monitor_service.py
대상 테이블     : chym.icu_features_scaled, chym.cohort
─────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from sqlalchemy import text


def sql_scaled_features_for_stay(schema: str, table: str, selected: str) -> text:
    """해당 stay 의 모델 입력(스케일 완료) 35 피처값.

    호출처: icu_monitor_service.scaled_features_for_stay()
    연관  : model_explainer (SHAP 계산)
    """
    return text(
        f"SELECT {selected} FROM {schema}.{table} WHERE stay_id = :s LIMIT 1"
    )


def sql_scaled_table_regclass(schema: str, table: str) -> text:
    """테이블 존재 확인 (to_regclass).

    호출처: icu_monitor_service._scaled_table_count()
    """
    return text("SELECT to_regclass(:t)")


def sql_scaled_table_count(schema: str, table: str) -> text:
    """스케일 피처 테이블 행 수.

    호출처: icu_monitor_service._scaled_table_count()
    """
    return text(f"SELECT count(*) FROM {schema}.{table}")


def sql_predictions_join(schema: str, table: str) -> text:
    """스케일 피처 + 코호트 메타 조인 — 배치 예측용.

    호출처: icu_monitor_service._predictions()
    연관  : 전체 42,210명 코호트에 모델을 배치 적용할 때 사용.
            gender 는 cohort 의 raw 값을 쓴다(스케일 테이블 값은 표준화돼 있어 부적합).
            age 는 cohort.age 도 표준화(z-score)된 smallint(-2~1)라 부적합 →
            원본 나이는 mimiciv_hosp.patients.anchor_age 에서 조인해 가져온다.
    """
    return text(
        f"SELECT s.*, c.first_careunit, c.icu_los_hours, "
        f"       p.anchor_age AS cohort_age, c.gender AS cohort_gender, c.icu_intime "
        f"FROM {schema}.{table} s "
        f"LEFT JOIN {schema}.cohort c ON c.stay_id = s.stay_id "
        f"LEFT JOIN mimiciv_hosp.patients p ON p.subject_id = c.subject_id"
    )
