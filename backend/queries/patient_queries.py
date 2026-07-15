"""환자 서비스 SQL 쿼리.

원본: services/real_patient_service.py 에 인라인으로 있던 SQL 을 추출.
─────────────────────────────────────────────────────────────
연관 백엔드     : services/real_patient_service.py
대상 테이블     : chym.final_features_48h
─────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from sqlalchemy import text


def sql_raw_labs(schema: str) -> text:
    """final_features_48h 의 raw 집계(bun/na/k/hco3)를 stay 별로 배치 조회.

    호출처: real_patient_service._raw_labs()
    연관  : GET /api/patients — 대시보드 Lab 데이터 표시.
            이 쿼리는 반드시 Raw 값(원본 의료 단위)을 반환해야 한다.
            icu_features_scaled(RobustScaler 적용) 테이블과 혼동 금지.
    """
    return text(
        f"SELECT stay_id, bun_max, sodium_min, potassium_max, bicarbonate_min "
        f"FROM {schema}.final_features_48h WHERE stay_id = ANY(:ids)"
    )


def sql_raw_bun_for_stay(schema: str) -> text:
    """단일 stay 의 BUN raw 값 조회.

    호출처: icu_patient_detail_service.summary() — BUN 누락 수정용.
    연관  : GET /api/nephrology/icu/patients/{stay_id}/summary
    """
    return text(
        f"SELECT bun_max FROM {schema}.final_features_48h "
        "WHERE stay_id = :s LIMIT 1"
    )
