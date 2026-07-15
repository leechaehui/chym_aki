"""Concept Builder — MIMIC raw 및 KPMP 메타데이터 → ConceptVector (동일 스키마).

- KPMP: split_manifest(patient_id·kdigo·dx) ⨝ manifest_aki_full(egfr·prot·a1c·dm·htn·age·sex)
        on file_id → 환자단위 ConceptVector. 프로토타입 메타 벡터의 재료.
- MIMIC: 기존 임상 서비스(kdigo_staging·aki_feature_transform 등) 산출값을 동일 필드로 어댑트.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import encoders as E
from .schema import ConceptVector


def build_kpmp_concept(row: dict) -> ConceptVector:
    """조인된 KPMP 임상 row(dict) → ConceptVector. KPMP엔 Cr trend·Oliguria 없음(None)."""
    return ConceptVector(
        kdigo_stage=E.kdigo_stage(row.get("kdigo_stage")),
        cr_trend_slope=None,        # KPMP static — 중립
        oliguria=None,              # KPMP 없음
        egfr=E.egfr_value(row.get("baseline_egfr")),
        proteinuria=E.proteinuria(row.get("proteinuria")),
        a1c=E.a1c(row.get("a1c")),
        age=E.age_years(row.get("age_binned")),
        sex_male=E.sex_male(row.get("sex")),
        diabetes=E.yes_no(row.get("diabetes_history")),
        hypertension=E.yes_no(row.get("hypertension_history")),
        etiology_hint=E.etiology(row.get("primary_adjudicated_category")),
    )


def load_kpmp_concepts(
    artifacts_dir: Path, *, cohort_patients: set[str] | None = None
) -> dict[str, ConceptVector]:
    """KPMP CSV 조인 → {patient_id: ConceptVector}.

    cohort_patients 지정 시 그 환자(임베딩·라벨 보유 집합)로 제한.
    """
    sm = pd.read_csv(artifacts_dir / "split_manifest.csv", low_memory=False).drop_duplicates("patient_id")
    mf = pd.read_csv(artifacts_dir / "manifest_aki_full.csv", low_memory=False)
    sm["patient_id"] = sm["patient_id"].astype(str)
    clin_cols = ["baseline_egfr", "proteinuria", "a1c", "diabetes_history",
                 "hypertension_history", "age_binned", "sex"]
    merged = sm.merge(mf[["file_id", *clin_cols]], on="file_id", how="left")

    out: dict[str, ConceptVector] = {}
    for _, r in merged.iterrows():
        pid = str(r["patient_id"])
        if cohort_patients is not None and pid not in cohort_patients:
            continue
        out[pid] = build_kpmp_concept(r.to_dict())
    return out


def build_mimic_concept(
    *,
    kdigo_stage: int | None = None,
    cr_trend_slope: float | None = None,
    oliguria: int | None = None,
    egfr: float | None = None,
    proteinuria_mg_g: float | None = None,
    a1c_pct: float | None = None,
    age: float | None = None,
    sex: str | None = None,
    diabetes: bool | None = None,
    hypertension: bool | None = None,
    etiology_hint: str = "unknown",
) -> ConceptVector:
    """MIMIC 산출값(수치/불리언) → ConceptVector. KPMP와 동일 스키마로 정렬.

    호출측(MIMIC 어댑터)이 기존 서비스 출력을 이 인자로 넘긴다:
      kdigo_stage      ← services.kdigo_staging
      cr_trend_slope   ← services.aki_feature_transform (ΔCr/Δt)
      oliguria         ← 시간당 소변량 규칙(0 정상/1 oliguria<0.5/2 anuria)
      egfr             ← CKD-EPI(creatinine)
      proteinuria_mg_g ← urine protein/creatinine (mg/g)
      etiology_hint    ← 임상 규칙(prerenal/ATI/AIN/postrenal/unknown)
    """
    def prot_bin(v):
        if v is None:
            return None
        return 0 if v < 150 else 1 if v < 500 else 2 if v < 1000 else 3

    def a1c_bin(v):
        if v is None:
            return None
        return 0 if v < 6.5 else 1 if v < 7.5 else 2 if v < 8.5 else 3

    return ConceptVector(
        kdigo_stage=kdigo_stage,
        cr_trend_slope=cr_trend_slope,
        oliguria=oliguria,
        egfr=egfr,
        proteinuria=prot_bin(proteinuria_mg_g),
        a1c=a1c_bin(a1c_pct),
        age=age,
        sex_male=(1.0 if sex and sex.lower().startswith("m")
                  else 0.0 if sex and sex.lower().startswith("f") else None),
        diabetes=(None if diabetes is None else float(diabetes)),
        hypertension=(None if hypertension is None else float(hypertension)),
        etiology_hint=etiology_hint,
    )
