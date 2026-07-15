"""ConceptVector — 고정 차원 임상 concept 표현 (MIMIC·KPMP 공통 공간).

Frozen Architecture 의 Clinical Concept Layer:
  사용 concept = KDIGO Stage · Creatinine Trend · Oliguria · AKI Etiology Hint
              + (선택 (나)안) 공통 임상필드: eGFR · Proteinuria · a1c · Age · Sex · DM · HTN
  미사용 = Fibrosis Risk / Tubular Injury Probability / Inflammation Score 등 *병리 개념*
          (임상 데이터만으로 직접 검증 불가 → concept 에 넣지 않는다)

벡터 레이아웃(DIM=16, 모든 성분 0..1 정규화):
  [0] KDIGO            stage/3
  [1] Cr trend         (slope tanh +1)/2     ← MIMIC 전용(KPMP=0.5 중립)
  [2] Oliguria         level/2               ← MIMIC 전용(KPMP=0)
  [3] eGFR severity    1 - clip(eGFR/120)    (낮은 신기능 = 높은 값)
  [4] Proteinuria      bin/3
  [5] a1c              bin/3
  [6] Age              years/100
  [7] Sex(male)        1/0/0.5
  [8] Diabetes         1/0/0.5
  [9] Hypertension     1/0/0.5
  [10..15] Etiology one-hot(6): ATI·AIN·DKD·prerenal·postrenal·unknown

WEIGHTS: cross-cohort 유사도(concept↔prototype)가 *공통* 임상필드로 구동되도록,
MIMIC 전용(Cr trend·Oliguria)은 저가중. 양쪽 공통필드는 full 가중.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DIM = 16

ETIOLOGY_ORDER = ["ATI", "AIN", "DKD", "prerenal", "postrenal", "unknown"]
ETIOLOGY_IDX = {e: i for i, e in enumerate(ETIOLOGY_ORDER)}

# 성분별 가중(코사인 전 적용).
# 설계 원칙: 검색은 **객관 임상값**(KDIGO·eGFR·Proteinuria·DM·HTN·Age·Sex)이 구동한다.
# Etiology(ATI/AIN/DKD 등)는 MIMIC에서 항상 명확하지 않고 이미 병리적 의미가 강하므로,
# 강한 입력으로 쓰지 않고 **hint 수준(저가중)**으로만 반영한다. MIMIC전용(Cr trend·Oliguria)도 저가중.
WEIGHTS = np.array(
    [1.0,  # kdigo (객관, 양쪽)
     0.3,  # cr_trend (MIMIC only)
     0.3,  # oliguria (MIMIC only)
     1.0,  # egfr (객관)
     1.0,  # proteinuria (객관)
     0.7,  # a1c
     0.6,  # age
     0.4,  # sex
     0.8,  # dm
     0.8,  # htn
     0.25, 0.25, 0.25, 0.25, 0.25, 0.1],  # etiology one-hot = hint 수준(unknown 최저)
    dtype=np.float32,
)


@dataclass(frozen=True)
class ConceptVector:
    """임상 concept 값 객체. None=결측(중립 인코딩). MIMIC·KPMP 가 동일 스키마로 생성."""

    kdigo_stage: int | None = None        # 0..3
    cr_trend_slope: float | None = None   # ΔCr/Δt (MIMIC), KPMP None
    oliguria: int | None = None           # 0 정상/1 oliguria/2 anuria (MIMIC)
    egfr: float | None = None             # ml/min/1.73m^2
    proteinuria: int | None = None        # 0..3 ordinal bin
    a1c: int | None = None                # 0..3 ordinal bin
    age: float | None = None              # years
    sex_male: float | None = None         # 1 male / 0 female / 0.5 unknown
    diabetes: float | None = None         # 1/0/0.5
    hypertension: float | None = None     # 1/0/0.5
    etiology_hint: str = "unknown"        # ETIOLOGY_ORDER 중 하나

    def kdigo_band(self) -> str:
        if self.kdigo_stage is None:
            return "unknown"
        return "0" if self.kdigo_stage == 0 else "1" if self.kdigo_stage == 1 else "2-3"

    def to_vector(self, *, weighted: bool = True) -> np.ndarray:
        v = np.zeros(DIM, np.float32)
        v[0] = (self.kdigo_stage / 3.0) if self.kdigo_stage is not None else 0.0
        v[1] = ((np.tanh(self.cr_trend_slope) + 1) / 2) if self.cr_trend_slope is not None else 0.5
        v[2] = (self.oliguria / 2.0) if self.oliguria is not None else 0.0
        v[3] = (1 - np.clip(self.egfr / 120.0, 0, 1)) if self.egfr is not None else 0.0
        v[4] = (self.proteinuria / 3.0) if self.proteinuria is not None else 0.0
        v[5] = (self.a1c / 3.0) if self.a1c is not None else 0.0
        v[6] = np.clip((self.age or 0) / 100.0, 0, 1)
        v[7] = self.sex_male if self.sex_male is not None else 0.5
        v[8] = self.diabetes if self.diabetes is not None else 0.5
        v[9] = self.hypertension if self.hypertension is not None else 0.5
        v[10 + ETIOLOGY_IDX.get(self.etiology_hint, ETIOLOGY_IDX["unknown"])] = 1.0
        return v * WEIGHTS if weighted else v

    def completeness(self) -> float:
        """결측 아닌 concept 비율(0..1) — calibration 의 concept_completeness 입력."""
        fields = [self.kdigo_stage, self.egfr, self.proteinuria, self.a1c,
                  self.age, self.sex_male, self.diabetes, self.hypertension]
        present = sum(x is not None for x in fields)
        present += 1 if self.etiology_hint != "unknown" else 0
        return round(present / 9.0, 3)
