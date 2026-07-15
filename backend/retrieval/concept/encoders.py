"""concept 인코더 — KPMP 범주/구간 문자열 및 MIMIC 수치를 ConceptVector 필드로 변환.

KPMP 임상필드는 구간/범주형(예: '80-89 ml/min/1.73m2 (eGFR)', '>=1000 mg/g cr (prot)')이며
일부 행은 package 단위 집계(예: 'Stage 1;Stage 2;Stage 3', 'Male;Female')다.
집계는 보수적으로 처리: ordinal=최댓값(worst), 범주=동률이면 unknown.
"""
from __future__ import annotations

import re

# 범위 문자열('140-149')의 하이픈을 음수로 오인하지 않도록 부호 미포함(임상값 ≥0).
_NUM = re.compile(r"\d+\.?\d*")


def _first_token(s: str | None) -> str | None:
    """집계 문자열에서 첫 토큰. None/빈값/'Don't Know' → None."""
    if s is None or (isinstance(s, float)) or str(s).strip() == "":
        return None
    t = str(s).split(";")[0].strip()
    return None if t.lower().startswith("don") else t


def kdigo_stage(s: str | None) -> int | None:
    """'Stage 3 (ks)' / 'Stage 1 (ks);Stage 2 (ks);Stage 3 (ks)' → worst(int 1..3) 또는 0."""
    if s is None or str(s).strip() == "":
        return None
    nums = [int(n) for n in re.findall(r"Stage\s*(\d)", str(s))]
    if not nums:
        return None
    return max(nums)


def egfr_value(s: str | float | None) -> float | None:
    """'80-89 ml/min/1.73m2 (eGFR)' → 구간 중앙값(float). 숫자면 그대로."""
    if s is None or str(s).strip() == "":
        return None
    if isinstance(s, (int, float)):
        return float(s)
    nums = [float(n) for n in _NUM.findall(str(s).split("(")[0])]
    nums = [n for n in nums if n < 1000]  # '1.73' 제거
    if not nums:
        return None
    return sum(nums[:2]) / len(nums[:2])  # 구간 중앙


# proteinuria(mg/g cr) 구간 → ordinal 0..3
_PROT_BINS = [("<150", 0), ("150 to <500", 1), ("500 to <1000", 2), (">=1000", 3)]
_A1C_BINS = [("<6.5", 0), ("6.5 to <7.5", 1), ("7.5 to <8.5", 2), (">=8.5", 3)]


def _ordinal_bin(s: str | None, bins) -> int | None:
    t = _first_token(s)
    if t is None:
        return None
    for key, val in bins:
        if key in t:
            return val
    return None


def proteinuria(s: str | None) -> int | None:
    return _ordinal_bin(s, _PROT_BINS)


def a1c(s: str | None) -> int | None:
    return _ordinal_bin(s, _A1C_BINS)


def age_years(s: str | None) -> float | None:
    """'30-39 Years' → 35.0."""
    t = _first_token(s)
    if t is None:
        return None
    nums = [float(n) for n in _NUM.findall(t)]
    return (sum(nums[:2]) / len(nums[:2])) if nums else None


def sex_male(s: str | None) -> float | None:
    t = _first_token(s)
    if t is None:
        return None
    return 1.0 if t.lower().startswith("male") else 0.0 if t.lower().startswith("female") else None


def yes_no(s: str | None) -> float | None:
    """'Yes (dh)'/'No (dh)' → 1.0/0.0, 그 외 None."""
    t = _first_token(s)
    if t is None:
        return None
    tl = t.lower()
    return 1.0 if tl.startswith("yes") else 0.0 if tl.startswith("no") else None


# KPMP primary_adjudicated_category → etiology hint
_ETIOLOGY_MAP = {
    "acute tubular injury": "ATI",
    "acute interstitial nephritis": "AIN",
    "diabetic kidney disease": "DKD",
    "prerenal": "prerenal",
    "obstruct": "postrenal",
    "post-renal": "postrenal",
}


def etiology(s: str | None) -> str:
    t = _first_token(s)
    if t is None:
        return "unknown"
    tl = t.lower()
    for key, val in _ETIOLOGY_MAP.items():
        if key in tl:
            return val
    return "unknown"
