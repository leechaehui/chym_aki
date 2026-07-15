"""AI prediction refresh service for ICU AKI monitoring.

Loads test_data.csv when available and returns simultaneous Stage 1 and
Stage 2-3 probabilities. The service falls back to the existing ICU cohort
prediction cache so the refresh endpoint remains usable in this project setup.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from services import icu_monitor_service

BACKEND = Path(__file__).resolve().parent.parent
MODELS = BACKEND / "ml_models"
TEST_DATA_CANDIDATES = (
    BACKEND / "test_data.csv",
    BACKEND.parent / "test_data.csv",
    BACKEND.parent / "AKI" / "test_data.csv",
)


def _load_bundle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def _positive_probability(model: Any, x: pd.DataFrame):
    proba = model.predict_proba(x)
    if getattr(proba, "ndim", 1) == 2:
        return proba[:, 1]
    return proba


def _find_test_data() -> Path | None:
    for path in TEST_DATA_CANDIDATES:
        if path.exists():
            return path
    return None


def _stage_from_probs(stage1_probability: float, stage23_probability: float) -> tuple[str, float]:
    if stage23_probability >= stage1_probability:
        return "AKI Stage 2-3", stage23_probability
    return "AKI Stage 1", stage1_probability


def refresh_predictions(limit: int = 200) -> list[dict[str, Any]]:
    """Run Stage 1 and Stage 2-3 predictions and return API-ready rows."""
    test_data = _find_test_data()
    if test_data is None:
        return _fallback_from_icu_cache(limit=limit)

    stage1_bundle = _load_bundle(MODELS / "stage1_LR_full.pkl")
    stage23_bundle = _load_bundle(MODELS / "stage2_LGBM_v13_full_classweight.pkl")
    feature_cols = list(stage1_bundle["feature_cols"])

    df = pd.read_csv(test_data)
    if "stay_id" not in df.columns:
        df.insert(0, "stay_id", range(1, len(df) + 1))

    x = df.reindex(columns=feature_cols).fillna(0)
    stage1_probability = _positive_probability(stage1_bundle["model"], x.to_numpy(dtype=float))
    stage23_probability = _positive_probability(stage23_bundle["model"], x)

    rows: list[dict[str, Any]] = []
    for idx, row in df.head(limit).iterrows():
        p1 = round(float(stage1_probability[idx]), 4)
        p23 = round(float(stage23_probability[idx]), 4)
        stage, final_probability = _stage_from_probs(p1, p23)
        rows.append({
            "stay_id": int(row["stay_id"]),
            "prediction_stage": stage,
            "stage1_probability": p1,
            "stage23_probability": p23,
            "final_probability": round(float(final_probability), 4),
        })
    return rows


def _fallback_from_icu_cache(limit: int) -> list[dict[str, Any]]:
    patients = icu_monitor_service.IcuMonitorService().list_patients(limit=limit)
    rows: list[dict[str, Any]] = []
    for patient in patients:
        p1 = float(patient["p_stage1"])
        p23 = float(patient["p_stage2_plus"])
        stage, final_probability = _stage_from_probs(p1, p23)
        rows.append({
            "stay_id": int(patient["stay_id"]),
            "prediction_stage": stage,
            "stage1_probability": round(p1, 4),
            "stage23_probability": round(p23, 4),
            "final_probability": round(float(final_probability), 4),
        })
    return rows
