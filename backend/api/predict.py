"""Prediction refresh API."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from core.deps import require_roles
from models.user import User
from schemas.base import CamelModel
from services.ai_predict_service import refresh_predictions

router = APIRouter(prefix="/predict", tags=["predict"])


class PredictionRefreshOut(CamelModel):
    stay_id: int
    prediction_stage: str
    stage1_probability: float
    stage23_probability: float
    final_probability: float


@router.post("/refresh", response_model=list[PredictionRefreshOut])
def refresh_prediction_endpoint(
    limit: int = 200,
    _: User = Depends(require_roles("nephrology", "admin")),
):
    """Refresh ICU AKI predictions with Stage 1 and Stage 2-3 models."""
    try:
        return [PredictionRefreshOut(**row) for row in refresh_predictions(limit=limit)]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
