"""환자 API."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from core.deps import get_current_user, get_db
from core.query_optimizer import Page
from models.user import User
from queries.constants import STAGE_LABELS
from schemas.patient import PatientOut
from services.icu_monitor_service import is_available as icu_available, _predictions
from services.patient_service import PatientService
from services.real_patient_service import get_real_patient, list_real_patients

router = APIRouter(prefix="/patients", tags=["patients"])


def _predicted_stage_by_subject() -> dict[int, str]:
    """subject_id → 모델의 실제 예측 stage 라벨. chym.patients 목록에 이걸 붙여서
    "ICU AKI 모니터링"과 등급(고위험/주의/안정) 판정 기준을 동일하게 맞춘다."""
    if not icu_available():
        return {}
    try:
        df = _predictions()
        return {int(r.subject_id): STAGE_LABELS[int(r.pred)] for r in df.itertuples() if r.subject_id is not None}
    except Exception:
        return {}


@router.get("", response_model=list[PatientOut])
def list_patients(
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """환자 목록 — 검사값(labs/trend/urine) 포함(프론트 위험판정용).

    chym.patients(데모/EMR 데이터)가 있으면 그걸 그대로 쓴다 — "ICU AKI 모니터링" 등 다른 화면과
    같은 환자·같은 수치를 보게 하려면 데모가 채운 이 테이블이 항상 단일 소스여야 한다.
    chym.patients 가 비어있을 때만(데모 미실행) 실제 MIMIC-IV 코호트로 폴백한다.
    """
    patients = PatientService(db).list_detailed(Page.of(limit, offset))
    if patients:
        stage_by_subject = _predicted_stage_by_subject()
        out = []
        for p in patients:
            po = PatientOut.model_validate(p)
            stage = stage_by_subject.get(p.mimic_subject_id)
            if stage is not None:
                po = po.model_copy(update={"predicted_stage": stage})
            out.append(po)
        return out
    if icu_available():
        return [PatientOut(**p) for p in list_real_patients(limit, offset)]
    return []


@router.get("/{patient_id}", response_model=PatientOut)
def get_patient(
    patient_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    # 숫자 id = 실 ICU stay → 실데이터. 그 외 = 기존 mock 환자.
    if patient_id.isdigit() and icu_available():
        real = get_real_patient(int(patient_id))
        if real is not None:
            return PatientOut(**real)
    patient = PatientService(db).get_detail(patient_id)
    return PatientOut.model_validate(patient)
