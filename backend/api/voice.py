"""음성/AI 진료 초안 API (신장내과).

- /voice/transcribe : 오디오 업로드 → 전사(STT). 오디오 없으면 text 폴백.
- /voice/draft      : 전사 → NLP → AKI → SOAP 초안 생성·저장.
- /voice/drafts/... : 환자별 초안 이력 조회.
"""
from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from core.deps import get_current_user, get_db, require_roles
from core.query_optimizer import Page
from models.user import User
from schemas.nephrology import (
    AiDraftResultOut,
    TranscriptionResult,
    VoiceDraftRequest,
)
from services.ai_draft_service import AiDraftService

router = APIRouter(prefix="/voice", tags=["voice"])


@router.post("/transcribe", response_model=TranscriptionResult)
async def transcribe(
    audio: UploadFile | None = File(default=None),
    text: str = Form(default=""),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """음성 전사. faster-whisper 미설치 환경에서는 text(passthrough)로 동작."""
    audio_bytes = await audio.read() if audio else None
    result = AiDraftService(db).transcribe(audio_bytes, text)
    return TranscriptionResult(**result)


@router.post("/draft", response_model=AiDraftResultOut, status_code=201)
def generate_draft(
    body: VoiceDraftRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("nephrology", "admin")),
):
    """전사 → SOAP(근거기반) → Problem List → CDSS Risk → 검증·저장(트랜잭션).

    검증 실패 시 422(VALIDATION_FAILED) — 안전하지 않은 초안은 저장되지 않는다.
    숫자 patient_id = 실 ICU stay → 실데이터 기반 초안(IcuDraftService, 영속화 없음).
    """
    if body.patient_id.isdigit():
        from fastapi import HTTPException

        from services.icu_draft_service import IcuDraftService
        from services.icu_monitor_service import is_available as icu_available

        if not icu_available():
            raise HTTPException(status_code=503, detail="ICU cohort/model not available")
        result = IcuDraftService().draft_for_stay(int(body.patient_id), body.transcript or "")
        if result is None:
            raise HTTPException(status_code=404, detail="patient not found in ICU cohort")
        return AiDraftResultOut(**result)

    service = AiDraftService(db)
    note = service.generate_draft(
        patient_id=body.patient_id, transcript=body.transcript, actor_id=user.id
    )
    return AiDraftResultOut(**service.to_out(note))


@router.get("/drafts/{patient_id}", response_model=list[AiDraftResultOut])
def list_drafts(
    patient_id: str,
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    service = AiDraftService(db)
    notes = service.list_for_patient(patient_id, Page.of(limit, offset))
    return [AiDraftResultOut(**service.to_out(note)) for note in notes]
