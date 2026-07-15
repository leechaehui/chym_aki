"""AI 초안 노트 레포지토리."""
from sqlalchemy import select

from core.query_optimizer import Page, paginate
from models.ai_draft import AiDraftNote
from repositories.base import BaseRepository


class AiDraftRepository(BaseRepository[AiDraftNote]):
    model = AiDraftNote

    def list_for_patient(self, patient_id: str, page: Page) -> list[AiDraftNote]:
        """환자별 초안 이력 — 최신순 + 페이지네이션."""
        stmt = paginate(
            select(AiDraftNote)
            .where(AiDraftNote.patient_id == patient_id)
            .order_by(AiDraftNote.created_at.desc()),
            page,
        )
        return list(self.db.execute(stmt).scalars().all())
