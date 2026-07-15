"""협진 레포지토리 — 타임라인 자식 로딩 + 부서 인박스 필터."""
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from core.query_optimizer import Page, paginate
from models.consultation import Consultation
from repositories.base import BaseRepository


class ConsultationRepository(BaseRepository[Consultation]):
    model = Consultation

    def get_with_timeline(self, consult_id: str) -> Consultation | None:
        stmt = (
            select(Consultation)
            .where(Consultation.id == consult_id)
            .options(selectinload(Consultation.timeline))
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def list_with_timeline(
        self, page: Page, kind: str | None = None
    ) -> list[Consultation]:
        """협진 목록 — 타임라인 selectinload + 최신순 + 페이지네이션.

        kind 지정 시 부서 인박스(kind+status 인덱스 활용).
        """
        stmt = select(Consultation).options(selectinload(Consultation.timeline))
        if kind:
            stmt = stmt.where(Consultation.kind == kind)
        stmt = paginate(stmt.order_by(Consultation.requested_at.desc()), page)
        return list(self.db.execute(stmt).scalars().all())
