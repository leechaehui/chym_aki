"""병상 레포지토리 — 행 잠금(FOR UPDATE) 제공(병상 트랜잭션 핵심)."""
from sqlalchemy import select

from models.bed import Bed
from repositories.base import BaseRepository


class BedRepository(BaseRepository[Bed]):
    model = Bed

    def get_for_update(self, bed_id: str) -> Bed | None:
        """배정/해제 트랜잭션용 비관적 잠금 조회(SELECT ... FOR UPDATE).

        PostgreSQL 에서는 실제 행 잠금, SQLite 에서는 쓰기 직렬화로 동작한다.
        """
        stmt = select(Bed).where(Bed.id == bed_id).with_for_update()
        return self.db.execute(stmt).scalar_one_or_none()

    def list_ordered(self) -> list[Bed]:
        """보드 전체 — 구역+라벨 순(인덱스 정렬)."""
        stmt = select(Bed).order_by(Bed.zone, Bed.label)
        return list(self.db.execute(stmt).scalars().all())
