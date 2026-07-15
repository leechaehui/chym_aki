"""병리 결과 레포지토리."""
from sqlalchemy import select

from models.pathology import PathologyResult
from repositories.base import BaseRepository


class PathologyRepository(BaseRepository[PathologyResult]):
    model = PathologyResult

    def get_by_consult(self, consult_id: str) -> PathologyResult | None:
        return self.db.get(PathologyResult, consult_id)

    def list_all_ordered(self) -> list[PathologyResult]:
        # 건수가 적은 참조성 데이터 → 전체 조회.
        return list(self.db.execute(select(PathologyResult)).scalars().all())

    def get_or_create(self, consult_id: str) -> PathologyResult:
        """리포트만 먼저 작성하는 경우(WSI 분석 결과가 아직 없는 신규 협진)에도
        report_* 컬럼을 저장할 행이 있어야 하므로, 없으면 빈 레코드를 만든다."""
        row = self.get_by_consult(consult_id)
        if row:
            return row
        row = PathologyResult(consult_id=consult_id, stain="")
        return self.add(row)
