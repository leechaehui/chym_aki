"""환자 레포지토리 — 검사 자식 로딩 전략 명시(N+1 방지)."""
from sqlalchemy import select

from core.query_optimizer import Page, paginate
from models.patient import Patient
from repositories.base import BaseRepository


class PatientRepository(BaseRepository[Patient]):
    model = Patient

    def get_with_details(self, patient_id: str) -> Patient | None:
        """상세 조회 — labs/trend/urine 을 selectinload 로 한 번에 적재."""
        from sqlalchemy.orm import selectinload

        stmt = (
            select(Patient)
            .where(Patient.id == patient_id)
            .options(
                selectinload(Patient.labs),
                selectinload(Patient.trend),
                selectinload(Patient.urine_output),
            )
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def get_by_mrn(self, mrn: str) -> Patient | None:
        stmt = select(Patient).where(Patient.mrn == mrn)
        return self.db.execute(stmt).scalar_one_or_none()

    def list_summary(self, page: Page) -> list[Patient]:
        """목록 — 자식 미적재(요약 DTO 전용) + 페이지네이션."""
        stmt = paginate(select(Patient).order_by(Patient.admitted_at.desc()), page)
        return list(self.db.execute(stmt).scalars().all())

    def list_detailed(self, page: Page) -> list[Patient]:
        """목록 — labs/trend/urine 까지 selectinload(N+1 없음, 신장내과 위험판정용).

        화면이 목록의 각 환자에 대해 검사값 기반 위험도를 표시하므로 자식을 함께 적재한다.
        selectinload 라 환자 N명이어도 부모 1 + 자식 3 = 상수 4쿼리.

        정렬은 ai_risk_score DESC(admitted_at 아님) — 코호트가 페이지 limit(MAX_PAGE_SIZE=100)보다
        크면 admitted_at 순은 저위험 환자가 더 최근 시각을 가질 경우 진짜 고위험 환자가 페이지
        밖으로 잘려나가 화면(신장내과 대시보드)에 아예 안 보이는 문제가 있었다.
        """
        from sqlalchemy.orm import selectinload

        stmt = paginate(
            select(Patient)
            .order_by(Patient.ai_risk_score.desc())
            .options(
                selectinload(Patient.labs),
                selectinload(Patient.trend),
                selectinload(Patient.urine_output),
            ),
            page,
        )
        return list(self.db.execute(stmt).scalars().all())
