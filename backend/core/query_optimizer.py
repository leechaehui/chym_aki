"""쿼리 최적화 유틸 (QUERY OPTIMIZER).

목표: "메모리 낭비 없는 최소 비용 쿼리 구조".

설계 원칙
- SELECT * 금지 → load_only 로 필요한 컬럼만 적재.
- N+1 회피 → 연관 로딩 전략(selectinload/joinedload) 명시.
- 전체 스캔 방지 → pagination 기본 적용.
- 불필요 join 방지 → 레포지토리에서 명시적으로 옵션 부착.

이 모듈은 SQLAlchemy 위에 얇은 헬퍼만 제공하며, 어떤 비즈니스 로직도 갖지 않는다.
"""
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Select
from sqlalchemy.orm import InstrumentedAttribute, load_only, selectinload


# 페이지네이션 기본/상한 — 무한정 row 적재로 인한 메모리 폭증 방지.
# MAX_PAGE_SIZE 는 데모 코호트 전체(최대 1000명)를 단일 페이지로 조회할 수 있어야 하므로
# 그 이상으로 잡는다 — 낮으면 list_detailed 의 ai_risk_score DESC 정렬에도 불구하고
# 저위험 환자에 가려 일부 환자가 페이지 밖으로 잘려나갈 수 있다.
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 1000


@dataclass(frozen=True)
class Page:
    """페이지네이션 파라미터. limit 은 MAX_PAGE_SIZE 로 상한 클램프된다."""

    limit: int = DEFAULT_PAGE_SIZE
    offset: int = 0

    @classmethod
    def of(cls, limit: int | None, offset: int | None) -> "Page":
        safe_limit = min(max(1, limit or DEFAULT_PAGE_SIZE), MAX_PAGE_SIZE)
        safe_offset = max(0, offset or 0)
        return cls(limit=safe_limit, offset=safe_offset)


def select_columns(stmt: Select, model: Any, *columns: InstrumentedAttribute) -> Select:
    """필요한 컬럼만 적재(SELECT * 금지 원칙).

    예) select_columns(select(Bed), Bed, Bed.id, Bed.label, Bed.state)
    """
    return stmt.options(load_only(*columns))


def with_children(stmt: Select, *relationships: InstrumentedAttribute) -> Select:
    """1:N 연관을 selectinload 로 적재(N+1 방지, 카테시안 폭증 없는 별도 IN 쿼리)."""
    for rel in relationships:
        stmt = stmt.options(selectinload(rel))
    return stmt


def paginate(stmt: Select, page: Page) -> Select:
    """LIMIT/OFFSET 적용. 모든 목록 조회는 이 함수를 거치는 것을 원칙으로 한다."""
    return stmt.limit(page.limit).offset(page.offset)
