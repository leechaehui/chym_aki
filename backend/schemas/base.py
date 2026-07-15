"""스키마 공통 베이스.

프론트엔드(TypeScript) 는 camelCase 를 사용하고 백엔드는 snake_case 를 쓴다.
CamelModel 은 두 표현을 자동 변환한다:
- 출력(JSON) : camelCase alias 로 직렬화.
- 입력      : snake_case/camelCase 모두 허용(populate_by_name).
"""
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    """camelCase 직렬화 + ORM 객체 변환(from_attributes) 활성화 베이스."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
    )
