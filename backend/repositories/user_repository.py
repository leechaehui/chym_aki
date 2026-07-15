"""사용자 레포지토리."""
from sqlalchemy import select

from models.user import User
from repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    model = User

    def get_by_username(self, username: str) -> User | None:
        """로그인 조회 — username 유니크 인덱스 사용."""
        stmt = select(User).where(User.username == username)
        return self.db.execute(stmt).scalar_one_or_none()

    def exists_username(self, username: str) -> bool:
        """가입 중복 체크 — id 한 컬럼만 조회(불필요 적재 회피)."""
        stmt = select(User.id).where(User.username == username).limit(1)
        return self.db.execute(stmt).first() is not None

    def list_ordered(self) -> list[User]:
        """관리자 화면 — 최신 생성순."""
        stmt = select(User).order_by(User.created_at.desc())
        return list(self.db.execute(stmt).scalars().all())
