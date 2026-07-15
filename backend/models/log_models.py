import datetime
import uuid
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    Text,
    JSON,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from core.database import Base


class UserSession(Base):
    __tablename__ = "user_sessions"

    id = Column(String(40), primary_key=True, default=lambda: uuid.uuid4().hex)
    user_id = Column(String(40), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    login_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    last_seen_at = Column(DateTime, index=True)
    logout_at = Column(DateTime)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    expire_reason = Column(String(50))


class RequestLogDLQ(Base):
    __tablename__ = "request_logs_dlq"

    id = Column(Integer, primary_key=True, autoincrement=True)
    payload = Column(JSON().with_variant(JSONB, "postgresql"))
    reason = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


# RequestLog는 파티셔닝 베이스 테이블입니다. SQLAlchemy의 제약상 선언 시점에는 Base 맵핑으로만 지정합니다.
class RequestLog(Base):
    __tablename__ = "request_logs"
    __table_args__ = (
        {"postgresql_partition_by": "RANGE (request_time)"}
    )

    request_id = Column(String(36), primary_key=True)
    request_time = Column(DateTime, primary_key=True, index=True)
    session_id = Column(String(40), ForeignKey("user_sessions.id", ondelete="SET NULL"), nullable=True)
    user_id = Column(String(40), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    trace_depth = Column(SmallInteger, default=0)
    log_seq = Column(Integer, default=0)  # ORDERING KEY
    endpoint = Column(String(255), nullable=False)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer, nullable=False)
    latency_ms = Column(Integer, nullable=False)
    response_time = Column(DateTime, nullable=False)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    payload = Column(JSON().with_variant(JSONB, "postgresql"))
    is_sampled = Column(Boolean, default=False)
    log_version = Column(Integer, default=1)


from sqlalchemy import DDL, event

event.listen(
    RequestLog.__table__,
    "after_create",
    DDL(
        """
        CREATE TABLE IF NOT EXISTS request_logs_default PARTITION OF request_logs DEFAULT;
        
        CREATE TABLE IF NOT EXISTS request_logs_2026_06 PARTITION OF request_logs FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
        CREATE TABLE IF NOT EXISTS request_logs_2026_07 PARTITION OF request_logs FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');
        """
    ).execute_if(dialect="postgresql")
)
