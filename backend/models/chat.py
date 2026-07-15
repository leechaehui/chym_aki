"""채팅 엔티티 (의료진 간 1:1 직접 메시지)."""
from sqlalchemy import ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database import Base
from models.base import TimestampMixin


class ChatRoom(Base, TimestampMixin):
    """1:1 채팅방 — member_ids 로 두 참여자 식별(정렬 후 _ 구분)."""

    __tablename__ = "chat_rooms"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    member_ids: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)

    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="room",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
    )


class ChatMessage(Base, TimestampMixin):
    """채팅 메시지 한 건."""

    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    room_id: Mapped[str] = mapped_column(
        ForeignKey("chat_rooms.id", ondelete="CASCADE"), nullable=False
    )
    sender_id: Mapped[str] = mapped_column(String(40), nullable=False)
    sender_name: Mapped[str] = mapped_column(String(60), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)

    room: Mapped["ChatRoom"] = relationship(back_populates="messages")

    __table_args__ = (Index("ix_chat_messages_room_id", "room_id"),)
