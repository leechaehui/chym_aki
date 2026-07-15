"""채팅 리포지토리."""
from sqlalchemy.orm import Session

from models.base import new_id
from models.chat import ChatMessage, ChatRoom


class ChatRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_or_create_room(self, user_a: str, user_b: str) -> ChatRoom:
        member_ids = "_".join(sorted([user_a, user_b]))
        room = self.db.query(ChatRoom).filter_by(member_ids=member_ids).first()
        if not room:
            room = ChatRoom(id=new_id("r"), member_ids=member_ids)
            self.db.add(room)
            self.db.flush()
        return room

    def get_room(self, room_id: str) -> ChatRoom | None:
        return self.db.query(ChatRoom).filter_by(id=room_id).first()

    def list_rooms(self, user_id: str) -> list[ChatRoom]:
        return (
            self.db.query(ChatRoom)
            .filter(ChatRoom.member_ids.contains(user_id))
            .order_by(ChatRoom.created_at.desc())
            .all()
        )

    def list_messages(self, room_id: str, limit: int = 100) -> list[ChatMessage]:
        return (
            self.db.query(ChatMessage)
            .filter_by(room_id=room_id)
            .order_by(ChatMessage.created_at)
            .limit(limit)
            .all()
        )

    def add_message(
        self, room_id: str, sender_id: str, sender_name: str, body: str
    ) -> ChatMessage:
        msg = ChatMessage(
            id=new_id("m"),
            room_id=room_id,
            sender_id=sender_id,
            sender_name=sender_name,
            body=body,
        )
        self.db.add(msg)
        self.db.flush()
        return msg
