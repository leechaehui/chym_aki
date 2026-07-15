"""채팅 서비스."""
from sqlalchemy.orm import Session

from models.chat import ChatMessage, ChatRoom
from repositories.chat_repository import ChatRepository


class ChatService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.repo = ChatRepository(db)

    def get_or_create_room(self, user_a: str, user_b: str) -> ChatRoom:
        room = self.repo.get_or_create_room(user_a, user_b)
        self.db.commit()
        return room

    def list_rooms(self, user_id: str) -> list[ChatRoom]:
        return self.repo.list_rooms(user_id)

    def list_messages(self, room_id: str) -> list[ChatMessage]:
        return self.repo.list_messages(room_id)

    def is_member(self, room_id: str, user_id: str) -> bool:
        room = self.repo.get_room(room_id)
        if not room:
            return False
        return user_id in room.member_ids.split("_")

    def save_message(
        self, room_id: str, sender_id: str, sender_name: str, body: str
    ) -> ChatMessage:
        msg = self.repo.add_message(room_id, sender_id, sender_name, body)
        self.db.commit()
        return msg
