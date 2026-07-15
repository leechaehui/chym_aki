"""채팅 DTO."""
from datetime import datetime

from schemas.base import CamelModel


class ChatUserOut(CamelModel):
    id: str
    name: str
    role: str
    department: str


class ChatRoomOut(CamelModel):
    id: str
    member_ids: str
    created_at: datetime


class ChatMessageOut(CamelModel):
    id: str
    room_id: str
    sender_id: str
    sender_name: str
    body: str
    created_at: datetime
