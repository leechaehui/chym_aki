"""채팅 API — 방 관리 · 메시지 히스토리 · 대화 상대 목록."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.deps import get_current_user, get_db
from models.user import User
from schemas.chat import ChatMessageOut, ChatRoomOut, ChatUserOut
from services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.get("/users", response_model=list[ChatUserOut])
def list_users(db: Session = Depends(get_db), me: User = Depends(get_current_user)):
    """채팅 가능한 사용자 목록 — 자신/관리자 제외 승인된 사용자(관리자는 협진 채팅 대상 아님)."""
    users = (
        db.query(User)
        .filter(User.approval == "approved", User.id != me.id, User.role != "admin")
        .order_by(User.department, User.name)
        .all()
    )
    return [
        ChatUserOut(id=u.id, name=u.name, role=u.role, department=u.department)
        for u in users
    ]


@router.get("/rooms", response_model=list[ChatRoomOut])
def list_rooms(db: Session = Depends(get_db), me: User = Depends(get_current_user)):
    """내가 참여 중인 채팅방 목록."""
    return ChatService(db).list_rooms(me.id)


@router.post("/rooms/{peer_id}", response_model=ChatRoomOut, status_code=201)
def get_or_create_room(
    peer_id: str,
    db: Session = Depends(get_db),
    me: User = Depends(get_current_user),
):
    """상대방 ID 로 1:1 방 열기 — 없으면 생성, 있으면 재사용."""
    return ChatService(db).get_or_create_room(me.id, peer_id)


@router.get("/rooms/{room_id}/messages", response_model=list[ChatMessageOut])
def list_messages(
    room_id: str,
    db: Session = Depends(get_db),
    me: User = Depends(get_current_user),
):
    svc = ChatService(db)
    if not svc.is_member(room_id, me.id):
        raise HTTPException(status_code=403, detail="채팅방 접근 권한이 없습니다.")
    messages = svc.list_messages(room_id)
    sender_ids = {m.sender_id for m in messages}
    current_names: dict[str, str] = {
        u.id: u.name
        for u in db.query(User).filter(User.id.in_(sender_ids)).all()
    }
    return [
        ChatMessageOut(
            id=m.id,
            room_id=m.room_id,
            sender_id=m.sender_id,
            sender_name=current_names.get(m.sender_id, m.sender_name),
            body=m.body,
            created_at=m.created_at,
        )
        for m in messages
    ]
