"""WebSocket — 실시간 ALERT 푸시 (지시서 §6, §11).

Backend(동기 이벤트 버스) → WebSocket(async) 브리지:
- lifespan 에서 running loop 를 매니저에 주입.
- ALERT_EVENT 구독자(sync 요청 스레드)가 broadcast_threadsafe() 호출 →
  loop.call_soon_threadsafe 로 각 연결의 asyncio.Queue 에 메시지 적재 → 연결별 sender 가 전송.
- 부서(department)별로 라우팅(현재 AKI alert 는 nephrology).
UI 는 이 채널의 consumer 일 뿐 어떤 판단도 하지 않는다(지시서 §6).
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

import jwt
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.event_bus import ALERT_EVENT, event_bus
from core.logging import get_logger
from core.security import decode_access_token

log = get_logger("chym.ws")

router = APIRouter(tags=["ws"])


@dataclass(eq=False)  # 객체 식별 기반 해시(set 등록용) — 큐 등 가변 필드 보유.
class _Conn:
    ws: WebSocket
    department: str
    user_id: str = ""
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)


class WSConnectionManager:
    """부서/사용자별 WebSocket 연결 레지스트리 + 스레드세이프 브로드캐스트."""

    def __init__(self) -> None:
        self._conns: dict[str, set[_Conn]] = {}         # department → conns
        self._by_user: dict[str, set[_Conn]] = {}       # user_id → conns (정밀 타겟팅)
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def register(self, ws: WebSocket, department: str, user_id: str = "") -> _Conn:
        conn = _Conn(ws=ws, department=department, user_id=user_id)
        self._conns.setdefault(department, set()).add(conn)
        if user_id:
            self._by_user.setdefault(user_id, set()).add(conn)
        return conn

    def unregister(self, conn: _Conn) -> None:
        peers = self._conns.get(conn.department)
        if peers:
            peers.discard(conn)
        upeers = self._by_user.get(conn.user_id)
        if upeers:
            upeers.discard(conn)

    def _enqueue(self, conns, message: dict) -> None:
        if self._loop is None:
            return
        for conn in list(conns):
            self._loop.call_soon_threadsafe(conn.queue.put_nowait, message)

    def broadcast_threadsafe(self, department: str, message: dict) -> None:
        """동기(요청) 스레드에서 호출 — 해당 부서 연결들의 큐에 메시지 적재."""
        self._enqueue(self._conns.get(department, ()), message)

    def send_to_user_threadsafe(self, user_id: str, message: dict) -> None:
        """특정 사용자 계정의 모든 연결로만 푸시(정밀 타겟팅)."""
        self._enqueue(self._by_user.get(user_id, ()), message)


ws_manager = WSConnectionManager()


def push_alert_to_ws(event: dict) -> None:
    """ALERT_EVENT 구독 핸들러 — targetUserId 가 있으면 그 계정에만, 없으면 부서로 푸시."""
    target = event.get("targetUserId")
    if target:
        ws_manager.send_to_user_threadsafe(target, event)
    else:
        ws_manager.broadcast_threadsafe(event.get("department", "nephrology"), event)


# 파이프라인 부팅 시 구독 등록은 services.pipeline 에서 수행.


# ──────────────────────────────────────────────
# 채팅 WebSocket
# ──────────────────────────────────────────────

class ChatConnectionManager:
    """방별 WebSocket 연결 레지스트리 + 동일 방 브로드캐스트."""

    def __init__(self) -> None:
        self._conns: dict[str, set[_Conn]] = {}

    def register(self, ws: WebSocket, room_id: str) -> _Conn:
        conn = _Conn(ws=ws, department=room_id)
        self._conns.setdefault(room_id, set()).add(conn)
        return conn

    def unregister(self, conn: _Conn) -> None:
        peers = self._conns.get(conn.department)
        if peers:
            peers.discard(conn)

    def broadcast_room(self, room_id: str, message: dict) -> None:
        for conn in list(self._conns.get(room_id, ())):
            conn.queue.put_nowait(message)


chat_manager = ChatConnectionManager()


def _resolve_chat_user(room_id: str, user_id: str) -> str | None:
    """동기 DB 세션 — executor 에서 실행. 멤버 확인 후 사용자 이름 반환."""
    from core.database import SessionLocal
    from models.user import User
    from services.chat_service import ChatService

    db = SessionLocal()
    try:
        if not ChatService(db).is_member(room_id, user_id):
            return None
        user = db.query(User).filter_by(id=user_id).first()
        return user.name if user else None
    finally:
        db.close()


def _persist_chat_message(
    room_id: str, sender_id: str, sender_name: str, body: str
) -> dict:
    """동기 DB 세션 — executor 에서 실행. 메시지 저장 후 브로드캐스트용 dict 반환."""
    from core.database import SessionLocal
    from services.chat_service import ChatService

    db = SessionLocal()
    try:
        msg = ChatService(db).save_message(room_id, sender_id, sender_name, body)
        return {
            "type": "CHAT_MESSAGE",
            "id": msg.id,
            "roomId": room_id,
            "senderId": sender_id,
            "senderName": sender_name,
            "body": body,
            "createdAt": msg.created_at.isoformat(),
        }
    finally:
        db.close()


@router.websocket("/ws/chat/{room_id}")
async def chat_ws(ws: WebSocket, room_id: str, token: str | None = None) -> None:
    """채팅 실시간 채널. ?token=<jwt> 인증, room_id 멤버 검증."""
    if not token:
        await ws.close(code=1008)
        return
    try:
        payload = decode_access_token(token)
    except jwt.PyJWTError:
        await ws.close(code=1008)
        return

    user_id: str = payload.get("sub", "")
    loop = asyncio.get_event_loop()

    sender_name = await loop.run_in_executor(None, _resolve_chat_user, room_id, user_id)
    if sender_name is None:
        await ws.close(code=4003)
        return

    await ws.accept()
    conn = chat_manager.register(ws, room_id)
    log.info("Chat WS connected room=%s user=%s", room_id, user_id)

    async def _sender() -> None:
        while True:
            msg = await conn.queue.get()
            await ws.send_json(msg)

    sender_task = asyncio.create_task(_sender())
    try:
        while True:
            data = await ws.receive_text()
            try:
                body = json.loads(data).get("body", "").strip()
                if body:
                    msg_dict = await loop.run_in_executor(
                        None, _persist_chat_message, room_id, user_id, sender_name, body
                    )
                    chat_manager.broadcast_room(room_id, msg_dict)
            except Exception:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        sender_task.cancel()
        chat_manager.unregister(conn)
        log.info("Chat WS disconnected room=%s user=%s", room_id, user_id)


@router.websocket("/ws/alerts")
async def alerts_ws(ws: WebSocket, token: str | None = None) -> None:
    """ALERT_EVENT 실시간 수신 채널. ?token=<jwt> 로 인증, role 을 부서로 사용."""
    # 1) 인증 — 쿼리 토큰 검증(헤더를 쓸 수 없는 WS 환경).
    if not token:
        await ws.close(code=1008)
        return
    try:
        payload = decode_access_token(token)
    except jwt.PyJWTError:
        await ws.close(code=1008)
        return
    department = payload.get("role", "")
    user_id = payload.get("sub", "")

    await ws.accept()
    conn = ws_manager.register(ws, department, user_id)
    log.info("WS connected dept=%s user=%s", department, user_id)

    async def _sender() -> None:
        while True:
            msg = await conn.queue.get()
            await ws.send_json(msg)

    sender = asyncio.create_task(_sender())
    try:
        # 클라이언트 메시지는 사용하지 않음 — 연결 종료 감지용으로만 수신.
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        sender.cancel()
        ws_manager.unregister(conn)
        log.info("WS disconnected dept=%s", department)
