import type { ChatMessage } from "@/types";
import { getToken } from "./http";

const WS_BASE = (import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api")
  .replace(/^http/, "ws");

type MessageHandler = (msg: ChatMessage) => void;

let socket: WebSocket | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let attempts = 0;
let closedByUs = false;
let activeRoomId: string | null = null;
let onMessage: MessageHandler | null = null;

function open(roomId: string): void {
  const token = getToken();
  if (!token) return;
  const ws = new WebSocket(`${WS_BASE}/ws/chat/${roomId}?token=${encodeURIComponent(token)}`);
  socket = ws;

  ws.onopen = () => {
    attempts = 0;
  };
  ws.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data);
      if (data?.type === "CHAT_MESSAGE" && onMessage) {
        onMessage(data as ChatMessage);
      }
    } catch {
      /* 비정상 메시지 무시 */
    }
  };
  ws.onclose = () => {
    socket = null;
    if (closedByUs) return;
    const delay = Math.min(1000 * 2 ** attempts++, 10000);
    reconnectTimer = setTimeout(() => {
      if (activeRoomId) open(activeRoomId);
    }, delay);
  };
  ws.onerror = () => {
    ws.close();
  };
}

export const chatSocket = {
  connect(roomId: string, handler: MessageHandler): () => void {
    closedByUs = false;
    activeRoomId = roomId;
    onMessage = handler;
    if (socket) {
      socket.close();
      socket = null;
    }
    open(roomId);
    return () => {
      closedByUs = true;
      activeRoomId = null;
      onMessage = null;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      socket?.close();
      socket = null;
      attempts = 0;
    };
  },

  send(body: string): void {
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ body }));
    }
  },

  // 소켓이 아직 연결 중이면 OPEN 될 때까지 최대 timeout ms 동안 재시도 후 전송.
  sendWhenReady(body: string, timeout = 4000): void {
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ body }));
      return;
    }
    const start = Date.now();
    const retry = () => {
      if (socket?.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ body }));
      } else if (Date.now() - start < timeout) {
        setTimeout(retry, 100);
      }
    };
    setTimeout(retry, 100);
  },
};
