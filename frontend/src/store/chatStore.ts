import { create } from "zustand";
import type { ChatMessage, ChatRoom, ChatUser } from "@/types";
import { api } from "@/services/http";
import { chatSocket } from "@/services/chatSocket";

interface ChatState {
  panelOpen: boolean;
  users: ChatUser[];
  rooms: ChatRoom[];
  activeRoomId: string | null;
  messages: Record<string, ChatMessage[]>;
  _disconnect: (() => void) | null;

  togglePanel: () => void;
  openPanel: () => void;
  closePanel: () => void;
  loadUsers: () => Promise<void>;
  loadRooms: () => Promise<void>;
  openRoom: (peerId: string) => Promise<void>;
  closeRoom: () => void;
  sendMessage: (body: string) => void;
  receiveMessage: (msg: ChatMessage) => void;
  loadMessages: (roomId: string) => Promise<void>;
  /** 병리과 협진 회신 시 → 요청자에게 채팅 자동 전송 후 패널 열기. */
  openForConsultReply: (requesterName: string, draft: string) => Promise<void>;
  /** 신장내과 협진 요청 시 → 병리과 의사에게 채팅 알림 자동 전송(패널 미열기). */
  notifyConsultRequest: (patientName: string, urgencyLabel: string, keyLabs: string) => Promise<void>;
}

export const useChatStore = create<ChatState>((set, get) => ({
  panelOpen: false,
  users: [],
  rooms: [],
  activeRoomId: null,
  messages: {},
  _disconnect: null,

  togglePanel() {
    const next = !get().panelOpen;
    set({ panelOpen: next });
    if (next && get().users.length === 0) {
      get().loadUsers();
      get().loadRooms();
    }
  },
  openPanel() {
    set({ panelOpen: true });
    if (get().users.length === 0) {
      get().loadUsers();
      get().loadRooms();
    }
  },
  closePanel() {
    get().closeRoom();
    set({ panelOpen: false });
  },

  async loadUsers() {
    const data = await api.get<ChatUser[]>("/chat/users");
    set({ users: data });
  },

  async loadRooms() {
    const data = await api.get<ChatRoom[]>("/chat/rooms");
    set({ rooms: data });
  },

  async loadMessages(roomId: string) {
    const msgs = await api.get<ChatMessage[]>(`/chat/rooms/${roomId}/messages`);
    set((s) => ({ messages: { ...s.messages, [roomId]: msgs } }));
  },

  async openRoom(peerId: string) {
    const room = await api.post<ChatRoom>(`/chat/rooms/${peerId}`);
    set((s) => {
      const exists = s.rooms.find((r) => r.id === room.id);
      return { rooms: exists ? s.rooms : [room, ...s.rooms] };
    });

    await get().loadMessages(room.id);

    get()._disconnect?.();
    const disconnect = chatSocket.connect(room.id, (msg) => {
      get().receiveMessage(msg);
    });
    set({ activeRoomId: room.id, _disconnect: disconnect });
  },

  closeRoom() {
    get()._disconnect?.();
    set({ activeRoomId: null, _disconnect: null });
  },

  sendMessage(body: string) {
    chatSocket.send(body);
  },

  receiveMessage(msg: ChatMessage) {
    set((s) => ({
      messages: {
        ...s.messages,
        [msg.roomId]: [...(s.messages[msg.roomId] ?? []), msg],
      },
    }));
  },

  async openForConsultReply(requesterName, draft) {
    set({ panelOpen: true });
    if (get().users.length === 0) {
      await Promise.all([get().loadUsers(), get().loadRooms()]).catch(() => {});
    }
    const match = get().users.find((u) => u.name === requesterName);
    if (!match) return;
    await get().openRoom(match.id).catch(() => {});
    chatSocket.sendWhenReady(draft);
  },

  async notifyConsultRequest(patientName, urgencyLabel, keyLabs) {
    if (get().users.length === 0) {
      await get().loadUsers().catch(() => {});
    }
    const pathologist = get().users.find((u) => u.role === "pathology");
    if (!pathologist) return;
    await get().openRoom(pathologist.id).catch(() => {});
    chatSocket.sendWhenReady(
      `[협진 요청 · ${patientName}]\n긴급도: ${urgencyLabel}\n주요검사: ${keyLabs}`,
    );
  },
}));
