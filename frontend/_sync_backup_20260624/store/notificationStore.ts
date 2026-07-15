import { create } from "zustand";
import type { AppNotification, NotificationInput } from "@/types";
import { NotificationFactory } from "@/lib/notificationFactory";
import { eventBus, eventToNotification } from "@/features/notification/events";
import { alertSocket } from "@/services/alertSocket";
import { notificationService } from "@/services/notificationService";
import type { Department } from "@/types";

interface NotificationState {
  /** 영속 알림 로그(INFO 토스트 제외 — 드로어 히스토리). */
  items: AppNotification[];
  /** INFO 활성 토스트(휘발성). */
  toasts: AppNotification[];
  /** WARNING 활성 배너. */
  banners: AppNotification[];
  /** CRITICAL 모달 큐. */
  modalQueue: AppNotification[];
  /** ACTION_REQUIRED 우측 드로어 열림 여부. */
  drawerOpen: boolean;
  /** 시드 1회 적재 여부 — 부서 전환 재시드로 라이브 알림이 지워지지 않게 한다. */
  seeded: boolean;

  /** Factory 로 생성 후 severity→채널로 라우팅한다(알림 흐름의 단일 진입점). */
  notify: (input: NotificationInput) => AppNotification;
  /** 초기 시드(부서별 히스토리). */
  seed: (inputs: NotificationInput[]) => void;
  /** 서버 히스토리로 초기 적재(서버 id 보존). seed 와 달리 Factory 를 거치지 않는다. */
  hydrate: (items: AppNotification[]) => void;
  /** Observer 시작 — 이벤트 버스 구독 + 부서별 라이브 시뮬. 해제 함수 반환. */
  startObserver: (dept: Department) => () => void;

  dismissToast: (id: string) => void;
  dismissBanner: (id: string) => void;
  dismissModal: () => void;
  setDrawerOpen: (open: boolean) => void;
  markRead: (id: string) => void;
  markAllRead: (dept: Department) => void;
}

/** severity 채널에 따라 상태 버킷으로 라우팅. */
function route(state: NotificationState, n: AppNotification): Partial<NotificationState> {
  const channel = NotificationFactory.channelOf(n.severity);
  // 토스트를 제외한 모든 알림은 드로어 히스토리에 적재
  const base = channel === "toast" ? {} : { items: [n, ...state.items] };
  switch (channel) {
    case "toast":
      return { ...base, toasts: [...state.toasts, n] };
    case "banner":
      return { ...base, banners: [n, ...state.banners] };
    case "drawer":
      return { ...base, drawerOpen: true }; // 드로어 자동 열림 + items 에 표시
    case "modal":
      return { ...base, modalQueue: [...state.modalQueue, n] };
  }
}

export const useNotificationStore = create<NotificationState>((set, get) => ({
  items: [],
  toasts: [],
  banners: [],
  modalQueue: [],
  drawerOpen: false,
  seeded: false,

  notify(input) {
    const n = NotificationFactory.create(input);
    set((state) => route(state, n));
    return n;
  },

  seed(inputs) {
    // 1회만 적재. 부서 전환 시 AppLayout 이펙트가 재호출해도, 그 사이 도착한 라이브 알림
    // (예: 신장내과→병리과 협진 요청)을 덮어쓰지 않도록 이미 시드했으면 건너뛴다.
    if (get().seeded) return;
    const created = inputs.map((i) => NotificationFactory.create(i));
    // 시드는 히스토리에만 적재(자동 팝업 없음)
    set((s) => ({ items: [...s.items, ...created.filter((n) => n.severity !== "INFO")], seeded: true }));
  },

  hydrate(items) {
    // 서버 히스토리 1회 적재(서버 id 보존 → 읽음 처리 영속화 가능).
    if (get().seeded) return;
    set((s) => ({ items: [...s.items, ...items.filter((n) => n.severity !== "INFO")], seeded: true }));
  },

  startObserver(dept) {
    // (1) 앱 내 도메인 이벤트(협진 딥링크 등) — 기존 eventBus consumer 유지.
    const off = eventBus.subscribe((e) => get().notify(eventToNotification(e)));
    // (2) CDSS 파이프라인 실시간 알림 — WebSocket consumer(서버 ALERT_EVENT).
    //     지시서 §7: 알림은 서버 이벤트로만 발생(클라이언트 시뮬/state 트리거 금지).
    const stopSocket = alertSocket.connect();
    void dept; // 부서는 서버가 토큰 role 로 라우팅 → 클라이언트 인자 불필요.
    return () => {
      off();
      stopSocket();
    };
  },

  dismissToast(id) {
    set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) }));
  },
  dismissBanner(id) {
    set((s) => ({ banners: s.banners.filter((b) => b.id !== id) }));
  },
  dismissModal() {
    set((s) => ({ modalQueue: s.modalQueue.slice(1) }));
  },
  setDrawerOpen(open) {
    set({ drawerOpen: open });
  },
  markRead(id) {
    set((s) => ({ items: s.items.map((n) => (n.id === id ? { ...n, read: true } : n)) }));
    void notificationService.markRead(id); // 서버 영속화(라이브 알림은 무시됨)
  },
  markAllRead(dept) {
    const unread = get().items.filter((n) => n.department === dept && !n.read);
    set((s) => ({ items: s.items.map((n) => (n.department === dept ? { ...n, read: true } : n)) }));
    unread.forEach((n) => void notificationService.markRead(n.id)); // 각 건 서버 영속화
  },
}));
