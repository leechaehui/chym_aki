import type { AlertEvent, Department, NotificationInput, Severity } from "@/types";
import { useNotificationStore } from "@/store/notificationStore";
import { getToken } from "./http";
import { queryClient } from "@/lib/queryClient";

/**
 * ALERT WebSocket consumer (지시서 §6, §11).
 *
 * 백엔드 CDSS 파이프라인이 발행한 ALERT_EVENT 를 실시간 수신해 알림 채널로 라우팅한다.
 * - 알림은 **오직 서버 이벤트로만** 생성된다(지시서 §7 — render/state 트리거 금지).
 * - 연결은 부서당 1회만 수립하고, 끊기면 지수 백오프로 재연결한다.
 * - UI 는 consumer 일 뿐 — severity→채널 매핑만 하고 어떤 임상 판단도 하지 않는다.
 */

const WS_BASE = (import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api")
  .replace(/^http/, "ws");

/** 백엔드 severity → 프론트 Severity(채널). */
const SEVERITY_MAP: Record<AlertEvent["severity"], Severity> = {
  critical: "CRITICAL", // → Modal
  warning: "WARNING", //  → Banner
  info: "INFO", //        → Toast
};

let socket: WebSocket | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let attempts = 0;
let closedByUs = false;

function toInput(e: AlertEvent): NotificationInput {
  return {
    title: e.title,
    message: e.message,
    severity: SEVERITY_MAP[e.severity] ?? "WARNING",
    department: e.department as Department,
    // 서버가 환자별 딥링크를 실어주면 그걸 쓰고, 없으면 ICU AKI 목록으로 폴백.
    link: e.link ?? "/nephrology/icu-aki",
    alertId: e.alertId,
  };
}

function open(): void {
  const token = getToken();
  if (!token) return; // 비로그인 — 연결하지 않음
  const ws = new WebSocket(`${WS_BASE}/ws/alerts?token=${encodeURIComponent(token)}`);
  socket = ws;

  ws.onopen = () => {
    attempts = 0;
  };
  ws.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data) as AlertEvent;
      if (data?.eventType === "ALERT_EVENT") {
        // 시뮬레이션 시간 스킵 중이 아닐 때만 팝업 알림 노출
        const simulating = useNotificationStore.getState().isSimulating;
        if (!simulating) {
          useNotificationStore.getState().notify(toInput(data));
        }
        
        // CDSS 요구사항: 실제 라이브 이벤트 수신 시에만 백그라운드 리페치 트리거
        queryClient.invalidateQueries({ queryKey: ["alerts"] });
        queryClient.invalidateQueries({ queryKey: ["icuPatients"] });
      }
    } catch {
      /* 비정상 메시지 무시 */
    }
  };
  ws.onclose = () => {
    socket = null;
    if (closedByUs) return;
    // 지수 백오프 재연결(최대 ~10s).
    const delay = Math.min(1000 * 2 ** attempts++, 10000);
    reconnectTimer = setTimeout(open, delay);
  };
  ws.onerror = () => {
    ws.close();
  };
}

export const alertSocket = {
  /** 부서 알림 채널에 연결. 해제 함수 반환(notificationStore.startObserver 에서 사용). */
  connect(): () => void {
    closedByUs = false;
    if (!socket) open();
    return () => {
      closedByUs = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      socket?.close();
      socket = null;
      attempts = 0;
    };
  },
};
