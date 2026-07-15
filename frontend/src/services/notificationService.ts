import type { AppNotification, Department } from "@/types";
import { eventBus, type DomainEvent } from "@/features/notification/events";
import { api } from "./http";

/**
 * 부서별 데모 이벤트(서버 푸시 시뮬레이션).
 * 로그인한 부서가 4개 채널(Toast/Banner/Drawer/Modal)을 모두 체험하도록
 * 대표 이벤트를 시간차로 발행한다. 실제로는 WebSocket/SSE 수신부로 교체할 지점.
 */
const DEMO_EVENTS: Record<Department, DomainEvent[]> = {
  nephrology: [
    { type: "pathology.resultArrived", patientName: "정환자", mrn: "AKI-100231" }, // INFO → Toast
    { type: "consult.arrivedNeph", patientName: "한복막", mrn: "AKI-100244" }, // ACTION_REQUIRED → Drawer
    { type: "aki.stage3", patientName: "오신우", mrn: "AKI-100258" }, // CRITICAL → Modal
  ],
  pathology: [
    { type: "pathology.readSaved", patientName: "정환자" }, // INFO → Toast
    { type: "consult.requested", patientName: "한복막", mrn: "AKI-100244", urgency: "긴급", consultId: "c-001" }, // ACTION_REQUIRED → Drawer
    { type: "consult.urgentRead", patientName: "한복막", consultId: "c-001" }, // CRITICAL → Modal
  ],
  admin: [],
};

class NotificationService {
  /**
   * 부서 알림 히스토리(영속) 조회 — 드로어 초기 적재용. 서버가 AppNotification 형태로 반환.
   * 라이브 이벤트(simulate)는 이와 별개로 클라이언트에서 발행된다.
   */
  async history(dept: Department): Promise<AppNotification[]> {
    return api.get<AppNotification[]>(`/notifications?department=${dept}`);
  }

  /**
   * 읽음 처리 영속화(best-effort). 서버에 없는 라이브 알림(클라이언트 생성 id)은
   * 404 가 날 수 있으므로 실패를 무시한다 — UI 의 낙관적 갱신이 단일 진실원.
   */
  async markRead(id: string): Promise<void> {
    try {
      await api.post(`/notifications/${id}/read`);
    } catch {
      /* 라이브(미영속) 알림 — 무시 */
    }
  }

  /**
   * 로그인 부서의 데모 이벤트를 staggered 발행한다(2.6s 간격). 해제 함수 반환.
   * 실제 운영 시: 이 메서드를 WebSocket/SSE 구독으로 교체(eventBus.publish 는 그대로 사용).
   */
  simulate(dept: Department): () => void {
    const events = DEMO_EVENTS[dept] ?? [];
    const timers = events.map((e, i) => setTimeout(() => eventBus.publish(e), 2600 + i * 2800));
    return () => timers.forEach(clearTimeout);
  }
}

export const notificationService = new NotificationService();
