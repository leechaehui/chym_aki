import type { AlertAuditAction } from "@/types";
import { alertService } from "@/services/alertService";

/**
 * alert 감사 액션 기록 — **1회성 fired flag**(지시서 §7).
 *
 * 같은 (alertId, action) 조합은 단 한 번만 서버에 기록한다. 리렌더/중복 표시로
 * useEffect 가 여러 번 돌아도 audit 가 중복 발사되지 않게 한다.
 * 서버 alert id 가 없는 알림(앱 내 도메인 이벤트)은 무시한다.
 */
const fired = new Set<string>();

export function fireAlertAudit(
  alertId: string | undefined,
  action: AlertAuditAction,
  role?: string,
): void {
  if (!alertId) return;
  const key = `${alertId}:${action}`;
  if (fired.has(key)) return;
  fired.add(key);
  void alertService.recordAction(alertId, action, role).catch(() => {
    // 실패 시 재시도 허용(중복 위험보다 누락 방지 우선 — 의료 추적성).
    fired.delete(key);
  });
}
