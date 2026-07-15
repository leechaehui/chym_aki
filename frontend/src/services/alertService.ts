import type { AlertAuditAction, AlertRecord, IngestResult } from "@/types";
import { api } from "./http";

/**
 * Alert 서비스 (Facade) — CDSS 파이프라인의 REST 표면.
 * 실시간 알림은 alertSocket(WebSocket)이 담당하고, 여기서는 조회/액션/시뮬을 제공한다.
 */
class AlertService {
  /** 영속 alert 목록(우선순위순). 부서 미지정 시 서버가 현재 사용자 부서로 처리. */
  list(opts: { department?: string; status?: string; limit?: number } = {}): Promise<AlertRecord[]> {
    const q = new URLSearchParams();
    if (opts.department) q.set("department", opts.department);
    if (opts.status) q.set("status", opts.status);
    q.set("limit", String(opts.limit ?? 50));
    return api.get<AlertRecord[]>(`/alerts?${q.toString()}`);
  }

  /** alert 감사 액션 기록(VIEWED/DISMISSED/ACKNOWLEDGED/ESCALATED) — 상태 갱신 + AUDIT_EVENT. */
  recordAction(alertId: string, action: AlertAuditAction, role?: string): Promise<AlertRecord> {
    return api.post<AlertRecord>(`/alerts/${alertId}/audit`, { action, role });
  }

  /** ICU 스트림 시뮬레이션 — 다양한 LAB_EVENT 다발 발행(데모). */
  simulateIcu(n = 6): Promise<IngestResult> {
    return api.post<IngestResult>(`/events/lab/simulate-icu?n=${n}`);
  }

  /** 실제 MIMIC-IV ICU stay 인제스트 — public.* 시계열로 Safety 파이프라인 구동. */
  ingestFromStay(stayId: number): Promise<IngestResult> {
    return api.post<IngestResult>(`/events/lab/from-stay/${stayId}`);
  }
}

export const alertService = new AlertService();
