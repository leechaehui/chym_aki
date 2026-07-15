/** 이벤트 / Alert 도메인 타입 — 백엔드 CDSS 파이프라인과 직결(camelCase). */

/** WebSocket 으로 수신하는 ALERT_EVENT (지시서 §3.3). */
export interface AlertEvent {
  readonly eventType: "ALERT_EVENT";
  readonly alertId: string;
  readonly patientId: string;
  readonly subjectId?: number | null;
  readonly type: string; // AKI_CONFIRMED | AKI_SUSPECTED | PRE_AKI | SYSTEM
  readonly severity: "info" | "warning" | "critical";
  readonly priority: number;
  readonly dedupKey: string;
  readonly title: string;
  readonly message: string;
  /** 바로가기 딥링크(환자 Quick View 등). 없으면 프론트가 기본 경로로 폴백. */
  readonly link?: string | null;
  readonly akiStage?: string | null;
  readonly akiScore?: number | null;
  readonly department: string;
  readonly status: string;
  readonly sourceEventId?: string | null;
  readonly createdAt: string;
}

/** 영속 alert 레코드(GET /alerts). */
export interface AlertRecord {
  readonly id: string;
  readonly patientId: string;
  readonly subjectId?: number | null;
  readonly type: string;
  readonly severity: "info" | "warning" | "critical";
  readonly priority: number;
  readonly akiStage?: string | null;
  readonly akiScore?: number | null;
  readonly dedupKey: string;
  readonly title: string;
  readonly message: string;
  readonly sourceEventId?: string | null;
  readonly department: string;
  readonly status: string;
  readonly createdAt: string;
}

/** alert 감사 액션(지시서 §8). */
export type AlertAuditAction = "VIEWED" | "DISMISSED" | "ACKNOWLEDGED" | "ESCALATED";

/** 인제스트/시뮬 응답. */
export interface IngestResult {
  readonly accepted: number;
  readonly alerts: AlertRecord[];
}
