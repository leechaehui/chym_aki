import type { LucideIcon } from "lucide-react";
import { Info, AlertTriangle, ClipboardCheck, AlertOctagon } from "lucide-react";
import type { AppNotification, NotificationChannel, NotificationInput, Severity } from "@/types";

/** severity 별 표현 메타 — 채널/아이콘/색조/소멸 정책을 한 곳에 정의(단일 진실). */
export interface SeverityMeta {
  channel: NotificationChannel;
  label: string;
  icon: LucideIcon;
  /** Tailwind tone 키(배지/배경 색상 규칙). */
  tone: "info" | "warning" | "action" | "critical";
  /** 사용자가 닫을 수 있는지. CRITICAL 은 확인만 가능(닫기 불가). */
  dismissible: boolean;
  /** 자동 소멸(ms). Toast 만 사용. */
  autoDismissMs?: number;
}

export const SEVERITY_META: Record<Severity, SeverityMeta> = {
  INFO: { channel: "toast", label: "정보", icon: Info, tone: "info", dismissible: true, autoDismissMs: 4000 },
  WARNING: { channel: "banner", label: "주의", icon: AlertTriangle, tone: "warning", dismissible: true },
  ACTION_REQUIRED: { channel: "drawer", label: "조치 필요", icon: ClipboardCheck, tone: "action", dismissible: true },
  CRITICAL: { channel: "modal", label: "위급", icon: AlertOctagon, tone: "critical", dismissible: false },
};

let seq = 0;

/**
 * NotificationFactory (Factory 패턴).
 * - `create()`: 입력으로 완성된 AppNotification 을 만든다(id/시간/read 자동).
 * - `channelOf()`: severity → 렌더링 채널을 결정한다.
 * 호출부는 severity 만 지정하고, 어떤 컴포넌트로 렌더링될지는 Factory 가 캡슐화한다(OCP).
 */
export const NotificationFactory = {
  create(input: NotificationInput): AppNotification {
    return {
      id: `n-${Date.now()}-${++seq}`,
      createdAt: new Date().toISOString(),
      read: input.read ?? false,
      title: input.title,
      message: input.message,
      severity: input.severity,
      department: input.department,
      link: input.link,
      tone: input.tone,
      alertId: input.alertId,
    };
  },
  channelOf(severity: Severity): NotificationChannel {
    return SEVERITY_META[severity].channel;
  },
  metaOf(severity: Severity): SeverityMeta {
    return SEVERITY_META[severity];
  },
};
