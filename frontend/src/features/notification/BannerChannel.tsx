import { useEffect } from "react";
import { AlertTriangle, CheckCircle2, AlertOctagon, X } from "lucide-react";
import type { NotificationTone } from "@/types";
import { useNotificationStore } from "@/store/notificationStore";
import { useAuthStore } from "@/store/authStore";
import { fireAlertAudit } from "@/lib/alertAudit";
import { cn } from "@/lib/cn";

/** 색조별 배너 스타일 + 아이콘. tone 미지정 시 warning(주황)이 기본. */
const toneStyle: Record<NotificationTone, { wrap: string; icon: typeof AlertTriangle }> = {
  danger: { wrap: "border-destructive/30 bg-destructive/15 text-destructive", icon: AlertOctagon },
  warning: { wrap: "border-warning/30 bg-warning/15 text-warning", icon: AlertTriangle },
  success: { wrap: "border-success/30 bg-success/15 text-success", icon: CheckCircle2 },
  info: { wrap: "border-accent/30 bg-accent/15 text-accent", icon: AlertTriangle },
};

/**
 * WARNING 채널 — 화면 상단 고정 배너. 현재 부서 대상의 가장 최근 경고 1건을 노출한다.
 * 색조(tone)가 있으면 그 색으로(예: ICU 잔여 0=빨강 / 1~2=주황 / 3+=녹색), 없으면 주황 기본.
 * 부서 필터로 응급의학과 등 대상 직군에게만 표시된다.
 * CDSS alert(alertId 보유)는 표시 시 VIEWED, 닫을 때 DISMISSED 를 기록한다.
 */
export function BannerChannel() {
  const dept = useAuthStore((s) => s.user?.role);
  const banner = useNotificationStore((s) => s.banners.find((b) => b.department === dept) ?? null);
  const dismiss = useNotificationStore((s) => s.dismissBanner);

  // 배너가 표시되면 VIEWED 기록(1회성).
  useEffect(() => {
    if (banner?.alertId) fireAlertAudit(banner.alertId, "VIEWED", dept);
  }, [banner?.alertId, dept]);

  if (!banner) return null;
  const style = toneStyle[banner.tone ?? "warning"];
  const Icon = style.icon;
  return (
    <div className={cn("flex items-center gap-2 border-b px-4 py-2", style.wrap)}>
      <Icon className="size-4 shrink-0" />
      <p className="flex-1 text-xs font-medium">
        <span className="font-semibold">{banner.title}</span> · {banner.message}
      </p>
      <button
        onClick={() => {
          fireAlertAudit(banner.alertId, "DISMISSED", dept);
          dismiss(banner.id);
        }}
        className="opacity-80 hover:opacity-100"
      >
        <X className="size-4" />
      </button>
    </div>
  );
}
