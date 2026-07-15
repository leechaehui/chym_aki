import { useEffect } from "react";
import { X } from "lucide-react";
import type { AppNotification } from "@/types";
import { SEVERITY_META } from "@/lib/notificationFactory";
import { useNotificationStore } from "@/store/notificationStore";
import { useAuthStore } from "@/store/authStore";
import { cn } from "@/lib/cn";

/** 단일 INFO 토스트 — 마운트 후 메타의 autoDismissMs 뒤 자동 해제. */
function ToastRow({ toast }: { toast: AppNotification }) {
  const dismiss = useNotificationStore((s) => s.dismissToast);
  const meta = SEVERITY_META[toast.severity];
  const Icon = meta.icon;
  useEffect(() => {
    if (!meta.autoDismissMs) return;
    const t = setTimeout(() => dismiss(toast.id), meta.autoDismissMs);
    return () => clearTimeout(t);
  }, [toast.id, dismiss, meta.autoDismissMs]);

  return (
    <div
      className={cn(
        "pointer-events-auto flex items-start gap-2.5 rounded-md border border-l-4 border-border border-l-accent bg-card p-3 shadow-lg",
        "[animation:chym-slide-in_.2s_ease]",
      )}
    >
      <Icon className="mt-0.5 size-4 shrink-0 text-accent" />
      <div className="flex-1">
        <p className="text-xs font-semibold text-foreground">{toast.title}</p>
        <p className="text-[11px] text-muted-foreground">{toast.message}</p>
      </div>
      <button onClick={() => dismiss(toast.id)} className="text-muted-foreground hover:text-foreground">
        <X className="size-3.5" />
      </button>
    </div>
  );
}

/** INFO 채널 — 우하단 토스트 스택. 현재 부서 대상 알림만 표시. */
export function ToastChannel() {
  const dept = useAuthStore((s) => s.user?.role);
  const toasts = useNotificationStore((s) => s.toasts);
  const mine = toasts.filter((t) => t.department === dept);
  if (mine.length === 0) return null;
  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-[60] flex w-80 flex-col gap-2">
      {mine.map((t) => (
        <ToastRow key={t.id} toast={t} />
      ))}
    </div>
  );
}
