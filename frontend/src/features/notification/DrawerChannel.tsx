import { useNavigate } from "react-router-dom";
import { Bell, X } from "lucide-react";
import type { AppNotification } from "@/types";
import { SEVERITY_META } from "@/lib/notificationFactory";
import { useNotificationStore } from "@/store/notificationStore";
import { useAuthStore } from "@/store/authStore";
import { formatRelative } from "@/lib/format";
import { cn } from "@/lib/cn";
import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/common/EmptyState";

const toneCls = {
  info: "bg-accent/10 text-accent",
  warning: "bg-warning/15 text-warning",
  action: "bg-primary/10 text-primary",
  critical: "bg-destructive/10 text-destructive",
} as const;

function Row({ n, onClose }: { n: AppNotification; onClose: () => void }) {
  const navigate = useNavigate();
  const markRead = useNotificationStore((s) => s.markRead);
  const meta = SEVERITY_META[n.severity];
  const Icon = meta.icon;
  return (
    <button
      onClick={() => {
        markRead(n.id);
        if (n.link) {
          navigate(n.link);
          onClose();
        }
      }}
      className={cn(
        "flex w-full gap-3 border-b border-border p-3 text-left transition-colors hover:bg-muted/50",
        !n.read && "bg-primary/[0.03]",
      )}
    >
      <div className={cn("flex size-8 shrink-0 items-center justify-center rounded-lg", toneCls[meta.tone])}>
        <Icon className="size-4" />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-center justify-between gap-2">
          <p className="truncate text-xs font-semibold text-foreground">{n.title}</p>
          {!n.read && <span className="size-2 shrink-0 rounded-full bg-primary" />}
        </div>
        <p className="mt-0.5 line-clamp-2 text-[11px] text-muted-foreground">{n.message}</p>
        <p className="mt-1 text-[10px] text-muted-foreground/70">
          {meta.label} · {formatRelative(n.createdAt)}
        </p>
      </div>
    </button>
  );
}

/**
 * ACTION_REQUIRED 채널 — 우측 슬라이드 드로어(사이드 패널).
 * 부서 알림 히스토리를 보여주고, ACTION_REQUIRED 발생 시 자동으로 열린다.
 */
export function DrawerChannel() {
  const dept = useAuthStore((s) => s.user?.role);
  const open = useNotificationStore((s) => s.drawerOpen);
  const setOpen = useNotificationStore((s) => s.setDrawerOpen);
  // 셀렉터는 안정적인 items 참조만 반환하고, 부서 필터는 렌더에서 수행한다.
  // (셀렉터가 매번 새 배열을 반환하면 useSyncExternalStore 가 무한 루프에 빠진다 — React #185)
  const allItems = useNotificationStore((s) => s.items);
  const markAllRead = useNotificationStore((s) => s.markAllRead);
  const items = allItems.filter((n) => n.department === dept);

  return (
    <>
      <div
        className={cn("fixed inset-0 z-40 bg-black/30 transition-opacity", open ? "opacity-100" : "pointer-events-none opacity-0")}
        onClick={() => setOpen(false)}
      />
      <aside
        className={cn(
          "fixed right-0 top-0 z-50 flex h-full w-[360px] max-w-[88vw] flex-col border-l border-border bg-card shadow-2xl transition-transform duration-200",
          open ? "translate-x-0" : "translate-x-full",
        )}
      >
        <header className="flex items-center justify-between border-b border-border px-4 py-3">
          <div className="flex items-center gap-2">
            <Bell className="size-4 text-primary" />
            <h2 className="text-sm font-semibold">알림 센터</h2>
          </div>
          <div className="flex items-center gap-1">
            <Button variant="ghost" size="sm" onClick={() => dept && markAllRead(dept)} className="text-[11px]">
              모두 읽음
            </Button>
            <Button variant="ghost" size="icon" onClick={() => setOpen(false)}>
              <X className="size-4" />
            </Button>
          </div>
        </header>
        <div className="flex-1 overflow-y-auto">
          {items.length === 0 ? (
            <EmptyState icon={Bell} title="알림이 없습니다" />
          ) : (
            items.map((n) => <Row key={n.id} n={n} onClose={() => setOpen(false)} />)
          )}
        </div>
      </aside>
    </>
  );
}
