import { useEffect, useState } from "react";
import { Bell, AlertTriangle, Info, ShieldAlert } from "lucide-react";
import type { AlertRecord } from "@/types";
import { alertService } from "@/services/alertService";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LoadingSpinner } from "@/components/common/LoadingSpinner";
import { EmptyState } from "@/components/common/EmptyState";
import { cn } from "@/lib/cn";
import { formatRelative } from "@/lib/format";

const SEV_STYLE: Record<AlertRecord["severity"], { icon: typeof Bell; cls: string }> = {
  critical: { icon: ShieldAlert, cls: "text-destructive" },
  warning:  { icon: AlertTriangle, cls: "text-warning" },
  info:     { icon: Info, cls: "text-blue-500" },
};

export function AlarmCenter() {
  const [alerts, setAlerts] = useState<AlertRecord[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    alertService
      .list({ limit: 30 })
      .then(setAlerts)
      .catch(() => setAlerts([]))
      .finally(() => setLoading(false));
  }, []);

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="flex-row items-center gap-2 pb-2">
        <Bell className="size-4 text-primary" />
        <CardTitle>알람 센터</CardTitle>
        {alerts.length > 0 && (
          <span className="ml-auto flex min-w-5 items-center justify-center rounded-full bg-destructive px-1.5 text-[10px] font-bold text-white">
            {alerts.length}
          </span>
        )}
      </CardHeader>
      <CardContent className="flex-1 min-h-0 overflow-y-auto p-2">
        {loading ? (
          <LoadingSpinner label="알람 불러오는 중" />
        ) : alerts.length === 0 ? (
          <EmptyState title="활성 알람 없음" />
        ) : (
          <ul className="flex flex-col gap-1.5">
            {alerts.map((a) => {
              const { icon: Icon, cls } = SEV_STYLE[a.severity];
              return (
                <li
                  key={a.id}
                  className={cn(
                    "flex items-start gap-2 rounded-md border px-2.5 py-2 text-xs",
                    a.severity === "critical"
                      ? "border-destructive/30 bg-destructive/5"
                      : a.severity === "warning"
                      ? "border-warning/30 bg-warning/5"
                      : "border-border bg-muted/30",
                  )}
                >
                  <Icon className={cn("mt-0.5 size-3.5 shrink-0", cls)} />
                  <div className="min-w-0 flex-1">
                    <p className="font-semibold text-foreground">{a.title}</p>
                    <p className="text-[11px] text-muted-foreground">{a.message}</p>
                  </div>
                  <span className="shrink-0 text-[10px] text-muted-foreground/70">
                    {formatRelative(a.createdAt)}
                  </span>
                </li>
              );
            })}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}
