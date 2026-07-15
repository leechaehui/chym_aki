import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/cn";
import { Card } from "@/components/ui/card";

type Tone = "primary" | "accent" | "success" | "warning" | "danger" | "neutral";

const toneMap: Record<Tone, string> = {
  primary: "bg-primary/10 text-primary",
  accent: "bg-accent/10 text-accent",
  success: "bg-success/10 text-success",
  warning: "bg-warning/15 text-warning",
  danger: "bg-destructive/10 text-destructive",
  neutral: "bg-secondary text-secondary-foreground",
};

/** KPI 카드 — 상단 지표 표시(값/라벨/아이콘 + 보조 텍스트). */
export function StatCard({
  label,
  value,
  unit,
  icon: Icon,
  tone = "primary",
  hint,
}: {
  label: string;
  value: React.ReactNode;
  unit?: string;
  icon?: LucideIcon;
  tone?: Tone;
  hint?: string;
}) {
  return (
    <Card className="flex items-center gap-3 p-4">
      {Icon && (
        <div className={cn("flex size-10 shrink-0 items-center justify-center rounded-lg", toneMap[tone])}>
          <Icon className="size-5" />
        </div>
      )}
      <div className="min-w-0">
        <p className="truncate text-xs text-muted-foreground">{label}</p>
        <p className="text-xl font-bold leading-tight text-foreground">
          {value}
          {unit && <span className="ml-0.5 text-xs font-medium text-muted-foreground">{unit}</span>}
        </p>
        {hint && <p className="truncate text-[11px] text-muted-foreground">{hint}</p>}
      </div>
    </Card>
  );
}
