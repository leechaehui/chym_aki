import { cn } from "@/lib/cn";
import type { BadgeProps } from "@/components/ui/badge";

type Tone = NonNullable<BadgeProps["tone"]>;

const toneText: Record<Tone, string> = {
  neutral: "text-muted-foreground",
  info:    "text-info",
  success: "text-success",
  warning: "text-warning",
  high:    "text-high",
  danger:  "text-destructive",
  primary: "text-primary",
};

/** 상태 표시 — 박스 없이 컬러 텍스트만. */
export function StatusBadge({ label, tone }: { label: string; tone: Tone; dot?: boolean }) {
  return (
    <span className={cn("text-xs font-semibold", toneText[tone])}>
      {label}
    </span>
  );
}
