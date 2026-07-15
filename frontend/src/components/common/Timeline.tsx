import { cn } from "@/lib/cn";

export interface TimelineItem {
  id: string;
  title: string;
  meta?: string;
  description?: string;
  /** 완료 여부 — 점 색상/연결선 강조. */
  done?: boolean;
}

/** 세로 타임라인 — 협진 진행 단계(요청→접수→분석→판독→회신) 표시. */
export function Timeline({ items }: { items: TimelineItem[] }) {
  return (
    <ol className="relative ml-2 border-l border-border">
      {items.map((it) => (
        <li key={it.id} className="mb-4 ml-4 last:mb-0">
          <span
            className={cn(
              "absolute -left-[7px] mt-1 size-3 rounded-full border-2",
              it.done ? "border-primary bg-primary" : "border-border bg-card",
            )}
          />
          <div className="flex items-center justify-between gap-2">
            <p className="text-sm font-medium text-foreground">{it.title}</p>
            {it.meta && <span className="shrink-0 text-[11px] text-muted-foreground">{it.meta}</span>}
          </div>
          {it.description && <p className="mt-0.5 text-xs text-muted-foreground">{it.description}</p>}
        </li>
      ))}
    </ol>
  );
}
