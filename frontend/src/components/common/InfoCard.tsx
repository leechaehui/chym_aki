import { cn } from "@/lib/cn";

export interface InfoItem {
  label: string;
  value: React.ReactNode;
}

/** 키-값 정보 그리드 — 환자 기본정보 등 라벨/값 쌍을 정렬 표시. */
export function InfoCard({ items, columns = 2, className }: { items: InfoItem[]; columns?: 2 | 3 | 4; className?: string }) {
  const colClass = { 2: "grid-cols-2", 3: "grid-cols-3", 4: "grid-cols-4" }[columns];
  return (
    <div className={cn("grid gap-x-4 gap-y-3", colClass, className)}>
      {items.map((it) => (
        <div key={it.label}>
          <dt className="text-[11px] text-muted-foreground">{it.label}</dt>
          <dd className="mt-0.5 text-sm font-medium text-foreground">{it.value}</dd>
        </div>
      ))}
    </div>
  );
}
