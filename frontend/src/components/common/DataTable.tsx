import { cn } from "@/lib/cn";
import { EmptyState } from "./EmptyState";

/** 컬럼 정의 — header 와 cell 렌더러(접근자)로 표를 선언적으로 구성. */
export interface Column<T> {
  key: string;
  header: React.ReactNode;
  /** 셀 렌더러. 없으면 row[key] 표시. */
  render?: (row: T) => React.ReactNode;
  className?: string;
  align?: "left" | "right" | "center";
}

/**
 * 행 심각도 → 좌측 border 색상(EMR 표준).
 * 색은 보조 강조 수단 — 행 내용(텍스트/배지)이 1차 정보다.
 */
export type RowTone = "critical" | "warning";
const rowToneCls: Record<RowTone, string> = {
  critical: "border-l-2 border-l-destructive",
  warning: "border-l-2 border-l-high",
};

/**
 * 제네릭 데이터 테이블 — 컬럼 설정 + 데이터로 어떤 목록이든 재사용.
 * 행 클릭/선택 강조/빈 상태를 표준화한다(DRY).
 */
export function DataTable<T>({
  columns,
  data,
  rowKey,
  onRowClick,
  selectedKey,
  rowTone,
  zebra = true,
  emptyTitle = "데이터가 없습니다",
  emptyDescription,
  className,
}: {
  columns: Column<T>[];
  data: T[];
  rowKey: (row: T) => string;
  onRowClick?: (row: T) => void;
  selectedKey?: string | null;
  /** 행 심각도 접근자 → 좌측 border(critical=빨강 / warning=주황). 없으면 무색. */
  rowTone?: (row: T) => RowTone | null | undefined;
  /** zebra 줄무늬(짝수 행 음영). 기본 true. */
  zebra?: boolean;
  emptyTitle?: string;
  emptyDescription?: string;
  className?: string;
}) {
  if (data.length === 0) return <EmptyState title={emptyTitle} description={emptyDescription} />;

  const alignClass = { left: "text-left", right: "text-right", center: "text-center" };

  return (
    <div className={cn("overflow-x-auto", className)}>
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="border-b border-border">
            {columns.map((c) => (
              <th
                key={c.key}
                className={cn(
                  "px-3 py-2.5 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground",
                  alignClass[c.align ?? "left"],
                  c.className,
                )}
              >
                {c.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => {
            const key = rowKey(row);
            const selected = selectedKey === key;
            const tone = rowTone?.(row);
            return (
              <tr
                key={key}
                onClick={() => onRowClick?.(row)}
                className={cn(
                  "border-b border-border/60 transition-colors",
                  // zebra: 짝수 행만 옅은 회색(#F8FAFC ≈ secondary) — 선택/심각도 강조가 우선
                  zebra && i % 2 === 1 && "bg-secondary/50",
                  onRowClick && "cursor-pointer hover:bg-muted/60",
                  tone && rowToneCls[tone],
                  selected && "bg-primary/5",
                )}
              >
                {columns.map((c) => (
                  <td key={c.key} className={cn("px-3 py-2.5 text-foreground", alignClass[c.align ?? "left"], c.className)}>
                    {c.render ? c.render(row) : (row as Record<string, React.ReactNode>)[c.key]}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
