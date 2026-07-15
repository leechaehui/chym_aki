import type { ShapFeature } from "@/types";
import { cn } from "@/lib/cn";

/**
 * 환자별 SHAP 기여도 — 상위 피처를 좌우 막대로 표시(환자마다 다름).
 * 양수(위험 ↑)는 붉은색·오른쪽, 음수(위험 ↓)는 초록색·왼쪽.
 */
export function ShapBars({ features }: { features: readonly ShapFeature[] }) {
  if (features.length === 0) {
    return <p className="py-2 text-[11px] text-muted-foreground">이 환자의 피처 기여도를 계산할 수 없습니다.</p>;
  }
  const maxAbs = Math.max(
    ...features.map((f) => {
      const shapValue = typeof (f as { shap_value?: number }).shap_value === "number"
        ? (f as { shap_value?: number }).shap_value
        : f.contribution;
      return Math.abs(Number(shapValue ?? 0));
    }),
  ) || 1;

  return (
    <div className="space-y-1.5">
      {features.map((f) => {
        const shapValue = typeof (f as { shap_value?: number }).shap_value === "number" ? (f as { shap_value?: number }).shap_value : f.contribution;
        const safeShapValue = Number(shapValue ?? 0);
        const pct = (Math.abs(safeShapValue) / maxAbs) * 100;
        const up = safeShapValue >= 0;
        return (
          <div key={f.feature} className="grid grid-cols-[120px_1fr_56px] items-center gap-2 text-[11px]">
            <span className="truncate text-muted-foreground" title={f.label}>{f.label}</span>
            <span className="relative flex h-3 items-center">
              {/* 중앙 0선 기준 좌(↓)/우(↑) 막대 */}
              <span className="absolute left-1/2 top-0 h-full w-px bg-border" />
              <span
                className={cn("absolute h-2 rounded-sm", up ? "bg-destructive/70" : "bg-success/70")}
                style={up
                  ? { left: "50%", width: `${pct / 2}%` }
                  : { right: "50%", width: `${pct / 2}%` }}
              />
            </span>
            <span className={cn("text-right tabular-nums font-medium", up ? "text-destructive" : "text-success")}>
              {up ? "+" : ""}{safeShapValue.toFixed(3)}
            </span>
          </div>
        );
      })}
      <p className="pt-1 text-[10px] text-muted-foreground">
        + 위험을 높인 인자 · − 위험을 낮춘 인자 (모델이 본 표준화 피처 기준 정확 기여도)
      </p>
    </div>
  );
}
