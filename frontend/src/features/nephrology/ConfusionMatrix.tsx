import type { ModelStageMetrics } from "@/types";
import { cn } from "@/lib/cn";

/**
 * 혼동 행렬(Confusion Matrix) — TP/FP/FN/TN 2×2 시각화.
 * 행=실제, 열=예측. 정답(TP/TN)은 초록 계열, 오류(FP/FN)는 붉은 계열로 강조.
 */
function Cell({ label, value, tone }: { label: string; value: number; tone: "good" | "bad" }) {
  return (
    <div className={cn(
      "flex flex-col items-center justify-center rounded-md border px-3 py-3",
      tone === "good" ? "border-success/40 bg-success/10" : "border-destructive/40 bg-destructive/10",
    )}>
      <span className="text-[10px] text-muted-foreground">{label}</span>
      <span className={cn("text-lg font-bold tabular-nums", tone === "good" ? "text-success" : "text-destructive")}>
        {value.toLocaleString()}
      </span>
    </div>
  );
}

export function ConfusionMatrix({ stage }: { stage: ModelStageMetrics }) {
  return (
    <div>
      <div className="grid grid-cols-[auto_1fr_1fr] gap-1 text-[10px]">
        <div />
        <div className="text-center font-medium text-muted-foreground">예측: AKI</div>
        <div className="text-center font-medium text-muted-foreground">예측: 정상</div>

        <div className="flex items-center font-medium text-muted-foreground">실제:<br />AKI</div>
        <Cell label="TP (정탐)" value={stage.tp} tone="good" />
        <Cell label="FN (놓침)" value={stage.fn} tone="bad" />

        <div className="flex items-center font-medium text-muted-foreground">실제:<br />정상</div>
        <Cell label="FP (오경보)" value={stage.fp} tone="bad" />
        <Cell label="TN (정음성)" value={stage.tn} tone="good" />
      </div>
      <p className="mt-1.5 text-[10px] text-muted-foreground">
        {stage.label} · 민감도 {(stage.recall * 100).toFixed(1)}% · 위음성률(놓침) {(stage.fnr * 100).toFixed(1)}%
      </p>
    </div>
  );
}
