import { cn } from "@/lib/cn";
import { getRiskLabel } from "@/lib/riskLabel";

type Band = { label: string; ring: string; text: string; bar: string };

/** predictedStage → 위험 구간(색상/문구). 3단계(고위험/주의/안정) 다 명시 —
 * "그 외 주의"로 뭉뚱그리면 Non-AKI 도 안정이 아니라 주의로 잘못 뜬다(과거 버그, riskLabel.ts와 동일 원인). */
function bandFor(_score: number, predictedStage?: string | null): Band {
  const risk = getRiskLabel(predictedStage);
  if (risk.label === "고위험") return { label: risk.label, ring: "text-destructive", text: "text-destructive", bar: "bg-destructive" };
  if (risk.label === "주의") return { label: risk.label, ring: "text-warning", text: "text-warning", bar: "bg-warning" };
  return { label: "안정", ring: "text-success", text: "text-success", bar: "bg-success" };
}

/**
 * AI Risk Score 게이지 — 0~100 점수를 반원 게이지 + 위험 구간으로 표시.
 * 신장내과 환자 상세에서 모델 종합 위험도를 한눈에 보여준다.
 */
export function AiRiskGauge({ score, predictedStage }: { score: number; predictedStage?: string | null }) {
  const band = bandFor(score, predictedStage);
  const r = 62;
  const circ = Math.PI * r;
  const offset = circ * (1 - score / 100);

  return (
    <div className="flex h-full flex-col items-center justify-center gap-3">
      <div className="relative h-[120px] w-[220px]">
        <svg viewBox="0 0 180 100" className="h-full w-full">
          <path d="M 18 84 A 72 72 0 0 1 162 84" fill="none" stroke="hsl(var(--muted))" strokeWidth="14" strokeLinecap="round" />
          <path
            d="M 18 84 A 72 72 0 0 1 162 84"
            fill="none"
            className={band.ring}
            stroke="currentColor"
            strokeWidth="14"
            strokeLinecap="round"
            strokeDasharray={circ}
            strokeDashoffset={offset}
          />
        </svg>
        <div className="absolute inset-x-0 bottom-0 flex flex-col items-center">
          <span className={cn("text-4xl font-bold leading-none", band.text)}>{score}</span>
          <span className="text-[10px] text-muted-foreground">/ 100</span>
        </div>
      </div>

      <div className="flex items-center gap-1.5">
        <span className={cn("text-sm font-semibold", band.text)}>{band.label}</span>
      </div>
      <p className="text-center text-[11px] leading-snug text-muted-foreground">
        AI가 Stage 2-3 AKI를 예측한 점수입니다.
      </p>
    </div>
  );
}
