import { RISK_BANDS, riskBandFor } from "@/lib/riskLabel";
import { cn } from "@/lib/cn";

/**
 * 위험 구간표(Risk Stratification) — 0–30 Low · 30–60 Moderate · 60–80 High · 80–100 Critical.
 * 현재 환자가 속한 구간을 강조한다.
 */
export function RiskStratification({ score }: { score: number }) {
  const current = riskBandFor(score);
  return (
    <div>
      <table className="w-full text-[11px]">
        <thead>
          <tr className="border-b border-border text-left text-muted-foreground">
            <th className="px-2 py-1 font-medium">범위</th>
            <th className="px-2 py-1 font-medium">등급</th>
          </tr>
        </thead>
        <tbody>
          {RISK_BANDS.map((b) => {
            const active = b.tier === current.tier;
            return (
              <tr
                key={b.tier}
                className={cn("border-b border-border/40", active && "bg-primary/10 font-semibold")}
              >
                <td className="px-2 py-1 tabular-nums">{b.range}</td>
                <td className={cn("px-2 py-1", b.textClass)}>
                  {b.dot} {b.label}
                  {active && <span className="ml-2 text-foreground">← 현재 ({score})</span>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="mt-1.5 text-[11px] font-semibold">
        <span className={current.textClass}>{current.dot} {current.label} Risk</span>
      </p>
    </div>
  );
}
