import {
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { RiskHistory } from "@/types";

/**
 * 위험도 추세 — Cr/기저치 기반 KDIGO '추정'(점선) + 실제 모델 출력 1점(◆).
 * 정직성: 추세선은 모델 출력이 아니라 Cr 파생 추정이며, 진짜 모델 점은 별도 마커로 구분한다.
 *
 * 두 시리즈를 hours 기준으로 **하나의 data 배열**로 합쳐 그린다(시리즈별 data 분리 시
 * 툴팁이 같은 라벨로 중복 표시되던 문제 방지). 각 행: { hours, estimate?, model? }.
 */
export function RiskHistoryChart({ history }: { history: RiskHistory }) {
  const byHour = new Map<number, { hours: number; estimate?: number; model?: number }>();
  for (const p of history.trajectory) {
    byHour.set(p.hours, { hours: p.hours, estimate: p.riskScore });
  }
  if (history.modelPoint) {
    const h = history.modelPoint.hours;
    const row = byHour.get(h) ?? { hours: h };
    row.model = history.modelPoint.riskScore;
    byHour.set(h, row);
  }
  const data = [...byHour.values()].sort((a, b) => a.hours - b.hours);

  if (data.length === 0) {
    return <p className="py-4 text-center text-[11px] text-muted-foreground">위험도 추세 데이터 없음</p>;
  }

  return (
    <div>
      <ResponsiveContainer width="100%" height={170}>
        <ComposedChart data={data} margin={{ top: 8, right: 12, left: -18, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
          <XAxis
            dataKey="hours"
            type="number"
            domain={["dataMin", "dataMax"]}
            tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
            tickFormatter={(h) => `${Math.round(h)}h`}
          />
          <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} width={34} />
          <Tooltip
            contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8, fontSize: 12 }}
            labelFormatter={(h) => `입실 후 ${Math.round(Number(h))}시간`}
            formatter={(v, name) => [v, name === "실제 모델 출력" ? "실제 모델 출력(48h)" : "Cr 기반 추정"]}
          />
          <Line
            name="Cr 기반 추정"
            dataKey="estimate"
            type="monotone"
            stroke="hsl(var(--warning))"
            strokeWidth={2}
            strokeDasharray="5 4"
            dot={{ r: 2 }}
            connectNulls
          />
          <Scatter name="실제 모델 출력" dataKey="model" fill="hsl(var(--primary))" shape="diamond" />
        </ComposedChart>
      </ResponsiveContainer>
      <p className="mt-1 text-[10px] leading-snug text-muted-foreground">
        <span className="font-medium text-warning">┄ Cr/기저치 기반 KDIGO 추정선</span>(모델 출력 아님) ·{" "}
        <span className="font-medium text-primary">◆ 실제 모델 예측(입실+48h, 단 1회)</span>
        <br />
        {history.disclaimer}
      </p>
    </div>
  );
}
