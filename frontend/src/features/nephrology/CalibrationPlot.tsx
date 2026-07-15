import {
  CartesianGrid,
  Line,
  ComposedChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import type { Calibration } from "@/types";

/**
 * 신뢰도 곡선(Calibration plot) — ECE 숫자 대체 핵심 시각화.
 * x축 예측확률 vs y축 실제 발생률. 점선 대각선(완벽 보정)에 가까울수록 잘 보정된 모델.
 */
export function CalibrationPlot({ calibration }: { calibration: Calibration }) {
  const points = calibration.bins.map((b) => ({
    predicted: Math.round(b.predictedMean * 100),
    observed: Math.round(b.observedRate * 100),
    count: b.count,
  }));

  return (
    <div>
      <ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={points} margin={{ top: 8, right: 12, left: -8, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis
            type="number"
            dataKey="predicted"
            domain={[0, 100]}
            tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
            label={{ value: "예측 확률 (%)", position: "insideBottom", offset: -2, fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
          />
          <YAxis
            type="number"
            domain={[0, 100]}
            tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
            label={{ value: "실제 발생률 (%)", angle: -90, position: "insideLeft", offset: 14, fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
          />
          <ZAxis type="number" dataKey="count" range={[40, 320]} name="표본수" />
          <Tooltip
            contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8, fontSize: 12 }}
            formatter={(value, name) => {
              if (name === "표본수") return [value, "표본수"];
              return [`${value}%`, name === "observed" ? "실제 발생률" : "예측 확률"];
            }}
          />
          {/* 완벽 보정 대각선 */}
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 100, y: 100 }]} stroke="hsl(var(--muted-foreground))" strokeDasharray="5 5" />
          <Line type="monotone" dataKey="observed" stroke="hsl(var(--primary))" strokeWidth={2} dot={false} />
          <Scatter dataKey="observed" fill="hsl(var(--primary))" />
        </ComposedChart>
      </ResponsiveContainer>
      <p className="mt-1 text-center text-[10px] text-muted-foreground">
        점선(대각선)에 가까울수록 잘 보정됨 · 점 크기 = 구간 표본수 · ECE {calibration.ece.toFixed(3)} (n={calibration.n.toLocaleString()})
      </p>
    </div>
  );
}
