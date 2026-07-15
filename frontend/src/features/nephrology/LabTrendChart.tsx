import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";
import type { LabTrendPoint } from "@/types";

/** 검사 결과 변화 추이 — Creatinine/eGFR/BUN 일별 라인 차트(Recharts). */
export function LabTrendChart({ data }: { data: LabTrendPoint[] }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data as LabTrendPoint[]} margin={{ top: 8, right: 12, left: -8, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
        <XAxis dataKey="date" tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }} tickFormatter={(d) => d.slice(5)} />
        <YAxis tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }} />
        <Tooltip
          contentStyle={{
            background: "hsl(var(--card))",
            border: "1px solid hsl(var(--border))",
            borderRadius: 8,
            fontSize: 12,
          }}
        />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Line type="monotone" dataKey="creatinine" name="Creatinine" stroke="#e24b4a" strokeWidth={2} dot={{ r: 3 }} />
        <Line type="monotone" dataKey="bun" name="BUN" stroke="#ef9f27" strokeWidth={2} dot={{ r: 3 }} />
        <Line type="monotone" dataKey="egfr" name="eGFR" stroke="#378add" strokeWidth={2} dot={{ r: 3 }} />
      </LineChart>
    </ResponsiveContainer>
  );
}
