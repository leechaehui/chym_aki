import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";

export interface MetricPoint {
  date: string;
  value: number;
}

/**
 * 단일 지표 추이(미니 영역 차트) — Cr/eGFR/소변량 등 한 지표를 작게 보여준다.
 * NephrologyWorkspace 우측 환자 상세에서 지표별로 재사용.
 */
export function MetricTrendChart({
  data,
  color,
  unit,
  /** 참조선(예: 핍뇨 기준 0.5 mL/kg/hr) — 있으면 점선으로 표시. */
  refLine,
  refLabel,
  /** 참조선 색상(미지정 시 위험색). KDIGO AKI 기준=위험색, CKD 참고선 등은 다른 색으로 구분. */
  refColor = "hsl(var(--destructive))",
  height = 120,
}: {
  data: MetricPoint[];
  color: string;
  unit?: string;
  refLine?: number;
  refLabel?: string;
  refColor?: string;
  height?: number;
}) {
  const gradId = `grad-${color.replace(/[^a-z0-9]/gi, "")}`;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 6, right: 10, left: -18, bottom: 0 }}>
        <defs>
          <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.28} />
            <stop offset="100%" stopColor={color} stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
        <XAxis dataKey="date" tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} tickFormatter={(d) => d.slice(5)} />
        <YAxis tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} width={34} />
        <Tooltip
          contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8, fontSize: 12 }}
          formatter={(v) => [`${v}${unit ? ` ${unit}` : ""}`, ""]}
        />
        {refLine !== undefined && (
          <ReferenceLine
            y={refLine}
            stroke={refColor}
            strokeDasharray="4 4"
            // 기준선이 데이터 Y범위 밖이어도 축을 확장해 **항상** 표시(기본 discard 면 누락됨).
            ifOverflow="extendDomain"
            label={{ value: refLabel, position: "insideTopRight", fontSize: 9, fill: refColor }}
          />
        )}
        <Area type="monotone" dataKey="value" stroke={color} strokeWidth={2} fill={`url(#${gradId})`} dot={{ r: 2.5 }} />
      </AreaChart>
    </ResponsiveContainer>
  );
}
