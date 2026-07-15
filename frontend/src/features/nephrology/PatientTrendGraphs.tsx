import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { PatientTrends, TrendPoint } from "@/types";

/** 입실 후 경과시간(hours) x축 미니 추세 차트. */
function HoursTrendChart({
  data,
  color,
  unit,
  refLine,
  refLabel,
  refColor = "hsl(var(--destructive))",
}: {
  data: readonly TrendPoint[];
  color: string;
  unit?: string;
  refLine?: number;
  refLabel?: string;
  refColor?: string;
}) {
  if (data.length === 0) {
    return <p className="py-6 text-center text-[11px] text-muted-foreground">데이터 없음</p>;
  }
  const gradId = `trend-${color.replace(/[^a-z0-9]/gi, "")}`;
  return (
    <ResponsiveContainer width="100%" height={120}>
      <AreaChart data={data as TrendPoint[]} margin={{ top: 6, right: 10, left: -20, bottom: 0 }}>
        <defs>
          <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.28} />
            <stop offset="100%" stopColor={color} stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
        <XAxis
          dataKey="hours"
          type="number"
          domain={["dataMin", "dataMax"]}
          tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
          tickFormatter={(h) => `${Math.round(h)}h`}
        />
        <YAxis tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} width={34} />
        <Tooltip
          contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8, fontSize: 12 }}
          labelFormatter={(h) => `입실 후 ${Math.round(Number(h))}시간`}
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
        <Area type="monotone" dataKey="value" stroke={color} strokeWidth={2} fill={`url(#${gradId})`} dot={{ r: 2 }} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

/**
 * 환자별 Cr · eGFR · 소변량 추세(개별 데이터) + KDIGO 기준선.
 * - Cr: AKI Stage 1 진입선 = min(기저×1.5, 기저+0.3) — 둘 중 먼저 충족되는 값(KDIGO Cr 기준, AKI 폴더 02_aki_labels.sql).
 * - eGFR: 60(KDIGO **CKD** G3 경계) — AKI staging 기준 아님(참고용, 호박색으로 구분).
 * - 소변량: 0.5 mL/kg/h(KDIGO 핍뇨 기준).
 */
export function PatientTrendGraphs({ trends, baseline }: { trends: PatientTrends; baseline?: number | null }) {
  const crRef =
    baseline && baseline > 0
      ? Math.round(Math.min(baseline * 1.5, baseline + 0.3) * 100) / 100
      : undefined;
  return (
    <div className="grid gap-3 sm:grid-cols-3">
      <div>
        <div className="mb-1 text-[11px] font-medium">Creatinine (mg/dL)</div>
        <HoursTrendChart
          data={trends.creatinine}
          color="hsl(var(--destructive))"
          unit="mg/dL"
          refLine={crRef}
          refLabel={crRef !== undefined ? `AKI ${crRef}` : undefined}
        />
      </div>
      <div>
        <div className="mb-1 text-[11px] font-medium">eGFR (mL/min/1.73m²)</div>
        <HoursTrendChart
          data={trends.egfr}
          color="hsl(var(--primary))"
          unit=""
          refLine={60}
          refLabel="CKD 60"
          refColor="hsl(var(--warning))"
        />
      </div>
      <div>
        <div className="mb-1 text-[11px] font-medium">Urine Output (mL/kg/h)</div>
        <HoursTrendChart data={trends.urineOutput} color="hsl(var(--warning))" unit="mL/kg/h" refLine={0.5} refLabel="핍뇨 0.5" />
      </div>
    </div>
  );
}
