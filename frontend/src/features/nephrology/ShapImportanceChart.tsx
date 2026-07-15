import { Bar, BarChart, CartesianGrid, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ShapFeature } from "@/types";

type NormalizedShapFeature = {
  feature: string;
  label: string;
  shapValue: number;
  displayValue: number;
};

function normalizeFeatures(features: readonly ShapFeature[]): NormalizedShapFeature[] {
  return features
    .map((feature) => {
      const value = typeof (feature as { shap_value?: number }).shap_value === "number"
        ? (feature as { shap_value?: number }).shap_value
        : typeof (feature as { contribution?: number }).contribution === "number"
          ? (feature as { contribution?: number }).contribution
          : typeof (feature as { scaled_value?: number }).scaled_value === "number"
            ? (feature as { scaled_value?: number }).scaled_value
            : null;

      if (value == null) return null;
      return {
        feature: feature.feature,
        label: feature.label ?? feature.feature,
        shapValue: value,
        displayValue: Math.abs(value),
      };
    })
    .filter((entry): entry is NormalizedShapFeature => entry != null)
    .sort((a, b) => b.displayValue - a.displayValue)
    .slice(0, 5);
}

export function ShapImportanceChart({
  features,
  height = 300,
}: {
  features: readonly ShapFeature[];
  height?: number;
}) {
  const data = normalizeFeatures(features);

  if (data.length === 0) {
    return <p className="py-3 text-center text-[11px] text-muted-foreground">SHAP 기여도 데이터를 불러올 수 없습니다.</p>;
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 min-h-0">
        <ResponsiveContainer width="100%" height={height}>
        <BarChart
          data={data.map((item) => ({ ...item, name: item.label }))}
          layout="vertical"
          margin={{ top: 6, right: 52, left: 8, bottom: 6 }}
          barCategoryGap={10}
        >
          <CartesianGrid strokeDasharray="2 2" stroke="hsl(var(--border))" horizontal={false} />
          <XAxis
            type="number"
            axisLine={false}
            tickLine={false}
            tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
            // 가장 긴 막대가 오른쪽 끝에 닿아 값 라벨이 잘리지 않도록 15% 여백을 둔다.
            domain={[0, (dataMax: number) => (dataMax > 0 ? dataMax * 1.15 : 1)]}
          />
          <YAxis
            type="category"
            dataKey="name"
            width={122}
            axisLine={false}
            tickLine={false}
            tick={{ fontSize: 10, fill: "hsl(var(--foreground))" }}
          />
          <Tooltip
            cursor={{ fill: "hsl(var(--muted)/0.35)" }}
            contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8, fontSize: 12 }}
            formatter={(value) => [`${Number(value ?? 0).toFixed(3)}`, "SHAP Value"]}
          />
          <Bar dataKey="displayValue" fill="#dc2626" radius={[0, 4, 4, 0]}>
            <LabelList
              dataKey="shapValue"
              position="right"
              offset={8}
              style={{ fontSize: 11, fontWeight: 600, fill: "hsl(var(--foreground))" }}
              formatter={(value) => `${Number(value ?? 0).toFixed(3)}`}
            />
          </Bar>
        </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="shrink-0 pt-1 text-[10px] text-muted-foreground">API로 전달된 SHAP 값 기준 Top 5 · 절댓값 내림차순</p>
    </div>
  );
}
