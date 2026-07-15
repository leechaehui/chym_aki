import { useEffect, useState } from "react";
import type { Patient, ShapFeature } from "@/types";
import { riskBandFor } from "@/lib/riskLabel";
import { cn } from "@/lib/cn";
import { icuMonitorService } from "@/services/icuMonitorService";

const SHAP_RULES: Record<string, { label: string; unit: string; meaning: string; range?: { low?: number; high?: number } }> = {
  urine_output_6h: {
    label: "최근 6시간 소변량",
    unit: "mL/kg/h",
    meaning: "소변량 감소는 AKI 진행 및 저관류를 시사합니다.",
    range: { low: 0.5 },
  },
  creatinine_max: {
    label: "Creatinine 최대값(48h)",
    unit: "mg/dL",
    meaning: "Cr 최고치 상승은 신기능 악화와 AKI 위험 증가를 반영합니다.",
    range: { high: 1.5 },
  },
  bun_max: {
    label: "BUN 최대값",
    unit: "mg/dL",
    meaning: "BUN 상승은 신기능 저하 및 탈수·혈류 저하를 시사합니다.",
    range: { high: 20 },
  },
  sbp_mean: {
    label: "평균 수축기혈압",
    unit: "mmHg",
    meaning: "저혈압 지속은 장기 관류 저하와 AKI 위험과 연관됩니다.",
    range: { low: 90, high: 140 },
  },
  spo2_mean: {
    label: "평균 SpO₂",
    unit: "%",
    meaning: "저산소 상태는 저관류·염증 부담과 연관될 수 있습니다.",
    range: { low: 95 },
  },
};

function formattedValue(value: number | string | null | undefined, unit: string) {
  if (value == null || value === "") return "—";
  const numberValue = typeof value === "number" ? value : Number(value);
  if (Number.isNaN(numberValue)) return "—";
  return `${numberValue.toFixed(2)}${unit ? ` ${unit}` : ""}`;
}

function verdictFor(feature: ShapFeature): { text: string; cls: string } {
  const rule = SHAP_RULES[feature.feature];
  const value = typeof feature.value === "number" ? feature.value : Number(feature.value ?? feature.scaledValue ?? 0);
  if (!rule || Number.isNaN(value)) {
    return { text: "—", cls: "text-muted-foreground" };
  }

  if (rule.range?.low != null && value < rule.range.low) {
    return { text: "↓ 낮음", cls: "text-warning font-semibold" };
  }
  if (rule.range?.high != null && value > rule.range.high) {
    return { text: "↑ 높음", cls: "text-destructive font-semibold" };
  }
  return { text: "정상", cls: "text-foreground" };
}

function rangeText(feature: ShapFeature): string {
  const rule = SHAP_RULES[feature.feature];
  if (!rule?.range) return "—";
  const low = rule.range.low != null ? `≥ ${rule.range.low}` : null;
  const high = rule.range.high != null ? `≤ ${rule.range.high}` : null;
  if (low && high) return `${low} / ${high}`;
  return low ?? high ?? "—";
}

function clinicalMeaning(feature: ShapFeature): string {
  return SHAP_RULES[feature.feature]?.meaning ?? "모델이 이 변수를 위험 예측에 반영했습니다.";
}

export function LabResultReport({
  patient,
  predictedStage,
  predictionScore,
}: {
  patient: Patient;
  predictedStage?: string | null;
  predictionScore?: number | null;
}) {
  const [shapFeatures, setShapFeatures] = useState<readonly ShapFeature[]>([]);
  // patient.diagnosis는 자유 텍스트라 등급 판정에 쓰면 문구가 바뀔 때마다 깨진다 —
  // patient.predictedStage(모델의 실제 예측, 목록 응답에 이미 포함)를 폴백으로 쓴다.
  const band = riskBandFor(predictionScore ?? patient.aiRiskScore, predictedStage ?? patient.predictedStage ?? null);

  const trend = patient.trend;
  const baseline = trend.length ? trend[0].creatinine : undefined;
  const crNow = patient.labs.find((lab) => lab.key === "cr")?.value ?? (trend.length ? trend[trend.length - 1].creatinine : undefined);
  const ratio = baseline && crNow && baseline > 0 ? crNow / baseline : undefined;
  const delta = baseline != null && crNow != null ? crNow - baseline : undefined;
  const crStage =
    ratio == null
      ? null
      : (crNow != null && crNow >= 4.0) || ratio >= 3.0
        ? 3
        : ratio >= 2.0
          ? 2
          : ratio >= 1.5 || (delta != null && delta >= 0.3)
            ? 1
            : 0;

  const uo = patient.urineOutput.length ? patient.urineOutput[patient.urineOutput.length - 1].value : undefined;
  const uoStatus = uo == null ? null : uo < 0.3 ? "무뇨(<0.3)" : uo < 0.5 ? "핍뇨(<0.5)" : "정상(≥0.5)";

  useEffect(() => {
    const idSrc = patient as { stayId?: number; id?: string; mrn?: string };
    const cleanId = String(idSrc.stayId ?? "").replace(/[^\d]/g, "") || String(idSrc.id ?? "").replace(/[^\d]/g, "") || String(idSrc.mrn ?? "").replace(/[^\d]/g, "");
    const stayId = Number(cleanId);
    if (!Number.isFinite(stayId) || stayId === 0) {
      setShapFeatures([]);
      return;
    }
    icuMonitorService.shap(stayId).then((res) => setShapFeatures(res.features)).catch(() => setShapFeatures([]));
  }, [patient.id]);

  const topShapFeatures = [...shapFeatures]
    .map((feature) => {
      const shapValue = typeof feature.shap_value === "number" ? feature.shap_value : feature.contribution;
      return { feature, shapValue: Number(shapValue ?? 0) };
    })
    .sort((a, b) => Math.abs(b.shapValue) - Math.abs(a.shapValue))
    .slice(0, 5);

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-border p-3">
        <p className="mb-2 text-sm font-semibold text-foreground">SHAP Top5 변수 카드</p>
        {topShapFeatures.length === 0 ? (
          <p className="text-[11px] text-muted-foreground">SHAP 주요 변수를 불러오지 못했습니다.</p>
        ) : (
          <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
            {topShapFeatures.map(({ feature }) => {
              const valueText = formattedValue(feature.value ?? feature.scaledValue, feature.unit ?? SHAP_RULES[feature.feature]?.unit ?? "");
              const shapValue = Number(typeof feature.shap_value === "number" ? feature.shap_value : feature.contribution ?? 0);
              return (
                <div key={feature.feature} className="rounded-md border border-border bg-muted/30 p-3">
                  <p className="text-[11px] font-semibold text-foreground">{SHAP_RULES[feature.feature]?.label ?? feature.label}</p>
                  <p className="mt-2 text-base font-semibold text-foreground">{valueText}</p>
                  <div className="mt-1 flex items-center justify-between text-[11px] text-muted-foreground">
                    <span>SHAP {shapValue.toFixed(3)}</span>
                    <span>{verdictFor(feature).text}</span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div className="rounded-lg border border-border p-3">
        <p className="mb-2 text-sm font-semibold text-foreground">SHAP Top5 상세 리포트</p>
        {topShapFeatures.length === 0 ? (
          <p className="text-[11px] text-muted-foreground">SHAP 주요 변수를 불러오지 못했습니다.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[620px] text-xs">
              <thead>
                <tr className="border-b border-border text-left text-muted-foreground">
                  <th className="px-2 py-2 font-medium">변수명</th>
                  <th className="px-2 py-2 font-medium">실제값</th>
                  <th className="px-2 py-2 font-medium">정상범위</th>
                  <th className="px-2 py-2 font-medium">판정</th>
                  <th className="px-2 py-2 font-medium">임상적 의미</th>
                </tr>
              </thead>
              <tbody>
                {topShapFeatures.map(({ feature }) => {
                  const valueText = formattedValue(feature.value ?? feature.scaledValue, feature.unit ?? SHAP_RULES[feature.feature]?.unit ?? "");
                  const verdict = verdictFor(feature);
                  return (
                    <tr key={feature.feature} className="border-b border-border/40 align-top">
                      <td className="px-2 py-2 font-medium text-foreground">{SHAP_RULES[feature.feature]?.label ?? feature.label}</td>
                      <td className="px-2 py-2 tabular-nums text-foreground">{valueText}</td>
                      <td className="px-2 py-2 text-muted-foreground">{rangeText(feature)}</td>
                      <td className={cn("px-2 py-2 font-semibold", verdict.cls)}>{verdict.text}</td>
                      <td className="px-2 py-2 text-foreground/80">{clinicalMeaning(feature)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        <div className="rounded-lg border border-border bg-muted/30 p-3 text-xs leading-relaxed text-foreground">
          <p className="mb-1 font-semibold text-foreground">KDIGO AKI 기준 해석</p>
          <ul className="flex flex-col gap-0.5">
            <li>
              · Cr 기준: {ratio != null ? `현재 ${crNow} / 기저 ${baseline} = ${ratio.toFixed(2)}배` : "기저 또는 현재값 부족"}
              {delta != null ? ` (증가 ${delta >= 0 ? "+" : ""}${delta.toFixed(2)} mg/dL)` : ""}
              {crStage != null && (
                <b className={cn("ml-1", crStage >= 2 ? "text-destructive" : "")}>
                  → {crStage === 0 ? "Cr 기준 비충족" : `Stage ${crStage} 기준 충족`}
                </b>
              )}
            </li>
            <li>
              · 소변량 기준: {uoStatus != null ? (
                <b className={cn(uo != null && uo < 0.3 ? "text-destructive" : "")}>{uoStatus} mL/kg/h</b>
              ) : "소변량 데이터 없음"}
            </li>
            <li className="text-muted-foreground">
              · KDIGO Stage: Cr 1.5/2.0/3.0배 또는 +0.3 mg/dL(48h) · 소변량 &lt;0.5 mL/kg/h(≥6h). 두 기준 중 높은 단계가 최종.
            </li>
          </ul>
        </div>

        <div className="rounded-lg border border-border p-3 text-xs leading-relaxed">
          <p className="mb-1 font-semibold text-foreground">종합 평가 (AI 모델)</p>
          <p className="text-foreground">
            AI 위험도 <b className={band.textClass}>{band.label} ({predictionScore ?? patient.aiRiskScore})</b>
            {" · "}모델 예측 <b>{patient.diagnosis}</b>
          </p>
          <p className="mt-1 text-foreground/80">
            {band.tier === "high"
              ? "신장내과 평가/협진 권장 — 신독성 약물 점검, 전해질·산염기 교정, 시간당 소변량·추적 Cr 모니터링."
              : band.tier === "moderate"
                ? "AKI 위험 — 수액 상태 점검, 신독성 약물 회피, 추적 신기능 검사 권장."
                : "현재 위험 낮음 — 위험인자 동반 시 추적 관찰."}
          </p>
          <p className="mt-1 text-[11px] text-muted-foreground">
            * 위험도·예측은 48h 학습 모델 출력입니다. 위 'KDIGO 기준 해석'은 현재 검사값 기준이라 모델과 다를 수 있습니다(조기경보). 최종 판단은 의료진이 수행합니다.
          </p>
        </div>
      </div>
    </div>
  );
}
