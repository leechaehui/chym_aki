/**
 * 위험 등급은 점수 대신 최종 예측 Stage 기준으로 결정한다.
 * Stage 2-3 → 고위험, 그 외 → 주의.
 */

export type RiskTier = "high" | "moderate" | "low";

export interface RiskBand {
  readonly tier: RiskTier;
  readonly label: string;
  readonly dot: string;
  readonly textClass: string;
  /** 표시용 점수 구간(예: "60–100") — 범례(RiskStratification)에서만 사용. */
  readonly range?: string;
}

// 백엔드 IcuMonitorService.summary()의 임계값(≥60 고위험 · 30~59 주의 · <30 안정)과 동일 3단계.
// 예전엔 "안정" 구간이 빠져있어서 범례 표에 0–30(안정) 이 아예 안 보이는 버그가 있었다.
export const RISK_BANDS: readonly RiskBand[] = [
  { tier: "high", label: "고위험", dot: "🔴", textClass: "text-destructive", range: "60–100" },
  { tier: "moderate", label: "주의", dot: "🟡", textClass: "text-warning", range: "30–59" },
  { tier: "low", label: "안정", dot: "🟢", textClass: "text-success", range: "0–29" },
];

/** predictedStage 문자열 → {label,color}. 3단계(고위험/주의/안정)를 명시적으로 다 처리한다 —
 * "그 외엔 전부 주의" 식 fallback을 쓰면 Non-AKI 도 안정이 아니라 주의로 잘못 뜬다(과거 버그). */
export function getRiskLabel(predictedStage: string | null | undefined): { label: string; color: string } {
  const stage = predictedStage?.toLowerCase() ?? "";
  if (stage.includes("stage 2-3") || stage.includes("2-3") || stage.includes("stage2") || stage.includes("stage 3")) {
    return { label: "고위험", color: "#EF4444" };
  }
  if (stage.includes("stage 1") || stage.includes("stage1")) {
    return { label: "주의", color: "#F59E0B" };
  }
  return { label: "안정", color: "#10B981" };
}

export function riskBandFor(score: number, predictedStage?: string | null): RiskBand {
  if (predictedStage != null) {
    const label = getRiskLabel(predictedStage);
    if (label.label === "고위험") return { tier: "high", label: label.label, dot: "🔴", textClass: "text-destructive" };
    if (label.label === "주의") return { tier: "moderate", label: label.label, dot: "🟡", textClass: "text-warning" };
    return { tier: "low", label: label.label, dot: "🟢", textClass: "text-success" };
  }
  // predictedStage(모델의 실제 예측 stage 문자열)가 없으면 AI 위험점수로 등급 판정.
  // 백엔드 IcuMonitorService.summary()의 임계값(≥60 고위험 · 30~59 주의 · <30 안정)과 동일 기준.
  if (score >= 60) return { tier: "high", label: "고위험", dot: "🔴", textClass: "text-destructive" };
  if (score >= 30) return { tier: "moderate", label: "주의", dot: "🟡", textClass: "text-warning" };
  return { tier: "low", label: "안정", dot: "🟢", textClass: "text-success" };
}

export const RISK_BADGE_TONE: Record<RiskTier, "danger" | "warning" | "success" | "neutral"> = {
  high: "danger",
  moderate: "warning",
  low: "neutral",
};
