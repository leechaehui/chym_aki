import type { Patient } from "@/types";
import type { BadgeProps } from "@/components/ui/badge";

type Tone = NonNullable<BadgeProps["tone"]>;
export type AkiRisk = "high" | "moderate" | "low";

export interface AkiAssessment {
  risk: AkiRisk;
  riskLabel: string;
  stage: string;
  /** AKI 단계 숫자 — KPI 집계용(0 = 안정 범위). */
  stageNum: 0 | 1 | 2 | 3;
  tone: Tone;
}

function labVal(patient: Patient, key: string): number | undefined {
  // labs 미적재(목록 경량 응답 등)에도 안전하게 동작 — 크래시 대신 undefined.
  return patient.labs?.find((l) => l.key === key)?.value;
}

/**
 * AKI 위험도/단계 간이 판정 — 화면 강조용 휴리스틱(실제 KDIGO 정식 채점 아님).
 * eGFR·Creatinine·K 를 기준으로 상단 요약바의 배지 색상/문구를 결정한다.
 */
export function assessAki(patient: Patient): AkiAssessment {
  const egfr = labVal(patient, "egfr") ?? 999;
  const cr = labVal(patient, "cr") ?? 0;
  const k = labVal(patient, "k") ?? 0;

  if (egfr < 15 || cr >= 4 || k >= 6) {
    return { risk: "high", riskLabel: "고위험", stage: "AKI Stage 3", stageNum: 3, tone: "danger" };
  }
  if (egfr < 30 || cr >= 2) {
    return { risk: "moderate", riskLabel: "중등도", stage: "AKI Stage 2", stageNum: 2, tone: "high" };
  }
  if (egfr < 60 || cr >= 1.5) {
    return { risk: "low", riskLabel: "경증", stage: "AKI Stage 1", stageNum: 1, tone: "warning" };
  }
  return { risk: "low", riskLabel: "안정", stage: "안정 범위", stageNum: 0, tone: "success" };
}
