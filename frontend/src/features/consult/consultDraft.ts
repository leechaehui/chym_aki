import type { Patient } from "@/types";
import type { ConsultUrgency } from "@/types";

interface ConsultDraft {
  urgency: ConsultUrgency;
  reason: string;
}

/** 환자 검사 결과를 기반으로 협진 긴급도·사유 초안을 자동 생성한다. */
export function draftConsultFromLabs(patient: Patient): ConsultDraft {
  const lab = (key: string) => patient.labs.find((l) => l.key === key)?.value ?? null;

  const cr = lab("cr");
  const egfr = lab("egfr");
  const k = lab("k");
  const upcr = lab("upcr");

  // 긴급도 판정 — 중증 기준: Cr > 5, eGFR < 15, K > 6.0
  let urgency: ConsultUrgency = "routine";
  if ((cr !== null && cr > 5) || (egfr !== null && egfr < 15) || (k !== null && k > 6.0)) {
    urgency = "emergency";
  } else if ((cr !== null && cr > 2) || (egfr !== null && egfr < 30) || (k !== null && k > 5.5)) {
    urgency = "urgent";
  }

  // 사유 초안 구성
  const parts: string[] = [];
  if (cr !== null) parts.push(`크레아티닌 ${cr} mg/dL`);
  if (egfr !== null) parts.push(`eGFR ${egfr} mL/min`);
  if (k !== null) parts.push(`칼륨 ${k} mEq/L`);
  if (upcr !== null) parts.push(`UPCR ${upcr}`);

  const labSummary = parts.length > 0 ? `(${parts.join(", ")})` : "";
  const reason = `${patient.diagnosis} 환자로, 신장 기능 평가 및 조직학적 감별 진단을 위한 병리 협진을 요청합니다. ${labSummary}`.trim();

  return { urgency, reason };
}
