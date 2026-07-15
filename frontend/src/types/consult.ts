/** 협진 도메인 타입 (신장내과 ↔ 병리과, 응급/ICU → 신장내과). */

/**
 * 협진 종류(요청 → 회신 부서 방향). 같은 Consult 엔티티/스토어/서비스를 두 흐름이 공유한다.
 * - pathology : 신장내과 → 병리과 (신생검 판독 협진)
 * - nephrology: 응급의학과/ICU → 신장내과 (AKI 응급 협진)
 */
export type ConsultKind = "pathology" | "nephrology";

export const CONSULT_KIND_LABEL: Record<ConsultKind, string> = {
  pathology: "병리 협진",
  nephrology: "신장 협진",
};

/**
 * 협진 상태. 두 흐름이 enum 을 공유한다(최대 재사용).
 * - 병리 협진:  requested → in_progress → read → replied
 * - 신장 협진:  requested(PENDING) → in_progress(IN_PROGRESS) → replied(COMPLETED)  (read 단계 미사용)
 */
export type ConsultStatus = "requested" | "in_progress" | "read" | "replied";

export const CONSULT_STATUS_LABEL: Record<ConsultStatus, string> = {
  requested: "대기중",
  in_progress: "진행중",
  read: "판독완료",
  replied: "회신완료",
};

export type ConsultUrgency = "routine" | "urgent" | "emergency";

export const URGENCY_LABEL: Record<ConsultUrgency, string> = {
  routine: "일반",
  urgent: "긴급",
  emergency: "응급",
};

/** 협진 타임라인 단계 한 항목. */
export interface ConsultEvent {
  readonly id: string;
  readonly stage: "requested" | "received" | "analyzing" | "read" | "replied";
  readonly label: string;
  readonly at: string;
  readonly actor: string;
}

/** 병리 회신에 실리는 stain별 AI 분석 요약(신장내과 수신 리포트의 ROI·heatmap·신뢰도용). */
export interface StainAnalysisSummary {
  readonly stain: string;                 // HE | MT | PAS
  readonly slideId: string;
  readonly status?: string | null;        // ALLOW | ABSTAIN (CDSS)
  readonly uncertainty?: number | null;   // 슬라이드 단위(CDSS)
  readonly confidences?: { label: string; value: number }[]; // descriptor별 신뢰도(0~1, PAS)
  readonly overlays?: { cx: number; cy: number; r: number; weight: number; px?: number; py?: number; psize?: number }[];
}

/** 병리과 회신. */
export interface ConsultReply {
  readonly findings: string;
  readonly diagnosis: string;
  readonly recommendation: string;
  readonly author: string;
  readonly repliedAt: string;
  /** 판독의 전자 서명 이미지 경로(공개 /uploads). */
  readonly signaturePath?: string | null;
  /** stain별 AI 분석 요약(ROI·heatmap·신뢰도). */
  readonly analysis?: StainAnalysisSummary[] | null;
}

export interface Consult {
  readonly id: string;
  readonly kind: ConsultKind; // 협진 종류(요청→회신 방향). 미지정 시드는 "pathology" 로 간주.
  readonly patientMrn: string;
  readonly patientName: string;
  readonly diagnosis: string;
  readonly keyLabs: string; // 주요 검사 요약
  readonly reason: string; // 협진 요청 사유
  readonly urgency: ConsultUrgency;
  readonly status: ConsultStatus;
  readonly requestedBy: string;
  readonly requestedAt: string;
  /** 신장 협진(응급/ICU → 신장내과) 출처 병상 라벨(예: "ICU-03"). 병리 협진에는 없음. */
  readonly bedLabel?: string;
  readonly timeline: readonly ConsultEvent[];
  readonly reply: ConsultReply | null;
}
