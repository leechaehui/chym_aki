import type { BadgeProps } from "@/components/ui/badge";
import type { ApprovalStatus, ConsultStatus, ConsultUrgency, BedState } from "@/types";

type Tone = NonNullable<BadgeProps["tone"]>;

/** 도메인 상태값 → 배지 색상(tone) 매핑. 화면 전반에서 일관된 색상 규칙을 보장한다. */

export const consultStatusTone: Record<ConsultStatus, Tone> = {
  requested: "danger",    // 대기중 — 아직 아무도 안 봄, 가장 시급
  in_progress: "warning", // 진행중
  read: "primary",
  replied: "neutral",     // 회신완료 — 처리 끝, 더 이상 주의 끌 필요 없음
};

export const urgencyTone: Record<ConsultUrgency, Tone> = {
  routine: "neutral",
  urgent: "high",
  emergency: "danger",
};

export const approvalTone: Record<ApprovalStatus, Tone> = {
  pending: "warning",
  approved: "success",
  rejected: "danger",
};

export const bedStateTone: Record<BedState, Tone> = {
  available: "success",
  occupied: "warning", // 사용중 = 노랑
  cleaning: "warning",
};
