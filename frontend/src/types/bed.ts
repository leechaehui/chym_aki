/** 병상/응급환자 도메인 타입 (응급의학과). */

/** 병상 구역. */
export type BedZone = "er" | "icu" | "ward" | "isolation";

export const BED_ZONE_LABEL: Record<BedZone, string> = {
  er: "응급실",
  icu: "ICU",
  ward: "일반병동",
  isolation: "격리병동",
};

/** 개별 병상 상태. */
export type BedState = "available" | "occupied" | "cleaning";

export const BED_STATE_LABEL: Record<BedState, string> = {
  available: "사용 가능",
  occupied: "사용중",
  cleaning: "정리중",
};

export interface Bed {
  readonly id: string;
  readonly zone: BedZone;
  readonly label: string; // 예: "ER-01"
  readonly state: BedState;
  readonly patientName: string | null;
}

/** 입실 환자의 현재 처방 약물 한 건. */
export interface Medication {
  readonly name: string;
  readonly dose: string;
  readonly route: string; // IV / PO / SC 등
  readonly status: string; // 예: 투여중, 1회 투여
}

/** 입실 환자의 혈액/검사 결과 한 항목. */
export interface BedLab {
  readonly label: string;
  readonly value: string;
  readonly unit: string;
  readonly flag: "normal" | "high" | "low";
}

/** 직전 입력 정보(활력징후·I/O 등) 한 항목 — AKI 상세 좌측 패널용. */
export interface VitalEntry {
  readonly label: string;
  readonly value: string;
}

/**
 * 병상에 입실한 환자의 상세(현재 처방·검사·AKI 평가).
 * 병상 클릭 시 BedDetailModal 에 표시되며, akiRisk=true 면 보드에서 빨강으로 강조하고
 * 좌(직전 입력)/우(처치 리포트 + 신장내과 협진) 2단 패널을 노출한다.
 */
export interface BedPatientDetail {
  readonly diagnosis: string;
  readonly attending: string;
  readonly admittedAt: string;
  readonly medications: readonly Medication[];
  readonly labs: readonly BedLab[];
  /** AKI 위험 여부 — 보드 빨강 표시 + 처치 리포트/협진 패널 트리거. */
  readonly akiRisk: boolean;
  readonly akiStage?: string;
  /** 직전 입력 정보(좌측 패널). */
  readonly recentInputs?: readonly VitalEntry[];
  /** 처치 권고 요약 리포트(우측 패널). */
  readonly treatmentReport?: string;
  /** 우선 처치 항목 체크리스트. */
  readonly treatmentItems?: readonly string[];
}

