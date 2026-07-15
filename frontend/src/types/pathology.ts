/** 병리 분석/보고서 도메인 타입 (병리과). */

/** 지원 염색. */
export type Stain = "HE" | "MT" | "TRI" | "PAS" | "Silver" | "Masson";

export const STAIN_LABEL: Record<Stain, string> = {
  HE:     "H&E",
  MT:     "Masson Trichrome (MT)",
  TRI:    "Masson Trichrome (MT)",
  PAS:    "PAS",
  Silver: "Silver Stain",
  Masson: "Masson Trichrome",
};

/** WSI 위에 토글 가능한 검출 구조 레이어. */
export interface DetectedLayer {
  readonly key: string;
  readonly label: string;
  readonly color: string;
  readonly count: number | null;
  readonly visible: boolean;
}

/** 정량 분석 지표 한 항목. */
export interface QuantMetric {
  readonly key: string;
  readonly label: string;
  readonly value: number | null;
  readonly unit: string;
}

/** 병리 보고서. */
export interface PathologyReport {
  readonly findings: string;
  readonly diagnosis: string;
  readonly status: "draft" | "final";
  readonly updatedAt: string | null;
}

/** 협진 건에 연결된 병리 분석 결과 묶음. */
export interface PathologyResult {
  readonly consultId: string;
  readonly stain: Stain;
  readonly imageUrl: string | null; // Mock — null 이면 업로드/뷰어 안내
  readonly layers: readonly DetectedLayer[];
  readonly metrics: readonly QuantMetric[];
  readonly report: PathologyReport;
}
