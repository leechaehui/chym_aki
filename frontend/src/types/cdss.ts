/**
 * CHYM v6.1 CDSS 3-layer 판정 타입 (실데이터 전용).
 *
 * 모든 값은 실제 모델 출력에서만 파생된다:
 *  - L2/L3 = 실제 ABMIL 헤드(WsiAnalysisResult.metrics, 8001)
 *  - L1    = 실제 EMR AKI risk(Patient.aiRiskScore, 8010) — 없으면 WSI 만성/급성 파생
 * stub·합성난수·가짜 점수는 사용하지 않는다(§ "실데이터만").
 */

export type V61Level = "HIGH" | "MODERATE" | "LOW";
export type V61Confidence = "CONFIDENT" | "SHADOW" | "ABSTAIN";
/** L1 risk 출처 — EMR 실모델 우선, 미연동 시 WSI 파생(라벨로 투명 표기). */
export type V61RiskSource = "EMR" | "WSI_DERIVED";

/** L2 축 / L3 descriptor 단일 항목. real=실제 학습 헤드 직접값 여부. */
export interface V61Axis {
  readonly key: string;
  readonly label: string;
  readonly level: V61Level;
  readonly score: number; // 0..1 정규화
  readonly real: boolean; // true=실헤드 직접, false=실헤드 합성/미측정
  readonly note?: string; // 합성/미측정 사유(투명성)
}

/** cdss_v5 OOF 디스크립터(8010 /api/wsi/cdss-descriptors) — 실제 학습 헤드. */
export interface CdssDescriptorMetric {
  readonly auroc: number | null;
  readonly ci95: readonly [number, number] | null;
  readonly n: number | null;
}
export interface CdssDescriptors {
  readonly caseCode: string;
  readonly available: boolean;
  /** atiSeverity(ATI)·immune(tubulitis)·chronic·stage3 (0..1). 코호트 미포함 시 null. */
  readonly descriptors: {
    readonly atiSeverity?: number;
    readonly immune?: number;
    readonly chronic?: number;
    readonly stage3?: number;
  } | null;
  readonly metrics?: Record<string, CdssDescriptorMetric>;
  readonly source?: string;
}

export interface CdssV61Decision {
  // L1 — 임상 결론
  readonly riskLevel: V61Level;
  readonly riskSource: V61RiskSource;
  readonly confidence: V61Confidence;
  readonly confidenceReason: string;
  readonly interpretation: string;
  // L2 — 핵심 3축
  readonly axes: readonly V61Axis[];
  // L3 — 상세 descriptor(실헤드)
  readonly descriptors: readonly V61Axis[];
  // MFDS model trace
  readonly modelLabel: string;
  readonly nPatches: number;
}
