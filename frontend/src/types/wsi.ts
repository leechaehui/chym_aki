/** WSI API (port 8001) 전용 타입 — HE/MT ABMIL 모델 결과. */

export type WsiStain = "HE" | "MT" | "PAS";

export interface WsiSlide {
  slide_id: string;
  case_code: string;
  stain: WsiStain;
  has_features: boolean;
  cached: boolean;
  description?: string;
  /** PACS 출처 여부 — true 면 pacsDziUrl(/pacs/cases/{id}/dzi) 사용(로컬 /dzi 는 404). PACS-only 라 통상 true. */
  is_pacs: boolean;
}

export interface WsiAttnOverlay {
  cx: number;       // 슬라이드 너비 기준 정규화 중심 x (0~1)
  cy: number;       // 슬라이드 너비 기준 정규화 중심 y (0~1)
  r: number;        // 슬라이드 너비 기준 정규화 반지름
  weight: number;   // attention 가중치 (0~1, 정규화)
  contrib: Record<string, number>; // target별 패치 기여도 (정규화)
  px?: number;      // 원본 슬라이드 level 0 기준 좌상단 x 픽셀 좌표
  py?: number;      // 원본 슬라이드 level 0 기준 좌상단 y 픽셀 좌표
  psize?: number;   // 원본 슬라이드 level 0 기준 패치 픽셀 크기
}

export interface WsiTissue {
  id: number;
  x: number;
  y: number;
  w: number;
  h: number;
  area: number;
}

export interface WsiMetric {
  key: string;
  label: string;
  value: number;
  unit: string;
  raw: number;
  /** descriptor별 보정 신뢰도(0~1) — CDSS(PAS)만. ABMIL(HE/MT)은 null. */
  confidence?: number | null;
}

export interface WsiLayer {
  key: string;
  label: string;
  color: string;
  count: number | null;
  visible: boolean;
}

export interface WsiReport {
  findings: string;
  diagnosis: string;
  status: string;
  updatedAt: string | null;
  /** 슬라이드 단위 불확실성(0~1, 낮을수록 확신) — CDSS(PAS)만. */
  uncertainty?: number | null;
}

export interface WsiAnalysisResult {
  stain: WsiStain;
  slide_id: string;
  model_label: string;
  metrics: WsiMetric[];
  layers: WsiLayer[];
  report: WsiReport;
  heatmap: Array<{ patch_idx: number; weight: number }>;
  attn_overlays: WsiAttnOverlay[];
  n_patches: number;
  tissues?: WsiTissue[];
  slide_w?: number;
  slide_h?: number;
}

