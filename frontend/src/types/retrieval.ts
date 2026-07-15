/** Retrieval CDSS 타입 — 백엔드 /api/retrieval (camelCase) 대응. Evidence Display 전용. */

export interface RetrievalQuery {
  kdigoStage?: number | null;
  crTrendSlope?: number | null;
  oliguria?: number | null;
  egfr?: number | null;
  proteinuriaMgG?: number | null;
  a1cPct?: number | null;
  age?: number | null;
  sex?: string | null;
  diabetes?: boolean | null;
  hypertension?: boolean | null;
  etiologyHint?: string;
  subjectId?: number | null;
  stayId?: number | null;     // 지정 시 MIMIC ICU stay 에서 concept 자동 추출
  k?: number;
}

export interface RepresentativeWsi {
  slideId: string;
  stain?: string | null;
  source: string;
}

export interface RetrievalHit {
  prototypeId: string;
  label: string;
  similarity: number;
  confidence: number;
  nMembers: number;
  isRare: boolean;
  kdigoBand?: string | null;
  etiologyHint?: string | null;
  egfrMean?: number | null;
  representativeWsi?: RepresentativeWsi | null;
}

export interface ConceptSummary {
  kdigoStage?: number | null;
  kdigoBand: string;
  egfr?: number | null;
  proteinuria?: number | null;
  a1c?: number | null;
  age?: number | null;
  etiologyHint: string;
  completeness: number;
}

export interface Ood {
  score?: number | null;
  isOod: boolean;
  message?: string | null;
}

export interface RetrievalResult {
  concept: ConceptSummary;
  ood: Ood;
  hits: RetrievalHit[];
  referenceNotice: string;
}
