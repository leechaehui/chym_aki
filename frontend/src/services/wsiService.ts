import { api, getToken } from "./http";
import type { WsiAnalysisResult, WsiSlide, WsiStain } from "@/types/wsi";
import type { CdssDescriptors } from "@/types/cdss";

/** WSI API 베이스 — WSI/PACS 전용 서버 8001(wsi_main, 별도 프로세스). vite proxy /wsi-api → 8001.
 *  [중요] 메인 백엔드 8010(/api/wsi)으로 바꾸지 말 것(레거시·HE/MT 한정·analyze 미동작). */
const WSI_BASE = import.meta.env.VITE_WSI_BASE_URL ?? "/wsi-api";

function authHeaders(): Record<string, string> {
  const t = getToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
}

/** artHyalinosis 미표시 결정(화면만) — 모델 타겟 인덱스는 그대로 두고 응답에서만 제외. */
function stripHyalinosis(result: WsiAnalysisResult): WsiAnalysisResult {
  return {
    ...result,
    metrics: result.metrics.filter((m) => m.key !== "artHyalinosis"),
    layers: result.layers.filter((l) => l.key !== "artHyalinosis"),
    attn_overlays: result.attn_overlays.map((o) => {
      if (!("artHyalinosis" in o.contrib)) return o;
      const { artHyalinosis: _drop, ...contrib } = o.contrib;
      return { ...o, contrib };
    }),
  };
}

async function wget<T>(path: string): Promise<T> {
  const res = await fetch(`${WSI_BASE}${path}`, { headers: authHeaders() });
  if (!res.ok) throw new Error(`WSI API ${res.status}: ${path}`);
  return res.json() as Promise<T>;
}

async function wpost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${WSI_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`WSI API ${res.status}: ${path}`);
  return res.json() as Promise<T>;
}

/** prepare/cache-status — 8001 우선, 미구현(404/405)이면 메인 백엔드 8010(api) 폴백. */
let wsiCacheReady = false;
async function cacheRoute<T>(method: "GET" | "POST", path: string, body?: unknown): Promise<T> {
  if (wsiCacheReady) return body !== undefined ? wpost<T>(path, body) : wget<T>(path);
  const res = await fetch(`${WSI_BASE}${path}`, {
    method,
    headers: { ...authHeaders(), ...(body !== undefined ? { "Content-Type": "application/json" } : {}) },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (res.status === 404 || res.status === 405) {
    return body !== undefined ? api.post<T>(path, body) : api.get<T>(path);
  }
  if (!res.ok) throw new Error(`WSI API ${res.status}: ${path}`);
  wsiCacheReady = true;
  return res.json() as Promise<T>;
}

export type CacheStatus =
  | { status: "ready" }
  | { status: "downloading"; downloaded_mb: number; step?: string }
  | { status: "not_started" }
  | { status: "error"; message: string };

/** 피처(.pt) 없는 슬라이드의 온디맨드 패치추출+인코딩 진행 상태(8001 전용, 8010 폴백 없음). */
export type ExtractStatus =
  | { status: "not_started" }
  | { status: "downloading"; progress?: number }
  | { status: "extracting"; progress?: number }
  | { status: "encoding"; progress: number; total: number }
  | { status: "ready" }
  | { status: "error"; message: string };

export const wsiService = {
  /** 슬라이드 목록 조회 (HE/MT/PAS) — 8001 WSI 서버. */
  listSlides: (stain: WsiStain) =>
    wget<{ stain: WsiStain; slides: WsiSlide[]; total: number }>(`/slides?stain=${stain}`),

  /** PACS에서 슬라이드 백그라운드 다운로드 시작. (8001 우선, 미구현 시 8010 폴백) */
  prepare: (slide_id: string) =>
    cacheRoute<CacheStatus>("POST", `/wsi/prepare/${slide_id}`, {}),

  /** 슬라이드 캐시 상태 조회 (polling용). (8001 우선, 미구현 시 8010 폴백) */
  cacheStatus: (slide_id: string) =>
    cacheRoute<CacheStatus>("GET", `/wsi/cache-status/${slide_id}`),

  /** 추론 실행 (캐시 있으면 바로 반환) — 8001 Task-Attention MIL. */
  analyze: (slide_id: string, stain: WsiStain, use_cache = true) =>
    wpost<WsiAnalysisResult>("/analyze", { slide_id, stain, use_cache }).then(stripHyalinosis),

  /** 이미 분석된 캐시 결과 조회. */
  getResult: (stain: WsiStain, slide_id: string) =>
    wget<WsiAnalysisResult>(`/result/${stain}/${slide_id}`).then(stripHyalinosis),

  /** 피처(.pt) 없는 슬라이드용 — 패치추출+인코딩 시작(백그라운드, 멱등). 8001 전용. */
  extractFeatures: (slide_id: string, stain: WsiStain, case_code: string) =>
    wpost<ExtractStatus>(
      `/wsi/extract-features/${slide_id}?stain=${stain}&case_code=${encodeURIComponent(case_code)}`,
      {},
    ),

  /** 패치추출 진행률 폴링. */
  extractStatus: (slide_id: string, stain: WsiStain) =>
    wget<ExtractStatus>(`/wsi/extract-status/${slide_id}?stain=${stain}`),

  /** cdss_v5 OOF 디스크립터(ATI/immune/chronic) — 메인 백엔드 8010(/api/wsi). v6.1 §6 용. */
  cdssDescriptors: (caseCode: string) =>
    api.get<CdssDescriptors>(`/wsi/cdss-descriptors/${encodeURIComponent(caseCode)}`),

  /** 썸네일 이미지 URL (img src에 직접 사용) — PACS 게이트웨이 경유(8001 BFF), PACS 서버에서 직접 조회. */
  thumbnailUrl: (slide_id: string, size = 400) =>
    `${WSI_BASE}/pacs/cases/${slide_id}/thumbnail?size=${size}`,

  /** 특정 패치 크롭 이미지 URL — 8001. */
  patchUrl: (stain: WsiStain, slide_id: string, x: number, y: number, width: number, height: number) =>
    `${WSI_BASE}/patch/${stain}/${slide_id}?x=${x}&y=${y}&width=${width}&height=${height}`,


  // ── PACS (실 PACS 게이트웨이, 병리과 전용) — 모두 8001 경유(BFF) ──
  pacsStatus: () => wget<{ enabled: boolean }>("/pacs/status"),
  listPacsCases: () => wget<PacsCase[]>("/pacs/cases"),
  listPacsProjects: () => wget<PacsProject[]>("/pacs/projects"),
  pacsDziUrl: (caseId: string) => `${WSI_BASE}/pacs/cases/${caseId}/dzi`,
  pacsUploadUrl: () => `${WSI_BASE}/pacs/upload`,
  pacsDownloadUrl: (caseId: string) => `${WSI_BASE}/pacs/cases/${caseId}/download`,
  pacsConvertUploadUrl: () => `${WSI_BASE}/pacs/convert-upload`,
  pacsJob: (jobId: string) => wget<PacsJob>(`/pacs/jobs/${jobId}`),
  cancelPacsJob: (jobId: string) => wpost<PacsJob>(`/pacs/jobs/${jobId}/cancel`, {}),
};

export interface PacsJob {
  job_id: string;
  status: "queued" | "converting" | "uploading" | "done" | "error" | "cancelled";
  stage: string;
  progress: number;
  case_code: string;
  result: unknown;
  error: string | null;
  created_at: string;
}

export interface PacsProject {
  id: number;
  code: string;
  name: string;
  team_id: string;
  description: string | null;
}

export interface PacsCase {
  case_id: string;
  case_code: string;
  project_code: string | null;
  study_uid: string;
  description: string | null;
  created_at: string;
}
