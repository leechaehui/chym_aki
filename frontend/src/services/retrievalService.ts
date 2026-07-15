import { api } from "./http";
import type { RetrievalQuery, RetrievalResult } from "@/types/retrieval";

/** Retrieval CDSS 서비스(Facade) — /api/retrieval (메인 백엔드 8010). */
export const retrievalService = {
  /** 임상 concept → Top-K 프로토타입 + 대표 WSI (Evidence). */
  query: (body: RetrievalQuery) => api.post<RetrievalResult>("/retrieval/query", body),

  /** 아틀라스 전체(대시보드/디버깅). */
  listPrototypes: () =>
    api.get<Array<{ prototypeId: string; label: string; nMembers: number; isRare: boolean;
                    kdigoBand: string | null; etiologyHint: string | null; egfrMean: number | null }>>(
      "/retrieval/prototypes"),
};
