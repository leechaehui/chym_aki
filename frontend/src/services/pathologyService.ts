import type { PathologyResult } from "@/types";
import { api } from "./http";

/** 병리 분석/보고서 서비스 (Facade) — FastAPI 연동. */
class PathologyService {
  async list(): Promise<PathologyResult[]> {
    return api.get<PathologyResult[]>("/pathology");
  }

  async getByConsult(consultId: string): Promise<PathologyResult | undefined> {
    try {
      return await api.get<PathologyResult>(`/pathology/by-consult/${consultId}`);
    } catch {
      // 해당 협진의 병리 결과가 아직 없으면 undefined(프론트 기존 계약 유지).
      return undefined;
    }
  }

  /** "임시 저장"/"판독 완료" — 소견/진단을 실제로 저장. */
  async saveReport(
    consultId: string,
    report: { findings: string; diagnosis: string; status: "draft" | "final" },
  ): Promise<PathologyResult> {
    return api.put<PathologyResult>(`/pathology/by-consult/${consultId}/report`, report);
  }

  /** 환자(patientMrn) ↔ WSI 슬라이드 매핑 조회 — 매핑 없으면 빈 배열(프론트가 전체 목록으로 폴백). */
  async getWsiMapping(patientMrn: string): Promise<{ slide_id: string; stain: string }[]> {
    try {
      return await api.get(`/pathology/wsi-mapping/${encodeURIComponent(patientMrn)}`);
    } catch {
      return [];
    }
  }
}

export const pathologyService = new PathologyService();
