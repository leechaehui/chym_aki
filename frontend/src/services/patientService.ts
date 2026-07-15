import type { Patient } from "@/types";
import { api } from "./http";

/**
 * 환자 서비스 (Facade). 환자 목록·상세 조회 — FastAPI 연동.
 * AI 진료 초안(SOAP) 생성은 voiceService(/voice) 로 분리되었다.
 */
class PatientService {
  /**
   * limit 기본값을 넉넉하게(1000, 데모 최대 코호트 크기) 잡는다 — 서버 기본값(20)에
   * admitted_at DESC 정렬이 겹치면, 데모처럼 저위험군이 더 최근 admitted_at을 갖는
   * 코호트에서 진짜 고위험 환자들이 페이지 밖으로 잘려나가 프론트가 재정렬해도
   * 애초에 못 보는 문제가 있었다. 서버 MAX_PAGE_SIZE(core/query_optimizer.py)도
   * 이 값 이상으로 맞춰둬야 한다.
   */
  async list(limit = 1000): Promise<Patient[]> {
    return api.get<Patient[]>(`/patients?limit=${limit}`);
  }

  async get(id: string): Promise<Patient | undefined> {
    try {
      return await api.get<Patient>(`/patients/${id}`);
    } catch {
      // 404 등은 undefined 로 (기존 mock 시그니처 호환).
      return undefined;
    }
  }
}

export const patientService = new PatientService();
