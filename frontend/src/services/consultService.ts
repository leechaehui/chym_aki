import type { Consult, ConsultKind, ConsultReply, ConsultUrgency } from "@/types";
import { api } from "./http";

/** 협진 요청 입력. kind 미지정 시 병리 협진(신장내과 → 병리과)으로 간주. */
export interface ConsultRequestInput {
  kind?: ConsultKind;
  patientMrn: string;
  patientName: string;
  diagnosis: string;
  keyLabs: string;
  reason: string;
  urgency: ConsultUrgency;
  requestedBy: string;
  bedLabel?: string;
}

/**
 * 협진 서비스 (Facade) — FastAPI 연동.
 * 신장내과 ↔ 병리과 협진 생성·상태전이·회신. 상태는 백엔드 DB(공유 store)에 영속된다
 * → 두 부서 화면이 동일 데이터를 교차 조회한다.
 */
class ConsultService {
  async list(): Promise<Consult[]> {
    return api.get<Consult[]>("/consultations");
  }

  /** 협진 요청 생성. */
  async request(input: ConsultRequestInput): Promise<Consult> {
    return api.post<Consult>("/consultations", { kind: "pathology", ...input });
  }

  /** 협진 접수 — requested → in_progress. */
  async accept(consult: Consult, actor: string): Promise<Consult> {
    return api.post<Consult>(`/consultations/${consult.id}/accept`, { actor });
  }

  /** 회신 등록 — replied 로 전이(repliedAt 은 백엔드가 기록). */
  async reply(consult: Consult, reply: ConsultReply): Promise<Consult> {
    return api.post<Consult>(`/consultations/${consult.id}/reply`, {
      findings: reply.findings,
      diagnosis: reply.diagnosis,
      recommendation: reply.recommendation,
      author: reply.author,
      // 판독의 서명·AI 분석 근거(ROI·heatmap·신뢰도)도 함께 전송해 회신에 영속시킨다
      // → 신장내과 대시보드/병리 리포트 모달이 재조회 시에도 근거를 표시할 수 있다.
      signaturePath: reply.signaturePath,
      analysis: reply.analysis,
    });
  }
}

export const consultService = new ConsultService();
