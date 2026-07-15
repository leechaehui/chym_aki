import type {
  AiDraftResult,
  CareunitStat,
  IcuAkiPatient,
  IcuMonitorSummary,
  IcuPatientSearchResult,
  IcuPatientSummary,
  ModelMetrics,
  PatientShap,
  PatientTrends,
  PatientValidation,
  RiskHistory,
  RrtAssessment,
} from "@/types";
import { api } from "./http";

/**
 * ICU AKI 모니터 서비스 (Facade) — 실제 MIMIC-IV ICU 코호트 + 학습 모델.
 * 신장내과 데모 환자(rule-based)와 달리, 표준화 48h 전체 피처가 있어 학습 모델(source=model)이 유효.
 */
class IcuMonitorService {
  summary(): Promise<IcuMonitorSummary> {
    return api.get<IcuMonitorSummary>("/nephrology/icu/summary");
  }

  careunits(): Promise<CareunitStat[]> {
    return api.get<CareunitStat[]>("/nephrology/icu/careunits");
  }

  modelMetrics(): Promise<ModelMetrics> {
    return api.get<ModelMetrics>("/nephrology/icu/model-metrics");
  }

  list(opts: { limit?: number; offset?: number; careunit?: string; minRisk?: number } = {}): Promise<IcuAkiPatient[]> {
    const q = new URLSearchParams();
    q.set("limit", String(opts.limit ?? 30));
    if (opts.offset) q.set("offset", String(opts.offset));
    if (opts.careunit) q.set("careunit", opts.careunit);
    if (opts.minRisk) q.set("min_risk", String(opts.minRisk));
    return api.get<IcuAkiPatient[]>(`/nephrology/icu/aki-monitor?${q.toString()}`);
  }

  // ---- 환자 상세(검색 · Quick View · Trend · SHAP · Validation · Risk History) ----
  search(query: string, limit = 10): Promise<IcuPatientSearchResult[]> {
    const q = new URLSearchParams({ q: query, limit: String(limit) });
    return api.get<IcuPatientSearchResult[]>(`/nephrology/icu/patients/search?${q.toString()}`);
  }

  patientSummary(stayId: number): Promise<IcuPatientSummary> {
    return api.get<IcuPatientSummary>(`/nephrology/icu/patients/${stayId}/summary`);
  }

  trends(stayId: number): Promise<PatientTrends> {
    return api.get<PatientTrends>(`/nephrology/icu/patients/${stayId}/trends`);
  }

  shap(stayId: number, limit = 8): Promise<PatientShap> {
    return api.get<PatientShap>(`/nephrology/icu/patients/${stayId}/shap?limit=${limit}`);
  }

  validation(stayId: number): Promise<PatientValidation> {
    return api.get<PatientValidation>(`/nephrology/icu/patients/${stayId}/validation`);
  }

  riskHistory(stayId: number): Promise<RiskHistory> {
    return api.get<RiskHistory>(`/nephrology/icu/patients/${stayId}/risk-history`);
  }

  rrtAssessment(stayId: number): Promise<RrtAssessment> {
    return api.get<RrtAssessment>(`/nephrology/icu/patients/${stayId}/rrt-assessment`);
  }

  // 실제 ICU stay 기반 AI 진료 초안(SOAP) — 전사 텍스트 → 근거기반 SOAP/Problem/CDSS/검증.
  draftForStay(stayId: number, transcript: string): Promise<AiDraftResult> {
    return api.post<AiDraftResult>(`/nephrology/icu/patients/${stayId}/draft`, {
      patientId: String(stayId),
      transcript,
    });
  }
}

export const icuMonitorService = new IcuMonitorService();
