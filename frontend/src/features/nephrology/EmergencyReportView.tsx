import React from "react";
import { AlertTriangle, Clock, Activity, BrainCircuit, Stethoscope, CheckCircle2, XCircle, HelpCircle } from "lucide-react";
import { cn } from "@/lib/cn";

// --- Types for CDSS JSON ---
export interface CdssEvent {
  time: string;
  type: "vital" | "lab" | "intervention" | "medication";
  data?: Record<string, string | number>;
  action?: string;
  detail?: string;
  drug?: string;
  dose?: string;
  trend?: "worsening" | "improving" | "stable";
}

export interface CdssIntervention {
  time: string;
  action: string;
  response: string;
  effect: "likely_effective" | "uncertain_effect" | "likely_non_responsive";
}

export interface CdssEmergencyJson {
  event_stream: CdssEvent[];
  state: {
    severity: string;
    syndrome: string[];
    suspected_diagnosis: string[];
    risk_score: Record<string, number>;
    uncertainty: {
      confidence: number;
      missing_baseline: boolean;
    };
  };
  decision: {
    rule_trigger: string[];
    ml_route: string;
    conflict: boolean;
    output_plan: string[];
  };
  interventions: CdssIntervention[];
  xgboost_output?: {
    risk_score: number;
    has_drug_data: boolean;
    has_dialysis_data: boolean;
  };
}

function formatEventData(data?: Record<string, string | number>) {
  if (!data) return "";
  return Object.entries(data).map(([k, v]) => {
    if (k === "Cr") return `Cr ${v}`;
    if (k === "SpO2") return `SpO2 ${v}%`;
    if (k === "HR") return `HR ${v}회/분`;
    return `${k} ${v}`;
  }).join(", ");
}

export function EmergencyReportView({ data }: { data: CdssEmergencyJson }) {
  // Rule 0: Override Rule (HUMAN REVIEW REQUIRED)
  const isHumanReviewRequired = 
    data.state.uncertainty.confidence < 0.6 ||
    data.state.uncertainty.missing_baseline ||
    data.decision.conflict ||
    (data.xgboost_output && (!data.xgboost_output.has_drug_data || !data.xgboost_output.has_dialysis_data));

  return (
    <div className="space-y-4 text-sm">
      {/* 0. OVERRIDE ALERT */}
      {isHumanReviewRequired && (
        <div className="flex items-center gap-2 rounded-md bg-red-600 px-4 py-3 font-bold text-white shadow-sm animate-pulse">
          <AlertTriangle className="size-5" />
          <span>의사 최종 확인 필요 (Human Review Required)</span>
        </div>
      )}

      {/* 1. 초기 중증 상태 */}
      <div className="rounded-lg border border-border bg-card p-4 shadow-sm">
        <h3 className="mb-3 flex items-center gap-2 font-bold text-primary">
          <Activity className="size-4" />
          1. 초기 중증 상태
        </h3>
        <ul className="list-disc pl-5 space-y-1 text-foreground">
          <li><strong>중증도:</strong> <span className="uppercase text-red-600 font-bold">{data.state.severity}</span></li>
          <li><strong>동반 증후군:</strong> {data.state.syndrome.join(", ")}</li>
          {data.xgboost_output && (
            <li>
              <strong>XGBoost 위험도 (부분 신호):</strong> {data.xgboost_output.risk_score} 
              {!data.xgboost_output.has_drug_data && " (약물 데이터 부재 - 신뢰도 하향)"}
            </li>
          )}
          {isHumanReviewRequired && (
            <li className="font-bold text-red-500 mt-2">신뢰도 제한 상태 (Limited Reliability State)</li>
          )}
        </ul>
      </div>

      {/* 2. 시간 기반 임상 이벤트 요약 */}
      <div className="rounded-lg border border-border bg-card p-4 shadow-sm">
        <h3 className="mb-3 flex items-center gap-2 font-bold text-primary">
          <Clock className="size-4" />
          2. 시간 기반 임상 이벤트 요약
        </h3>
        <div className="space-y-2 border-l-2 border-muted pl-4">
          {data.event_stream.map((ev, i) => (
            <div key={i} className="relative">
              <div className="absolute -left-[21px] top-1.5 size-2 rounded-full bg-primary/50" />
              <div className="flex items-start gap-2">
                <span className="font-mono text-xs font-semibold text-muted-foreground mt-0.5">{ev.time}</span>
                <div>
                  <span className="font-medium capitalize text-foreground">
                    [{ev.type === "lab" ? "검사" : ev.type === "vital" ? "활력징후" : ev.type === "intervention" ? "처치" : "약물"}]
                  </span>{" "}
                  {(ev.type === "vital" || ev.type === "lab") ? formatEventData(ev.data) : ""}
                  {ev.type === "intervention" ? `${ev.action} - ${ev.detail}` : ""}
                  {ev.type === "medication" ? `${ev.drug} (${ev.dose})` : ""}
                  {ev.trend === "worsening" && <span className="ml-2 font-bold text-red-500">(악화됨)</span>}
                  {ev.trend === "improving" && <span className="ml-2 font-bold text-green-500">(호전됨)</span>}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 3. 처치 및 반응 분석 */}
      <div className="rounded-lg border border-border bg-card p-4 shadow-sm">
        <h3 className="mb-3 flex items-center gap-2 font-bold text-primary">
          <Stethoscope className="size-4" />
          3. 처치 및 반응 분석
        </h3>
        <div className="space-y-3">
          {data.interventions.length === 0 && <div className="text-muted-foreground">시행된 처치 기록 없음</div>}
          {data.interventions.map((inv, i) => (
            <div key={i} className="flex flex-col gap-1 rounded bg-muted/30 p-2">
              <div className="flex items-center justify-between">
                <strong className="text-foreground">{inv.time} - {inv.action}</strong>
                <span className="text-xs bg-background px-2 py-0.5 rounded border">{inv.response}</span>
              </div>
              <div className="flex items-center gap-1 text-xs font-medium">
                분류: 
                {inv.effect === "likely_effective" && <span className="text-green-600 flex items-center gap-1"><CheckCircle2 className="size-3" /> 효과 있음</span>}
                {inv.effect === "uncertain_effect" && <span className="text-orange-500 flex items-center gap-1"><HelpCircle className="size-3" /> 불확실</span>}
                {inv.effect === "likely_non_responsive" && <span className="text-red-500 flex items-center gap-1"><XCircle className="size-3" /> 효과 없음</span>}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 4. CDSS 의사결정 요약 */}
      <div className="rounded-lg border border-border bg-card p-4 shadow-sm">
        <h3 className="mb-3 flex items-center gap-2 font-bold text-primary">
          <BrainCircuit className="size-4" />
          4. CDSS 의사결정 요약
        </h3>
        <ul className="list-disc pl-5 space-y-1 text-foreground">
          <li><strong>발동 규칙:</strong> {data.decision.rule_trigger.join(", ") || "없음"}</li>
          <li><strong>판단 경로(Route):</strong> {data.decision.ml_route}</li>
          {data.xgboost_output && (
            <li><strong>XGBoost 위험도 (생리학적 수치 한정):</strong> {data.xgboost_output.risk_score}</li>
          )}
          {data.decision.conflict && (
            <li className="font-bold text-red-500 mt-2">의사결정 충돌 감지 (Decision Conflict Detected)</li>
          )}
        </ul>
      </div>

      {/* 5. 임상 판단 */}
      <div className="rounded-lg border border-border bg-card p-4 shadow-sm bg-blue-50/30">
        <h3 className="mb-2 font-bold text-blue-800">5. 임상 판단 (Impression)</h3>
        <p className="font-medium text-blue-900">{data.state.suspected_diagnosis.join(", ")}</p>
      </div>

      {/* 6. 계획 (Plan) */}
      <div className="rounded-lg border border-border bg-card p-4 shadow-sm bg-green-50/30">
        <h3 className="mb-2 font-bold text-green-800">6. 계획 (Plan)</h3>
        {isHumanReviewRequired ? (
          <p className="font-bold text-red-600 text-lg">의사 최종 검토 필요</p>
        ) : (
          <ul className="list-disc pl-5 font-medium text-green-900">
            {data.decision.output_plan.map((p, i) => <li key={i}>{p}</li>)}
          </ul>
        )}
      </div>
    </div>
  );
}
