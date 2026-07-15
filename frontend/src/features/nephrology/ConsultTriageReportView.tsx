import React from "react";
import { AlertTriangle, Activity, BrainCircuit, Flag, Info, UserX } from "lucide-react";
import { cn } from "@/lib/cn";

export interface ConsultTriageData {
  renal_status: {
    stage: string;
    abnormality: string;
    trend: "rapidly_worsening" | "worsening" | "stable" | "improving";
    xgboost_risk?: number;
    has_drug_data: boolean;
    kts_score: number;
    uncertainty: {
      confidence: number;
      missing_baseline: boolean;
    };
  };
  triage_result: "URGENT_CONSULT" | "CONSULT_RECOMMENDED" | "MONITOR_ONLY" | "HUMAN_REVIEW_REQUIRED";
  rationale: {
    rule_based: string;
    trend_based: string;
    ml_based: string;
    false_positive_suppression: string | null;
  };
  conflict: boolean;
  final_plan: string;
}

export function ConsultTriageReportView({ data }: { data: ConsultTriageData }) {
  // Override Rules
  const isHumanReview = 
    data.renal_status.uncertainty.confidence < 0.6 ||
    data.renal_status.uncertainty.missing_baseline ||
    data.conflict ||
    data.triage_result === "HUMAN_REVIEW_REQUIRED";

  const renderFlag = () => {
    if (isHumanReview) {
      return (
        <div className="flex items-center gap-2 rounded-md bg-red-600 px-4 py-3 font-bold text-white shadow-sm animate-pulse mb-4">
          <UserX className="size-5" />
          <span>HUMAN REVIEW REQUIRED (안전 규칙 발동: 의사 최종 판단 요망)</span>
        </div>
      );
    }
    
    switch(data.triage_result) {
      case "URGENT_CONSULT":
        return (
          <div className="flex items-center gap-2 rounded-md bg-red-600 px-4 py-3 font-bold text-white shadow-sm mb-4">
            <AlertTriangle className="size-5" />
            <span>URGENT CONSULT FLAG (즉시 협진 필요)</span>
          </div>
        );
      case "CONSULT_RECOMMENDED":
        return (
          <div className="flex items-center gap-2 rounded-md bg-orange-500 px-4 py-3 font-bold text-white shadow-sm mb-4">
            <Flag className="size-5" />
            <span>CONSULT RECOMMENDED (협진 권장)</span>
          </div>
        );
      case "MONITOR_ONLY":
        return (
          <div className="flex items-center gap-2 rounded-md bg-green-600 px-4 py-3 font-bold text-white shadow-sm mb-4">
            <Activity className="size-5" />
            <span>MONITOR ONLY (경과 관찰)</span>
          </div>
        );
      default: return null;
    }
  };

  return (
    <div className="space-y-4 text-sm font-sans">
      {renderFlag()}

      {/* 1. 신장 상태 요약 & KTS */}
      <div className="rounded-lg border border-border bg-card p-4 shadow-sm flex flex-col md:flex-row gap-4 justify-between">
        <div className="flex-1">
          <h3 className="mb-3 flex items-center gap-2 font-bold text-primary">
            <Activity className="size-4" />
            1. 신장 상태 요약 (Renal Status)
          </h3>
          <ul className="list-disc pl-5 space-y-1 text-foreground">
            <li><strong>AKI Stage:</strong> {data.renal_status.stage}</li>
            <li><strong>주요 이상 소견:</strong> {data.renal_status.abnormality}</li>
            <li>
              <strong>속도(Trend):</strong>{" "}
              <span className={cn("font-bold", 
                data.renal_status.trend.includes("worsening") ? "text-red-500" : 
                data.renal_status.trend === "improving" ? "text-green-500" : "text-foreground")}>
                {data.renal_status.trend === "rapidly_worsening" ? "급격한 악화 (Rapid Deterioration)" : 
                 data.renal_status.trend === "worsening" ? "악화 진행 중" : 
                 data.renal_status.trend === "improving" ? "호전 추세" : "안정적 (Stable)"}
              </span>
            </li>
            {data.renal_status.xgboost_risk !== undefined && (
              <li>
                <strong>XGBoost 예측:</strong> {data.renal_status.xgboost_risk}
                {!data.renal_status.has_drug_data && " (약물 데이터 부재 - 생리학적 수치 한정 판단)"}
              </li>
            )}
          </ul>
        </div>
        
        {/* KTS Score Highlight */}
        <div className="flex flex-col items-center justify-center bg-muted/30 rounded-lg p-4 border shrink-0 min-w-[140px]">
          <span className="text-xs font-semibold text-muted-foreground mb-1">KDIGO Trend Score</span>
          <span className={cn("text-3xl font-black", 
            data.renal_status.kts_score > 0.8 ? "text-red-600" :
            data.renal_status.kts_score > 0.6 ? "text-orange-500" : "text-green-600"
          )}>
            {data.renal_status.kts_score.toFixed(2)}
          </span>
          <span className="text-[10px] text-muted-foreground mt-1 text-center leading-tight">
            (진행 속도 기반 함수)
          </span>
        </div>
      </div>

      {/* 3. 근거 요약 */}
      <div className="rounded-lg border border-border bg-card p-4 shadow-sm">
        <h3 className="mb-3 flex items-center gap-2 font-bold text-primary">
          <Info className="size-4" />
          2. 트리아지 판단 근거 (Rationale)
        </h3>
        <ul className="list-disc pl-5 space-y-2 text-foreground">
          <li><strong>Rule 기반:</strong> {data.rationale.rule_based}</li>
          <li><strong>Trend 기반:</strong> {data.rationale.trend_based}</li>
          <li><strong>ML 엔진:</strong> {data.rationale.ml_based}</li>
          {data.rationale.false_positive_suppression && (
            <li className="text-orange-600 font-medium">
              <strong>False Positive 억제 발동:</strong> {data.rationale.false_positive_suppression}
            </li>
          )}
        </ul>
      </div>

      {/* 4. 의사결정 충돌 여부 */}
      {data.conflict && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 shadow-sm">
          <h3 className="mb-1 flex items-center gap-2 font-bold text-red-600">
            <BrainCircuit className="size-4" />
            3. 의사결정 충돌 (Decision Conflict)
          </h3>
          <p className="text-sm text-red-700">시스템 내부 모듈 간의 판단이 상충되어 우선순위 룰(Safety First)이 강제 적용되었습니다.</p>
        </div>
      )}

      {/* 5. 최종 계획 */}
      <div className="rounded-lg border border-border bg-blue-50/50 p-4 shadow-sm">
        <h3 className="mb-2 font-bold text-blue-800">4. 최종 권고 계획 (Plan)</h3>
        {isHumanReview ? (
          <p className="font-bold text-red-600">전문의 직접 차트 리뷰를 통한 최종 판단 필요</p>
        ) : (
          <p className="font-medium text-blue-900">{data.final_plan}</p>
        )}
        <p className="text-[11px] text-muted-foreground mt-2 border-t border-blue-200 pt-2">
          * 본 리포트는 CDSS 보조 수단이며, 자동 협진 요청을 실행하지 않습니다.
        </p>
      </div>
    </div>
  );
}
