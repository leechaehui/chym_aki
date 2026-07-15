import React from "react";
import { BrainCircuit, Activity, HeartPulse, Network, AlertTriangle, FileQuestion, Flag, UserX } from "lucide-react";
import { cn } from "@/lib/cn";

export interface NephrologyTriageData {
  triage_result: "GREEN" | "YELLOW" | "ORANGE" | "RED";
  triage_desc: string;
  risk_score: number;
  recommendation: string;
  key_drivers: string[];
  xgboost_feature_vector: {
    group_1_kdigo: Record<string, number>;
    group_2_dynamics: Record<string, number>;
    group_3_physiologic: Record<string, number>;
    group_5_uncertainty: Record<string, number>;
    group_6_missingness: Record<string, number>;
    group_7_interaction: Record<string, number>;
  };
}

export function NephrologyTriageReportView({ data }: { data: NephrologyTriageData }) {
  if (!data || !data.xgboost_feature_vector) {
    return <div className="p-4 text-center text-muted-foreground">데이터가 없거나 처리 중 오류가 발생했습니다.</div>;
  }

  const vector = data.xgboost_feature_vector;

  const renderFlag = () => {
    switch(data.triage_result) {
      case "RED":
        return (
          <div className="flex items-center gap-2 rounded-md bg-red-600 px-4 py-3 font-bold text-white shadow-sm mb-4">
            <AlertTriangle className="size-5" />
            <span>{data.triage_desc}</span>
          </div>
        );
      case "ORANGE":
        return (
          <div className="flex items-center gap-2 rounded-md bg-orange-500 px-4 py-3 font-bold text-white shadow-sm mb-4">
            <Flag className="size-5" />
            <span>{data.triage_desc}</span>
          </div>
        );
      case "YELLOW":
        return (
          <div className="flex items-center gap-2 rounded-md bg-yellow-500 px-4 py-3 font-bold text-white shadow-sm mb-4">
            <Activity className="size-5" />
            <span>{data.triage_desc}</span>
          </div>
        );
      case "GREEN":
        return (
          <div className="flex items-center gap-2 rounded-md bg-green-600 px-4 py-3 font-bold text-white shadow-sm mb-4">
            <Activity className="size-5" />
            <span>{data.triage_desc}</span>
          </div>
        );
      default: return null;
    }
  };

  const renderGroup = (title: string, icon: React.ReactNode, groupData: Record<string, number>, color: string) => {
    return (
      <div className={cn("rounded-lg border bg-card p-4 shadow-sm", color)}>
        <h3 className="mb-3 flex items-center gap-2 font-bold">
          {icon}
          {title}
        </h3>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
          {Object.entries(groupData).map(([key, value]) => (
            <div key={key} className="flex justify-between border-b border-border/40 py-1 last:border-0">
              <span className="text-muted-foreground truncate pr-2" title={key}>{key}</span>
              <span className="font-semibold tabular-nums text-foreground">{value}</span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-4 text-sm font-sans">
      {renderFlag()}

      <div className="rounded-lg border border-border bg-card p-4 shadow-sm flex flex-col md:flex-row gap-4 justify-between">
        <div className="flex-1">
          <h3 className="mb-3 flex items-center gap-2 font-bold text-primary">
            <BrainCircuit className="size-4" />
            트리아지 판단 근거 (XGBoost Model)
          </h3>
          <ul className="list-disc pl-5 space-y-1 text-foreground">
            {data.key_drivers.map((driver, idx) => (
              <li key={idx}><strong>{driver}</strong></li>
            ))}
          </ul>
          <div className="mt-4 pt-3 border-t">
            <p className="font-medium text-blue-900">권고: {data.recommendation}</p>
          </div>
        </div>
        
        <div className="flex flex-col items-center justify-center bg-muted/30 rounded-lg p-4 border shrink-0 min-w-[140px]">
          <span className="text-xs font-semibold text-muted-foreground mb-1">XGBoost 위험 점수</span>
          <span className={cn("text-4xl font-black", 
            data.risk_score >= 80 ? "text-red-600" :
            data.risk_score >= 60 ? "text-orange-500" :
            data.risk_score >= 30 ? "text-yellow-600" : "text-green-600"
          )}>
            {data.risk_score}
          </span>
          <span className="text-[10px] text-muted-foreground mt-1 text-center leading-tight">
            (머신러닝 예측 위험도)
          </span>
        </div>
      </div>

      <h3 className="flex items-center gap-2 font-bold mt-6 mb-2">
        <Network className="size-4" />
        XGBoost 피처 벡터 상세 내역
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {renderGroup("1. KDIGO 신기능 지표", <Activity className="size-4" />, vector.group_1_kdigo, "border-blue-200 text-blue-900 bg-blue-50/30")}
        {renderGroup("2. 신장 기능 변화율 (Dynamics)", <Activity className="size-4" />, vector.group_2_dynamics, "border-indigo-200 text-indigo-900 bg-indigo-50/30")}
        {renderGroup("3. 생리학적 불안정성", <HeartPulse className="size-4" />, vector.group_3_physiologic, "border-red-200 text-red-900 bg-red-50/30")}
        {renderGroup("5. 데이터 불확실성 및 품질", <FileQuestion className="size-4" />, vector.group_5_uncertainty, "border-amber-200 text-amber-900 bg-amber-50/30")}
        {renderGroup("6. 결측치 플래그", <AlertTriangle className="size-4" />, vector.group_6_missingness, "border-orange-200 text-orange-900 bg-orange-50/30")}
      </div>

      {renderGroup("7. 파생 상호작용 지표", <Network className="size-4" />, vector.group_7_interaction, "border-emerald-200 text-emerald-900 bg-emerald-50/30")}
      
      <p className="text-[11px] text-muted-foreground mt-2 border-t pt-2 text-right">
        * 본 결과는 AKI 모델링 데이터를 기반으로 생성된 XGBoost Feature 트리아지 결과입니다. (병리/ATN 피처 제외)
      </p>
    </div>
  );
}
