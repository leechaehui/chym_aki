import type { Patient } from "@/types";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { StatusBadge } from "@/components/common/StatusBadge";
import { riskBandFor, RISK_BADGE_TONE } from "@/lib/riskLabel";
import { cn } from "@/lib/cn";

/**
 * 환자 핵심 요약 바 — 식별 정보 + AKI 단계/위험 + 핵심검사 + 병리 협진 요청.
 * 신장내과 대시보드 우측 상세 영역 최상단에 위치한다(재작성본 — 역할 동등).
 */
export function PatientSummaryBar({ patient, predictedStage, onConsult }: { patient: Patient; predictedStage?: string | null; onConsult: () => void }) {
  // 배지는 최종 예측 Stage 기준으로 통일한다.
  const band = riskBandFor(patient.aiRiskScore, predictedStage);

  return (
    <Card>
      <CardContent className="flex flex-col gap-3 p-4">
        {/* 상단: 식별 + AKI 단계/위험 + 협진 버튼 */}
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-base font-semibold text-foreground">
                {patient.name}
              </span>
              <span className="text-[12px] text-muted-foreground">
                {patient.sex === "M" ? "남" : "여"}/{patient.age} · {patient.mrn}
              </span>
              <StatusBadge label={band.label} tone={RISK_BADGE_TONE[band.tier]} />
            </div>
            <p className="mt-1 truncate text-[12px] text-muted-foreground" title={patient.diagnosis}>
              {patient.diagnosis} · {patient.room} · 담당 {patient.attending}
            </p>
          </div>

          <div className="flex items-center gap-3">
            <div className="text-right">
              <div className="text-[10px] text-muted-foreground">AKI Prediction Score</div>
              <div
                className={cn(
                  "text-lg font-bold tabular-nums",
                  patient.aiRiskScore >= 80 ? "text-destructive" : patient.aiRiskScore >= 50 ? "text-warning" : "text-success",
                )}
              >
                {patient.aiRiskScore}
              </div>
            </div>
            <Button size="sm" variant="outline" onClick={onConsult}>
              병리 협진 요청
            </Button>
          </div>
        </div>

      </CardContent>
    </Card>
  );
}
