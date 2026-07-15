import { User, FlaskConical, Send } from "lucide-react";
import type { LabValue, Patient } from "@/types";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { StatusBadge } from "@/components/common/StatusBadge";
import { riskBandFor, RISK_BADGE_TONE } from "@/lib/riskLabel";
import { cn } from "@/lib/cn";

// 요약 바에 노출할 핵심 검사 키(신장 기능 우선).
const KEY_LABS: { key: string; label: string }[] = [
  { key: "cr", label: "Cr" },
  { key: "egfr", label: "eGFR" },
  { key: "bun", label: "BUN" },
  { key: "k", label: "K" },
];

const flagCls: Record<LabValue["flag"], string> = {
  normal: "text-foreground",
  high: "text-destructive",
  low: "text-accent",
};

/**
 * 환자 핵심 요약 바 — 식별 정보 + AKI 단계/위험 + 핵심검사 + 병리 협진 요청.
 * 신장내과 대시보드 우측 상세 영역 최상단에 위치한다(재작성본 — 역할 동등).
 */
export function PatientSummaryBar({ patient, onConsult }: { patient: Patient; onConsult: () => void }) {
  // 배지는 AI 위험점수(모델) 기준으로 통일 — 목록 레일과 동일.
  const band = riskBandFor(patient.aiRiskScore);
  const labOf = (key: string) => patient.labs.find((l) => l.key === key);

  return (
    <Card>
      <CardContent className="flex flex-col gap-3 p-4">
        {/* 상단: 식별 + AKI 단계/위험 + 협진 버튼 */}
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <span className="flex items-center gap-1.5 text-base font-semibold text-foreground">
                <User className="size-4 text-primary" />
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
              <div className="text-[10px] text-muted-foreground">AI 위험점수</div>
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
              <Send className="size-4" /> 병리 협진 요청
            </Button>
          </div>
        </div>

        {/* 하단: 핵심 검사값 */}
        <div className="flex flex-wrap gap-2 border-t border-border pt-3">
          <span className="flex items-center gap-1 text-[11px] text-muted-foreground">
            <FlaskConical className="size-3.5 text-primary" /> 핵심검사
          </span>
          {KEY_LABS.map(({ key, label }) => {
            const lab = labOf(key);
            return (
              <span key={key} className="rounded-md border border-border bg-muted/30 px-2 py-1 text-[12px]">
                <span className="text-muted-foreground">{label} </span>
                <span className={cn("font-semibold tabular-nums", lab ? flagCls[lab.flag] : "text-muted-foreground")}>
                  {lab ? `${lab.value}${lab.unit ? ` ${lab.unit}` : ""}` : "—"}
                </span>
              </span>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}
