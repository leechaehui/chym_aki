import { useEffect, useState, type ReactNode } from "react";
import { X } from "lucide-react";
import type { IcuPatientSummary, ModelMetrics, PatientIdentity } from "@/types";
import { icuMonitorService } from "@/services/icuMonitorService";
import { AiRiskGauge } from "./AiRiskGauge";
import { riskBandFor } from "@/lib/riskLabel";
import { fakeKoreanName } from "@/lib/fakeName";
import { cn } from "@/lib/cn";
import { LoadingSpinner } from "@/components/common/LoadingSpinner";
import { ClinicalActionModal } from "./ClinicalActionModal";
import { PatientReportModal } from "./PatientReportModal";
import { useQueryClient } from "@tanstack/react-query";
import { useActiveAlarmStore } from "@/store/activeAlarmStore";

function Field({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-2 border-b border-border/40 py-1 last:border-0">
      <span className="text-[11px] text-muted-foreground">{label}</span>
      <span className="text-[12px] font-medium tabular-nums">{value}</span>
    </div>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="rounded-lg border border-border p-3">
      <div className="mb-1.5 text-[12px] font-semibold">{title}</div>
      {children}
    </div>
  );
}

const naIfNull = (v: number | string | null | undefined, suffix = "") =>
  v === null || v === undefined ? <span className="text-muted-foreground">비측정</span> : `${v}${suffix}`;

/**
 * Patient Quick View — 검색/행 클릭 시 우측 드로어로 환자 핵심 정보를 빠르게 보여준다.
 * 기본 정보 · 입원 정보 · AKI 상태 · 신장 기능 + 위험 게이지. 전체 검증 리포트로 진입 가능.
 * MIMIC 비식별로 결측인 값(체중/키/BMI/BUN/병실)은 '비측정'으로 표기한다.
 */
export function PatientQuickView({
  patient,
  metrics,
  onClose,
  onAddStay,
  isAdding,
}: {
  patient: PatientIdentity | null;
  metrics: ModelMetrics | null;
  onClose: () => void;
  onAddStay?: (stayId: number) => void;
  isAdding?: boolean;
}) {
  const [summary, setSummary] = useState<IcuPatientSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const queryClient = useQueryClient();
  const dismissAlarm = useActiveAlarmStore((state) => state.dismissAlarm);

  useEffect(() => {
    if (!patient) return;
    setSummary(null);
    setLoading(true);
    icuMonitorService
      .patientSummary(patient.stayId)
      .then(setSummary)
      .catch(() => setSummary(null))
      .finally(() => setLoading(false));
  }, [patient]);

  if (!patient) return null;
  // subjectId가 넘어오면 바로 쓰고, 없으면 summary에서 가져온다. 둘 다 없으면 "환자 이름" (임시)
  const subjectId = patient.subjectId ?? summary?.subjectId;
  const name = subjectId ? fakeKoreanName(subjectId) : "환자 정보 조회 중...";
  const band = riskBandFor(summary?.riskScore ?? 0, summary?.stage);

  return (
    <>
      <div className="fixed inset-0 z-40 bg-black/40 backdrop-blur-sm" onClick={onClose} />
      <aside className="fixed right-0 top-0 z-50 flex h-full w-full max-w-md flex-col border-l border-border bg-card shadow-xl">
        <header className="flex items-center justify-between border-b border-border px-4 py-3">
          <div>
            <div className="text-sm font-semibold">{name}</div>
            <div className="text-[11px] text-muted-foreground">Stay {patient.stayId} {subjectId ? `· Subject ${subjectId}` : ""}</div>
          </div>
          <button onClick={onClose} className="rounded-sm text-muted-foreground hover:text-foreground">
            <X className="size-4" />
          </button>
        </header>

        <div className="flex-1 space-y-3 overflow-y-auto p-4">
          {loading && <LoadingSpinner label="환자 정보 불러오는 중" />}
          {summary && (
            <>
              <div className="flex items-center justify-between gap-3 rounded-lg border border-border p-3">
                <AiRiskGauge score={summary.riskScore} predictedStage={summary.stage} />
                <div className="text-right">
                  <div className={cn("text-sm font-bold", band.textClass)}>{band.dot} {band.label}</div>
                  <div className="text-[11px] text-muted-foreground">{summary.stage}</div>
                  <div className="text-[11px] text-muted-foreground">
                    중증(Stage2-3) {(summary.pStage2Plus * 100).toFixed(0)}%
                  </div>
                </div>
              </div>

              <Section title="기본 정보">
                <Field label="이름(합성)" value={name} />
                <Field label="나이/성별" value={`${naIfNull(summary.age ?? patient.age)} / ${summary.gender ?? patient.gender ?? "—"}`} />
                <Field label="키 / 체중" value={<>{naIfNull(summary.heightCm, "cm")} / {naIfNull(summary.weightKg, "kg")}</>} />
                <Field label="BMI" value={naIfNull(summary.bmi)} />
              </Section>

              <Section title="입원 정보">
                <Field label="병동" value={summary.careunit} />
                <Field label="병실" value={naIfNull(summary.bed)} />
                <Field label="입원 유형" value={summary.admissionType ?? "—"} />
                <Field label="입실 후 경과" value={summary.elapsedText ?? "—"} />
                <Field label="예측 시각(입실+48h)" value={summary.predictAt ?? "—"} />
              </Section>

              <Section title="AKI 상태">
                <Field label="예측 단계" value={summary.stage} />
                <Field label="위험 점수" value={summary.riskScore} />
                <Field label="중증 AKI(Stage2-3) 확률" value={`${(summary.pStage2Plus * 100).toFixed(0)}%`} />
              </Section>

              <Section title="신장 기능">
                <Field label="Creatinine" value={naIfNull(summary.creatinine, " mg/dL")} />
                <Field label="기저 Creatinine" value={naIfNull(summary.baselineCreatinine, " mg/dL")} />
                <Field label="eGFR" value={naIfNull(summary.egfr, " mL/min")} />
                <Field label="BUN" value={naIfNull(summary.bun, " mg/dL")} />
                <Field label="Urine Output" value={naIfNull(summary.urineRateMlKgH, " mL/kg/h")} />
              </Section>
            </>
          )}
        </div>

        <footer className="border-t border-border p-3 space-y-2">
          <button
            type="button"
            onClick={() => onAddStay?.(patient.stayId)}
            disabled={isAdding}
            className="w-full rounded-md bg-secondary px-3 py-2 text-sm font-medium text-foreground hover:bg-muted disabled:cursor-not-allowed disabled:opacity-60"
          >
            {isAdding ? "파이프라인 추가 중…" : "파이프라인에 환자 추가"}
          </button>
          {(summary?.riskScore ?? 0) >= 70 ? (
            <ClinicalActionModal
              patient={patient}
              onComplete={() => {
                dismissAlarm(patient.stayId);
                queryClient.setQueriesData({ queryKey: ["icuPatients"] }, (old: any[] | undefined) => {
                  if (!old) return old;
                  return old.map((pt) => pt.stayId === patient.stayId ? { ...pt, status: "acknowledged" } : pt);
                });
                onClose();
              }}
              trigger={
                <button className={cn(
                  "w-full rounded-md px-3 py-2 text-sm font-medium text-white transition-colors shadow-sm",
                  (summary?.riskScore ?? 0) >= 90 ? "bg-red-600 hover:bg-red-700" : "bg-orange-500 hover:bg-orange-600"
                )}>
                  긴급 조치 리포트 열기
                </button>
              }
            />
          ) : (
            <PatientReportModal
              patient={patient}
              metrics={metrics}
              trigger={
                <button className="w-full rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90">
                  전체 검증 리포트 보기
                </button>
              }
            />
          )}
        </footer>
      </aside>
    </>
  );
}
