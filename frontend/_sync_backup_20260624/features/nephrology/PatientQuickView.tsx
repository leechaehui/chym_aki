import { useEffect, useState, type ReactNode } from "react";
import { X } from "lucide-react";
import type { IcuPatientSummary, ModelMetrics, PatientIdentity } from "@/types";
import { icuMonitorService } from "@/services/icuMonitorService";
import { AiRiskGauge } from "./AiRiskGauge";
import { riskBandFor } from "@/lib/riskLabel";
import { fakeKoreanName } from "@/lib/fakeName";
import { cn } from "@/lib/cn";
import { LoadingSpinner } from "@/components/common/LoadingSpinner";
import { PatientReportModal } from "./PatientReportModal";

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
}: {
  patient: PatientIdentity | null;
  metrics: ModelMetrics | null;
  onClose: () => void;
}) {
  const [summary, setSummary] = useState<IcuPatientSummary | null>(null);
  const [loading, setLoading] = useState(false);

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
  const name = fakeKoreanName(patient.subjectId);
  const band = riskBandFor(summary?.riskScore ?? 0);

  return (
    <>
      <div className="fixed inset-0 z-40 bg-black/40 backdrop-blur-sm" onClick={onClose} />
      <aside className="fixed right-0 top-0 z-50 flex h-full w-full max-w-md flex-col border-l border-border bg-card shadow-xl">
        <header className="flex items-center justify-between border-b border-border px-4 py-3">
          <div>
            <div className="text-sm font-semibold">{name}</div>
            <div className="text-[11px] text-muted-foreground">합성 표시명 · Stay {patient.stayId}</div>
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
                <AiRiskGauge score={summary.riskScore} />
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
                <Field label="나이/성별" value={`${naIfNull(summary.age)} / ${summary.gender}`} />
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

        <footer className="border-t border-border p-3">
          <PatientReportModal
            patient={patient}
            metrics={metrics}
            trigger={
              <button className="w-full rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90">
                전체 검증 리포트 보기
              </button>
            }
          />
        </footer>
      </aside>
    </>
  );
}
