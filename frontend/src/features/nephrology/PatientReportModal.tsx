import { useEffect, useState, type ReactNode } from "react";
import { FileText, CheckCircle2, XCircle, AlertTriangle } from "lucide-react";
import type {
  IcuPatientSummary,
  ModelMetrics,
  PatientIdentity,
  PatientShap,
  PatientTrends,
  PatientValidation,
  RiskHistory,
  RrtAssessment,
} from "@/types";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { cn } from "@/lib/cn";
import { icuMonitorService } from "@/services/icuMonitorService";
import { fakeKoreanName } from "@/lib/fakeName";
import { riskBandFor } from "@/lib/riskLabel";
import { AiRiskGauge } from "./AiRiskGauge";
import { ValidationReportBody } from "./ModelPerformancePanel";
import { CalibrationPlot } from "./CalibrationPlot";
import { ConfusionMatrix } from "./ConfusionMatrix";
import { RiskStratification } from "./RiskStratification";
import { ShapBars } from "./ShapBars";
import { PatientTrendGraphs } from "./PatientTrendGraphs";
import { RiskHistoryChart } from "./RiskHistoryChart";
import { RrtTriggerPanel } from "./RrtTriggerPanel";

function Row({ label, value, tone }: { label: string; value: ReactNode; tone?: string }) {
  return (
    <div className="flex items-center justify-between gap-2 border-b border-border/40 py-1 last:border-0">
      <span className="text-muted-foreground">{label}</span>
      <span className={cn("tabular-nums font-medium", tone)}>{value}</span>
    </div>
  );
}

function ReportSection({ index, title, subtitle, children }: { index: number; title: string; subtitle?: string; children: ReactNode }) {
  return (
    <section className="mt-4 first:mt-0">
      <div className="mb-2 flex items-baseline gap-2 border-b border-border pb-1">
        <span className="text-[11px] font-bold text-primary">{index}</span>
        <h3 className="text-[13px] font-semibold">{title}</h3>
        {subtitle && <span className="text-[11px] font-normal text-muted-foreground">· {subtitle}</span>}
      </div>
      {children}
    </section>
  );
}

/**
 * 환자별 검증 리포트 — "임상의가 빠르게 판단하는 구조".
 * 순서: 환자정보 → 현재상태 → AKI 예측 → 투석/RRT 주의 → 추세 → SHAP → 모델 지표(보정/혼동/게이지/구간).
 * 추세·SHAP·검증·위험추세는 모두 환자별 API 로 개별 계산한다(모든 환자 동일 금지).
 * 환자 식별 최소 정보(PatientIdentity)만 받고 표시값은 모달이 열릴 때 백엔드에서 로드한다.
 */
export function PatientReportModal({
  patient,
  metrics,
  trigger,
  open: controlledOpen,
  onOpenChange,
}: {
  patient: PatientIdentity;
  metrics: ModelMetrics | null;
  /** 커스텀 트리거(없으면 기본 "검증 리포트" 버튼). controlled 모드면 생략 가능. */
  trigger?: ReactNode;
  /** controlled 열림 상태(딥링크 등 외부에서 여는 경우). 주면 내부 상태 대신 이걸 쓴다. */
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}) {
  // 딥링크 등 외부 제어를 위해 controlled/uncontrolled 를 모두 지원한다.
  const [uncontrolledOpen, setUncontrolledOpen] = useState(false);
  const isControlled = controlledOpen !== undefined;
  const open = isControlled ? controlledOpen : uncontrolledOpen;
  const setOpen = (v: boolean) => {
    if (isControlled) onOpenChange?.(v);
    else setUncontrolledOpen(v);
  };
  const [summary, setSummary] = useState<IcuPatientSummary | null>(null);
  const [trends, setTrends] = useState<PatientTrends | null>(null);
  const [shap, setShap] = useState<PatientShap | null>(null);
  const [validation, setValidation] = useState<PatientValidation | null>(null);
  const [riskHistory, setRiskHistory] = useState<RiskHistory | null>(null);
  const [rrt, setRrt] = useState<RrtAssessment | null>(null);

  const name = fakeKoreanName(patient.subjectId ?? patient.stayId);
  // Patient/PatientIdentity 어느 쪽이 와도 되도록 식별자 필드는 느슨하게 접근(컴파일타임 캐스트, 런타임 무영향).
  const idSrc = patient as { stayId?: number; id?: string; mrn?: string };

  // EMR 환자 ID에서 숫자만 안전하게 추출하여 stay_id/subject_id 조회 (NaN 오류 방어)
  useEffect(() => {
    if (!open) return;
    const cleanId = String(idSrc.stayId ?? "").replace(/[^\d]/g, "") || String(idSrc.id ?? "").replace(/[^\d]/g, "") || String(idSrc.mrn ?? "").replace(/[^\d]/g, "");
    const id = Number(cleanId);
    if (!Number.isFinite(id) || id === 0) return;
    icuMonitorService.patientSummary(id).then(setSummary).catch(() => {});
    icuMonitorService.trends(id).then(setTrends).catch(() => {});
    icuMonitorService.shap(id).then(setShap).catch(() => {});
    icuMonitorService.validation(id).then(setValidation).catch(() => {});
    icuMonitorService.riskHistory(id).then(setRiskHistory).catch(() => {});
    icuMonitorService.rrtAssessment(id).then(setRrt).catch(() => setRrt(null));
  }, [open, idSrc.stayId, idSrc.id, idSrc.mrn]);

  const riskScore = summary?.riskScore ?? 0;
  const band = riskBandFor(riskScore, summary?.stage);
  const akiProb = validation?.pAki ?? null;
  const correct = validation?.correct ?? null;

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      {trigger ? (
        <span onClick={() => setOpen(true)}>{trigger}</span>
      ) : isControlled ? null : (
        <button
          onClick={() => setOpen(true)}
          className="flex items-center gap-1 rounded-md bg-secondary px-2 py-1 text-[11px] font-medium text-foreground hover:bg-primary hover:text-primary-foreground"
          title="이 환자의 검증 리포트(추세·SHAP·보정 포함)"
        >
          <FileText className="size-3.5" /> 검증 리포트
        </button>
      )}

      <DialogContent className="max-h-[88vh] max-w-3xl overflow-y-auto">
        <DialogHeader>
          <DialogTitle>검증 리포트 · {name}</DialogTitle>
        </DialogHeader>

        {/* 1. 환자 정보 */}
        <ReportSection index={1} title="환자 정보">
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
            <span>이름(합성) <b className="text-foreground">{name}</b></span>
            <span>Stay {patient.stayId}</span>
            <span>나이/성별 {patient.age ?? "—"} / {patient.gender}</span>
            <span>병동 {patient.careunit}</span>
          </div>
        </ReportSection>

        {/* 2. 현재 상태 요약 */}
        <ReportSection index={2} title="현재 상태 요약">
          <div className="flex items-center justify-between gap-3 rounded-lg border border-border p-3">
            <AiRiskGauge score={riskScore} predictedStage={summary?.stage} />
            <div className="text-right text-[12px]">
              <div className={cn("text-sm font-bold", band.textClass)}>{band.dot} {band.label} ({riskScore})</div>
              <div className="text-muted-foreground">예측 단계 {summary?.stage ?? "—"}</div>
              {summary?.predictAt && <div className="text-muted-foreground">예측 시각 {summary.predictAt}</div>}
              {summary?.elapsedText && <div className="text-muted-foreground">{summary.elapsedText}</div>}
            </div>
          </div>
        </ReportSection>

        {/* 3. 검사 결과 (신장 패널 상세 + KDIGO 해석) */}
        <ReportSection index={3} title="검사 결과" subtitle="신장 패널 · 현재값 + KDIGO 해석">
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="rounded-lg border border-border p-3 text-[12px]">
              <Row label="Creatinine (현재)" value={summary?.creatinine != null ? `${summary.creatinine} mg/dL` : "비측정"} />
              <Row label="기저 Creatinine" value={summary?.baselineCreatinine != null ? `${summary.baselineCreatinine} mg/dL` : "비측정"} />
              <Row label="eGFR" value={summary?.egfr != null ? `${summary.egfr} mL/min/1.73m²` : "비측정"} />
              <Row label="BUN" value={summary?.bun != null ? `${summary.bun} mg/dL` : "비측정"} />
            </div>
            <div className="rounded-lg border border-border p-3 text-[12px]">
              <Row label="시간당 소변량" value={summary?.urineRateMlKgH != null ? `${summary.urineRateMlKgH} mL/kg/h` : "비측정"} />
              <Row label="체중 / 키" value={`${summary?.weightKg ?? "—"} kg / ${summary?.heightCm ?? "—"} cm`} />
              <Row label="BMI" value={summary?.bmi ?? "비측정"} />
              <Row label="입실 후 경과" value={summary?.elapsedText ?? "—"} />
            </div>
          </div>
          {(() => {
            const cr = summary?.creatinine ?? null;
            const base = summary?.baselineCreatinine ?? null;
            const uo = summary?.urineRateMlKgH ?? null;
            const ratio = cr && base && base > 0 ? cr / base : null;
            const delta = cr != null && base != null ? cr - base : null;
            const crStage = ratio == null ? null : ratio >= 3.0 ? 3 : ratio >= 2.0 ? 2 : ratio >= 1.5 ? 1 : delta != null && delta >= 0.3 ? 1 : 0;
            const uoStage = uo == null ? null : uo < 0.3 ? "무뇨(<0.3)" : uo < 0.5 ? "핍뇨(<0.5)" : "정상(≥0.5)";
            return (
              <div className="mt-3 rounded-lg border border-border bg-muted/30 p-3 text-[11px] leading-relaxed text-foreground">
                <p className="mb-1 font-semibold text-primary">KDIGO 해석 (AKI 폴더 02_aki_labels.sql 기준)</p>
                <ul className="flex flex-col gap-0.5">
                  <li>
                    · Cr 기준: {ratio != null ? `현재/기저 = ${ratio.toFixed(2)}배` : "기저 또는 현재값 부족"}
                    {delta != null ? ` (증가 ${delta >= 0 ? "+" : ""}${delta.toFixed(2)} mg/dL)` : ""}
                    {crStage != null && (
                      <b className={cn("ml-1", crStage >= 2 ? "text-destructive" : crStage === 1 ? "text-warning" : "text-success")}>
                        → {crStage === 0 ? "Cr 기준 비충족" : `Stage ${crStage} 기준 충족`}
                      </b>
                    )}
                  </li>
                  <li>· 소변량 기준: {uoStage != null ? <b className={cn(uo != null && uo < 0.5 ? "text-warning" : "text-success")}>{uoStage} mL/kg/h</b> : "소변량 데이터 없음"}</li>
                  <li className="text-muted-foreground">· KDIGO Stage: Cr 1.5/2.0/3.0배 또는 +0.3 mg/dL(48h) · 소변량 &lt;0.5 mL/kg/h(≥6h). 두 기준 중 높은 단계가 최종.</li>
                </ul>
              </div>
            );
          })()}
        </ReportSection>

        {/* 4. AKI 예측 요약 (Model Prediction (환자명)) */}
        <ReportSection index={4} title={`Model Prediction (${name})`}>
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="rounded-lg border border-border p-3 text-[12px]">
              <Row label="예측 단계" value={summary?.stage ?? "—"} />
              <Row label="AKI 발생확률" value={akiProb !== null ? `${Math.round(akiProb * 100)}%` : "—"} tone="text-success" />
              <Row label="중증 AKI(Stage 2-3) 확률" value={summary ? `${(summary.pStage2Plus * 100).toFixed(0)}%` : "—"} />
              <Row label="위험 점수" value={riskScore} />
            </div>
            <div className="rounded-lg border border-border p-3 text-[12px]">
              <Row
                label="실제 AKI 발생 (검증)"
                value={validation ? (validation.actualLabel === 1 ? `AKI stage ${validation.actualStage}` : "Non-AKI") : "—"}
              />
              <Row
                label="예측 적중 (AKI 여부)"
                value={
                  correct === null ? "—" : (
                    <span className={cn("inline-flex items-center gap-1", correct ? "text-success" : "text-destructive")}>
                      {correct ? <CheckCircle2 className="size-3.5" /> : <XCircle className="size-3.5" />}
                      {correct ? "적중" : "불일치"}
                    </span>
                  )
                }
              />
              <p className="mt-2 text-[10px] text-muted-foreground">실제 결과(label)는 검증 대조용입니다.</p>
            </div>
          </div>
          {riskHistory && (
            <div className="mt-3 rounded-lg border border-border p-3">
              <div className="mb-1 text-[11px] font-medium">위험도 추세</div>
              <RiskHistoryChart history={riskHistory} />
            </div>
          )}
        </ReportSection>

        {/* 5. 투석/RRT 평가 — KDIGO+ICU 규칙 기반 트리거(예측 아님) */}
        <ReportSection index={5} title="투석 / RRT 평가" subtitle="임상 트리거 · 의사결정 보조">
          {rrt ? (
            <RrtTriggerPanel assessment={rrt} />
          ) : (
            <div className="flex items-start gap-2 rounded-md border border-warning/40 bg-warning/10 px-3 py-2 text-[11px] text-foreground">
              <AlertTriangle className="mt-0.5 size-3.5 shrink-0 text-warning" />
              <span>RRT 트리거를 계산할 수 없습니다(크레아티닌 시계열 부족). 신장내과 종합판단이 필요합니다.</span>
            </div>
          )}
        </ReportSection>

        {/* 6. 추세 그래프 (환자별 개별 데이터) + KDIGO 기준선 */}
        <ReportSection index={6} title="추세 그래프" subtitle="환자별 개별 데이터 · KDIGO 기준선">
          {trends ? (
            <PatientTrendGraphs trends={trends} baseline={summary?.baselineCreatinine ?? null} />
          ) : (
            <p className="py-4 text-center text-[11px] text-muted-foreground">추세 불러오는 중…</p>
          )}
        </ReportSection>

        {/* 7. SHAP 설명 (환자별) */}
        <ReportSection index={7} title="SHAP 설명" subtitle="이 환자의 위험 기여 인자">
          {shap ? <ShapBars features={shap.features} /> : <p className="py-4 text-center text-[11px] text-muted-foreground">기여도 계산 중…</p>}
        </ReportSection>

        {/* 8. 모델 지표 (보정 곡선 · 혼동 행렬 · 위험 구간) */}
        <ReportSection index={8} title="모델 지표" subtitle="AKI 2-Stage">
          {validation?.calibration && (
            <div className="mb-3 rounded-lg border border-border p-3">
              <div className="mb-1 text-[11px] font-medium">신뢰도 곡선 (Calibration) — ECE 대체</div>
              <CalibrationPlot calibration={validation.calibration} />
            </div>
          )}
          {metrics && (
            <div className="mb-3 grid gap-3 sm:grid-cols-2">
              <div className="rounded-lg border border-border p-3">
                <div className="mb-2 text-[11px] font-medium">혼동 행렬 (Stage1)</div>
                <ConfusionMatrix stage={metrics.stage1} />
              </div>
              <div className="rounded-lg border border-border p-3">
                <div className="mb-2 text-[11px] font-medium">위험 구간 (Stratification)</div>
                <RiskStratification score={riskScore} />
              </div>
            </div>
          )}
          {metrics ? (
            <ValidationReportBody metrics={metrics} />
          ) : (
            <p className="text-[11px] text-muted-foreground">모델 검증 리포트를 불러오지 못했습니다.</p>
          )}
        </ReportSection>
      </DialogContent>
    </Dialog>
  );
}
