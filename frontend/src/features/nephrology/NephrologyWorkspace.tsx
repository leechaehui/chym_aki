import { useEffect, useState } from "react";
import { CheckCircle2 } from "lucide-react";
import { useSearchParams } from "react-router-dom";
import type { IcuMonitorSummary, Patient, ShapFeature } from "@/types";
import { patientService } from "@/services/patientService";
import { icuMonitorService } from "@/services/icuMonitorService";
import { PageHeader } from "@/components/common/PageHeader";
import { StatCard } from "@/components/common/StatCard";
import { LoadingSpinner } from "@/components/common/LoadingSpinner";
import { EmptyState } from "@/components/common/EmptyState";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/cn";
import { MetricTrendChart, type MetricPoint } from "./MetricTrendChart";
import { AiRiskGauge } from "./AiRiskGauge";
import { PatientSummaryBar } from "./PatientSummaryBar";
import { LabResultReport } from "./LabResultReport";
import { riskBandFor, RISK_BADGE_TONE } from "@/lib/riskLabel";
import { ShapImportanceChart } from "./ShapImportanceChart";
import { IcuPatientSearch } from "./IcuPatientSearch";
import { ConsultRequestModal } from "@/features/consult/ConsultRequestModal";
import { ConsultStatusCard } from "@/features/consult/ConsultStatusCard";
import { StatusBadge } from "@/components/common/StatusBadge";

/** 지표별 추이 카드 — 제목 + 미니 영역 차트. */
function TrendCard({
  title,
  data,
  color,
  unit,
  refLine,
  refLabel,
  refColor,
}: {
  title: string;
  data: MetricPoint[];
  color: string;
  unit?: string;
  refLine?: number;
  refLabel?: string;
  refColor?: string;
}) {
  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-2">
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent className="flex-1 min-h-0 p-3">
        <MetricTrendChart data={data} color={color} unit={unit} refLine={refLine} refLabel={refLabel} refColor={refColor} height={270} />
      </CardContent>
    </Card>
  );
}

/** 모델의 실제 예측 stage 문자열 → 정렬용 등급 순위(ICU AKI 모니터링의 [pred, risk_score] 정렬과 동일 기준). */
function predRank(stage?: string): number {
  const s = (stage ?? "").toLowerCase();
  if (s.includes("stage 2-3") || s.includes("stage2") || s.includes("stage 3")) return 2;
  if (s.includes("stage 1") || s.includes("stage1")) return 1;
  if (s.includes("non-aki")) return 0;
  return -1; // predictedStage 없음(구데이터 등) — 맨 뒤로
}

/** 환자 목록 정렬 — 1순위 등급(pred), 2순위 위험점수, 동점이면 최근 입원(admittedAt)이 위로.
 * "ICU AKI 모니터링"(IcuMonitorService.list_patients)과 동일한 [pred, risk_score] 기준이라야
 * 두 화면에서 같은 환자가 같은 순서·같은 등급으로 보인다. */
function sortByRisk(list: Patient[]): Patient[] {
  return [...list].sort((a, b) => {
    const byPred = predRank(b.predictedStage) - predRank(a.predictedStage);
    if (byPred !== 0) return byPred;
    const byScore = b.aiRiskScore - a.aiRiskScore;
    if (byScore !== 0) return byScore;
    return a.admittedAt < b.admittedAt ? 1 : a.admittedAt > b.admittedAt ? -1 : 0;
  });
}

/** 목록에는 주의(moderate) 이상만 노출 — 안정군까지 다 보여주면 정작 봐야 할 환자가 묻힘.
 * 등급은 점수가 아니라 predictedStage(모델의 실제 예측) 기준 — ICU AKI 모니터링과 동일 소스. */
function needsAttention(p: Patient): boolean {
  return riskBandFor(p.aiRiskScore, p.predictedStage).tier !== "low";
}

/**
 * 신장내과 대시보드 — AKI 현황(KPI) + 환자 관리(목록·상세·추이·AI 위험도)가 핵심.
 * 협진은 메인 핵심 영역에서 내려 하단 보조 패널(ConsultPanel)로 배치(상세는 협진 센터).
 */
export function NephrologyWorkspace() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [patients, setPatients] = useState<Patient[]>([]);
  const [summary, setSummary] = useState<IcuMonitorSummary | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [consultOpen, setConsultOpen] = useState(false);
  const [shapFeatures, setShapFeatures] = useState<readonly ShapFeature[]>([]);
  const [predictionStage, setPredictionStage] = useState<string | null>(null);
  const [predictionScore, setPredictionScore] = useState<number | null>(null);

  useEffect(() => {
    patientService.list()
      .then((p) => {
        const sorted = sortByRisk(p);
        setPatients(sorted);
        setSelectedId(sorted.find(needsAttention)?.id ?? null);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
    // KPI 는 전체 코호트 요약(모델 분류 + AI 위험밴드) — 목록(상위 위험)과 별개.
    icuMonitorService.summary().then(setSummary).catch(() => {});
  }, []);

  useEffect(() => {
    // 데모 시뮬레이터 완료 시 이 화면은 React Query를 구독하지 않으므로 전역 이벤트로 직접 재조회한다.
    const onDemoRefresh = () => {
      patientService.list().then((p) => {
        const sorted = sortByRisk(p);
        setPatients(sorted);
        setSelectedId((prev) => {
          // 이미 주의 이상인 환자를 보고 있었다면 유지, 아니면 상위 위험 환자로 전환.
          if (prev && sorted.find((x) => x.id === prev && needsAttention(x))) return prev;
          return sorted.find(needsAttention)?.id ?? null;
        });
      }).catch(() => {});
      icuMonitorService.summary().then(setSummary).catch(() => {});
    };
    window.addEventListener("chym:demo-refresh", onDemoRefresh);
    return () => window.removeEventListener("chym:demo-refresh", onDemoRefresh);
  }, []);

  const attentionPatients = patients.filter(needsAttention);

  // 알림 딥링크(?patient=<mrn>) → 해당 환자를 선택한다. 적용 후 파라미터 소비(URL 정리).
  useEffect(() => {
    const mrn = searchParams.get("patient");
    if (!mrn || patients.length === 0) return;
    const match = patients.find((p) => p.mrn === mrn);
    if (match) setSelectedId(match.id);
    searchParams.delete("patient");
    setSearchParams(searchParams, { replace: true });
  }, [searchParams, patients, setSearchParams]);

  const patient = patients.find((p) => p.id === selectedId) ?? null;

  useEffect(() => {
    if (!patient) {
      setShapFeatures([]);
      setPredictionStage(null);
      setPredictionScore(null);
      return;
    }

    const idSrc = patient as { stayId?: number; id?: string; mrn?: string };
    const cleanId = String(idSrc.stayId ?? "").replace(/[^\d]/g, "") || String(idSrc.id ?? "").replace(/[^\d]/g, "") || String(idSrc.mrn ?? "").replace(/[^\d]/g, "");
    const stayId = Number(cleanId);
    if (!Number.isFinite(stayId) || stayId === 0) {
      setShapFeatures([]);
      setPredictionStage(null);
      setPredictionScore(null);
      return;
    }

    icuMonitorService.shap(stayId).then((res) => setShapFeatures(res.features)).catch(() => setShapFeatures([]));
    icuMonitorService.validation(stayId).then((res) => {
      setPredictionStage(res.predictedStage);
      setPredictionScore(res.riskScore);
    }).catch(() => {
      setPredictionStage(null);
      setPredictionScore(null);
    });
  }, [patient]);

  if (loading) return <LoadingSpinner label="환자 목록 불러오는 중" />;

  // /validation 은 숫자 stay_id 기반이라 데모 환자(id가 "p-{subjectId}" 형식)는 항상 조회에 실패해
  // predictionStage 가 null로 남는다 — 그럴 땐 목록에서 이미 받아온 patient.predictedStage로 대체해
  // 위 목록 뱃지와 상세 패널(요약바/게이지/SHAP)이 서로 다른 등급을 보여주지 않게 한다.
  const effectivePredictedStage = predictionStage ?? patient?.predictedStage ?? null;

  return (
    <div className="w-full">
      <PageHeader title="신장내과 대시보드" subtitle="AKI 현황 · 환자 관리 · 위험도 분석" />

      {/* ICU 코호트 환자 검색 — Stay/Subject ID·병동 조회 → Quick View */}
      <IcuPatientSearch className="mb-4" />

      {/* 상단 KPI — 전체 ICU 코호트 기준(목록은 상위 위험 worklist). */}
      {/* (b) 모델 예측 분류 — Non-AKI / Stage 1 / Stage 2-3 (3-클래스) */}
      <div className="mb-1 text-[11px] font-semibold text-muted-foreground">모델 예측 분류 (전체 코호트)</div>
      <div className="mb-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatCard label="ICU 환자" value={summary?.total ?? "—"} unit="명" tone="primary" />
        <StatCard label="Non-AKI" value={summary?.low ?? "—"} unit="명" tone="success" />
        <StatCard label="Stage 1" value={summary?.moderate ?? "—"} unit="명" tone="warning" />
        <StatCard label="Stage 2-3" value={summary?.high ?? "—"} unit="명" tone="danger" />
      </div>


      {/* 중앙: 환자 목록(좌) + 환자 상세(우) */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[260px_1fr]">
        {/* 좌측 환자 목록 레일 */}
        <Card className="h-fit lg:sticky lg:top-0">
          <CardHeader>
            <CardTitle>환자 목록 <span className="text-[11px] font-normal text-muted-foreground">(주의 이상 {attentionPatients.length}명)</span></CardTitle>
          </CardHeader>
          <CardContent className="p-2">
            {attentionPatients.length === 0 && (
              <div className="px-2 py-6 text-center text-xs text-muted-foreground">
                주의·고위험 환자 없음
              </div>
            )}
            {attentionPatients.map((p) => {
              // 배지는 predictedStage(모델의 실제 예측) 기준 — ICU AKI 모니터링과 동일 소스라
              // 두 화면에서 같은 환자가 같은 등급으로 보인다.
              const band = riskBandFor(p.aiRiskScore, p.predictedStage);
              const isSelected = p.id === selectedId;
              
              // 선택 시 위험도별 테두리 및 배경 강조 (고위험: 적색, 주의: 주황색)
              const borderHighlight = isSelected
                ? band.tier === "high" || p.predictedStage === "AKI Stage 2-3"
                  ? "border-red-500 border-2 bg-red-50/60 dark:bg-red-950/20 shadow-md scale-[1.01] ring-1 ring-red-500/30"
                  : "border-orange-500 border-2 bg-orange-50/60 dark:bg-orange-950/20 shadow-md scale-[1.01] ring-1 ring-orange-500/30"
                : "border-border/50 hover:bg-muted/60";

              return (
                <button
                  key={p.id}
                  onClick={() => setSelectedId(p.id)}
                  className={cn(
                    "mb-1.5 flex w-full flex-col gap-1 rounded-md border px-3 py-2.5 text-left transition-all duration-200 relative",
                    borderHighlight
                  )}
                >
                  {/* 선택된 환자 카드 왼쪽에 액센트 바 연출 */}
                  {isSelected && (
                    <div className={cn(
                      "absolute left-0 top-0 bottom-0 w-1 rounded-l-md",
                      band.tier === "high" || p.predictedStage === "AKI Stage 2-3" ? "bg-red-500" : "bg-orange-500"
                    )} />
                  )}
                  <span className="flex items-center justify-between gap-1 pl-1">
                    <span className="text-sm font-semibold text-foreground">
                      {p.name} <span className="text-[11px] font-normal text-muted-foreground">{p.sex === "M" ? "남" : "여"}/{p.age}</span>
                    </span>
                    <StatusBadge label={band.label} tone={RISK_BADGE_TONE[band.tier]} />
                  </span>
                  <span className="truncate text-[11px] text-muted-foreground pl-1">{p.diagnosis}</span>
                  <span className="text-[10px] text-muted-foreground/70 pl-1">
                    {p.mrn}
                  </span>
                </button>
              );
            })}
          </CardContent>
        </Card>

        {/* 우측 메인 — 환자 상세 */}
        {patient ? (
          <div className="flex flex-col gap-4">
            {/* ① 환자 핵심 요약(식별 + AKI 단계/위험 + 핵심검사 + 병리 협진 요청) */}
            <PatientSummaryBar patient={patient} predictedStage={effectivePredictedStage} onConsult={() => setConsultOpen(true)} />

            {/* ② 추이 차트 + AKI Prediction Score */}
            <div className="grid gap-4 xl:grid-cols-4 lg:grid-cols-2 auto-rows-fr">
              {(() => {
                const base = patient.trend[0]?.creatinine;
                const crRef = base ? Math.round(Math.min(base * 1.5, base + 0.3) * 100) / 100 : undefined;
                const crLast = patient.trend[patient.trend.length - 1]?.creatinine;
                const crColor = crRef == null ? "#374151"
                  : crLast != null && crLast >= crRef ? "#dc2626"
                  : patient.trend.some((t) => t.creatinine >= crRef) ? "#f97316"
                  : "#374151";
                return (
                  <TrendCard
                    title="Cr Trend"
                    data={patient.trend.map((t) => ({ date: t.date, value: t.creatinine }))}
                    color={crColor}
                    unit="mg/dL"
                    refLine={crRef}
                    refLabel={crRef !== undefined ? `AKI ${crRef}` : undefined}
                  />
                );
              })()}
              {(() => {
                const uoLast = patient.urineOutput[patient.urineOutput.length - 1]?.value;
                const uoColor = uoLast != null && uoLast < 0.5 ? "#dc2626"
                  : patient.urineOutput.some((u) => u.value < 0.5) ? "#f97316"
                  : "#374151";
                return (
                  <TrendCard
                    title="소변량 추이"
                    data={patient.urineOutput.map((u) => ({ date: u.date, value: u.value }))}
                    color={uoColor}
                    unit="mL/kg/hr"
                    refLine={0.5}
                    refLabel="핍뇨 0.5"
                  />
                );
              })()}
              <Card className="h-full flex flex-col">
                <CardHeader className="pb-2">
                  <CardTitle>AI 예측 주요 변수 (SHAP Top 5)</CardTitle>
                </CardHeader>
                <CardContent className="flex-1 min-h-0 pt-1">
                  <ShapImportanceChart features={shapFeatures} height={300} />
                </CardContent>
              </Card>
              <Card className="h-full flex flex-col">
                <CardHeader className="pb-2">
                  <CardTitle>AKI Prediction Score</CardTitle>
                </CardHeader>
                <CardContent className="flex-1 pt-1">
                  <AiRiskGauge score={predictionScore ?? patient.aiRiskScore} predictedStage={effectivePredictedStage} />
                </CardContent>
              </Card>
            </div>

            {/* ③ SHAP Top5 기반 검사 결과 — 전체 너비 */}
            <Card>
              <CardHeader>
                <CardTitle>SHAP Top5 변수</CardTitle>
              </CardHeader>
              <CardContent>
                <LabResultReport
                  patient={patient}
                  predictedStage={effectivePredictedStage}
                  predictionScore={predictionScore ?? patient.aiRiskScore}
                />
              </CardContent>
            </Card>

            {/* ④ 병리 협진 현황 — 요청한 병리 판독이 완료되면 소견·진단 회신을 이곳에서 받는다 */}
            <ConsultStatusCard patientMrn={patient.mrn} />

            <ConsultRequestModal patient={patient} open={consultOpen} onOpenChange={setConsultOpen} />
          </div>
        ) : (
          <EmptyState
            icon={CheckCircle2}
            title="전체 환자 안정적입니다"
            description="주의·고위험 등급의 환자가 없습니다. 아래는 전체 코호트 기본 통계입니다."
            action={
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <StatCard label="ICU 환자" value={summary?.total ?? "—"} unit="명" tone="primary" />
                <StatCard label="Non-AKI" value={summary?.low ?? "—"} unit="명" tone="success" />
                <StatCard label="Stage 1" value={summary?.moderate ?? "—"} unit="명" tone="warning" />
                <StatCard label="Stage 2-3" value={summary?.high ?? "—"} unit="명" tone="danger" />
              </div>
            }
          />
        )}
      </div>

      
    </div>
  );
}
