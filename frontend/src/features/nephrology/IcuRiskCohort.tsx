import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { RefreshCw } from "lucide-react";
import type {
  CareunitStat,
  IcuAkiPatient,
  IcuMonitorSummary,
  ModelMetrics,
  PatientIdentity,
} from "@/types";
import { icuMonitorService } from "@/services/icuMonitorService";
import { LoadingSpinner } from "@/components/common/LoadingSpinner";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/cn";
import { fakeKoreanName } from "@/lib/fakeName";
import { riskBandFor } from "@/lib/riskLabel";
import { PatientReportModal } from "./PatientReportModal";
import { PatientQuickView } from "./PatientQuickView";

const PAGE_SIZE = 20;

const TEXT = {
  defaultTitle: "위험순 환자 목록",
  all: "전체",
  allUnits: "전체 병동",
  retry: "다시 시도",
  refresh: "새로고침",
  loadFailed: "목록을 불러오지 못했습니다.",
  loading: "ICU 코호트 예측 불러오는 중",
  rank: "순위",
  patient: "환자",
  ageGender: "나이/성별",
  unit: "병동",
  stage: "예측 단계",
  risk: "위험도",
  probability: "AKI 확률",
  report: "검증 리포트",
  newTitle: "새로 추가된 환자",
  newHelp: "● 파란 점은 새로 추가된 환자입니다. (최근 24시간)",
  disclaimer: "행을 클릭하면 환자 Quick View가 열립니다. 학습 모델은 보조 지표로, 단독 임상판단에 사용하지 않습니다.",
};

function shortUnit(u: string): string {
  const m = u.match(/\(([^)]+)\)/);
  return m ? m[1] : u;
}

function RiskCell({ score, stage }: { score: number; stage?: string }) {
  const band = riskBandFor(score, stage);
  return (
    <span className={cn("inline-flex items-center gap-1 font-semibold tabular-nums", band.textClass)}>
      <span>{band.dot}</span>
      <span>{band.label}</span>
      <span>{score}</span>
    </span>
  );
}

export function IcuRiskCohort({
  refreshKey = 0,
  title = TEXT.defaultTitle,
}: {
  refreshKey?: number;
  title?: string;
}) {
  const [searchParams, setSearchParams] = useSearchParams();
  const [summary, setSummary] = useState<IcuMonitorSummary | null>(null);
  const [units, setUnits] = useState<CareunitStat[]>([]);
  const [patients, setPatients] = useState<IcuAkiPatient[]>([]);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [careunit, setCareunit] = useState("");
  const [page, setPage] = useState(0);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [quickViewPatient, setQuickViewPatient] = useState<PatientIdentity | null>(null);

  // 알림 바로가기(?patient=AKI-xxxx)를 누르고 들어오면 해당 환자를 자동 선택하여 Quick View 활성화
  useEffect(() => {
    const mrn = searchParams.get("patient");
    if (!mrn || patients.length === 0) return;
    const sid = Number(mrn.replace(/[^\d]/g, ""));
    if (!Number.isFinite(sid)) return;

    const match = patients.find((p) => p.subjectId === sid || p.stayId === sid);
    if (match) {
      setQuickViewPatient(match);
      // 파라미터를 소멸시켜 중복 트리거 방지
      searchParams.delete("patient");
      setSearchParams(searchParams, { replace: true });
    }
  }, [searchParams, patients, setSearchParams]);

  useEffect(() => {
    Promise.all([icuMonitorService.summary(), icuMonitorService.careunits(), icuMonitorService.modelMetrics()])
      .then(([s, u, m]) => {
        setSummary(s);
        setUnits(u);
        setMetrics(m);
      })
      .catch((e: unknown) => setLoadError(e instanceof Error ? e.message : TEXT.loadFailed));
  }, []);

  function loadPatients() {
    setLoading(true);
    setLoadError(null);
    return icuMonitorService
      .list({ limit: PAGE_SIZE, offset: page * PAGE_SIZE, careunit: careunit || undefined })
      .then((p) => {
        setPatients(p);
        setLoading(false);
      })
      .catch((e: unknown) => {
        setLoadError(e instanceof Error ? e.message : TEXT.loadFailed);
        setLoading(false);
      });
  }

  async function refreshPredictions() {
    setRefreshing(true);
    setLoadError(null);
    try {
      await icuMonitorService.refreshPredictions();
      await loadPatients();
    } catch (e: unknown) {
      setLoadError(e instanceof Error ? e.message : "AI predictions could not be refreshed.");
    } finally {
      setRefreshing(false);
    }
  }

  useEffect(() => {
    void loadPatients();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [careunit, page, refreshKey]);

  function selectCareunit(next: string) {
    setCareunit(next);
    setPage(0);
  }

  const totalForFilter = careunit ? units.find((u) => u.careunit === careunit)?.n ?? 0 : summary?.total ?? 0;
  const totalPages = Math.max(1, Math.ceil(totalForFilter / PAGE_SIZE));
  const winStart = Math.max(0, Math.min(page - 2, totalPages - 5));
  const pageWindow = Array.from({ length: Math.min(5, totalPages) }, (_, i) => winStart + i);

  return (
    <>
      <div className="mb-3 flex flex-wrap gap-1.5">
        <button
          onClick={() => selectCareunit("")}
          className={cn(
            "rounded-full px-3 py-1 text-[11px] font-medium",
            careunit === "" ? "bg-primary text-primary-foreground" : "bg-secondary text-muted-foreground",
          )}
        >
          {TEXT.all} {summary ? <span className="opacity-70">{summary.total.toLocaleString()}</span> : null}
        </button>
        {units.map((u) => (
          <button
            key={u.careunit}
            onClick={() => selectCareunit(u.careunit)}
            className={cn(
              "rounded-full px-3 py-1 text-[11px] font-medium",
              careunit === u.careunit ? "bg-primary text-primary-foreground" : "bg-secondary text-muted-foreground",
            )}
            title={`${u.careunit} - ${u.n}`}
          >
            {shortUnit(u.careunit)} <span className="opacity-70">{u.n}</span>
          </button>
        ))}
      </div>

      <Card>
        <CardHeader className="flex-row flex-wrap items-center justify-between gap-2">
          <CardTitle>
            {title} - {careunit ? shortUnit(careunit) : TEXT.allUnits}
          </CardTitle>
          <Button size="sm" variant="outline" onClick={refreshPredictions} disabled={refreshing}>
            <RefreshCw className={cn(refreshing && "animate-spin")} />
            {TEXT.refresh}
          </Button>
        </CardHeader>
        <CardContent>
          {loadError ? (
            <div className="rounded-md border border-danger/30 bg-danger/5 px-3 py-4 text-center text-xs text-danger">
              <p className="font-medium">{TEXT.loadFailed}</p>
              <p className="mt-1 text-[11px] text-danger/80">{loadError}</p>
              <Button size="sm" variant="outline" className="mt-2" onClick={() => loadPatients()}>
                {TEXT.retry}
              </Button>
            </div>
          ) : loading ? (
            <LoadingSpinner label={TEXT.loading} />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[820px] text-sm">
                <thead>
                  <tr className="border-b border-border text-left text-[11px] text-muted-foreground">
                    <th className="px-2 py-2 font-medium">{TEXT.rank}</th>
                    <th className="px-2 py-2 font-medium">Subject ID</th>
                    <th className="px-2 py-2 font-medium">{TEXT.patient}</th>
                    <th className="px-2 py-2 font-medium">{TEXT.ageGender}</th>
                    <th className="px-2 py-2 font-medium">{TEXT.unit}</th>
                    <th className="px-2 py-2 font-medium">{TEXT.stage}</th>
                    <th className="px-2 py-2 font-medium">{TEXT.risk}</th>
                    <th className="px-2 py-2 font-medium">{TEXT.probability}</th>
                    <th className="px-2 py-2 font-medium">{TEXT.report}</th>
                  </tr>
                </thead>
                <tbody>
                  {patients.map((p, index) => {
                    const isSelected = quickViewPatient !== null && (quickViewPatient.stayId === p.stayId || quickViewPatient.subjectId === p.subjectId);
                    const band = riskBandFor(p.riskScore, p.stage);
                    
                    // 선택/딥링크 환자 행 고대비 하이라이트 (고위험: 적색, 주의: 주황색)
                    const rowHighlight = isSelected
                      ? band.tier === "high" || p.stage === "AKI Stage 2-3"
                        ? "bg-red-50/90 dark:bg-red-950/30 font-semibold border-y-2 border-red-400/80 shadow-sm text-red-900 dark:text-red-200"
                        : "bg-orange-50/90 dark:bg-orange-950/30 font-semibold border-y-2 border-orange-400/80 shadow-sm text-orange-950 dark:text-orange-200"
                      : "hover:bg-muted/40";

                    return (
                      <tr
                        key={p.stayId}
                        onClick={() => setQuickViewPatient(p)}
                        className={cn("cursor-pointer border-b border-border/50 transition-all duration-200", rowHighlight)}
                      >
                        <td className="px-2 py-3.5 whitespace-nowrap tabular-nums text-[12px]">{page * PAGE_SIZE + index + 1}</td>
                        <td className="px-2 py-3.5 whitespace-nowrap tabular-nums text-[12px]">{p.subjectId}</td>
                        <td className="px-2 py-3.5 whitespace-nowrap font-semibold">
                          <span>{fakeKoreanName(p.subjectId)}</span>
                          {p.isNewPatient && <span className="ml-1 text-primary" title={TEXT.newTitle}>{"●"}</span>}
                        </td>
                        <td className="px-2 py-3.5 whitespace-nowrap">{p.age ?? "-"} / {p.gender}</td>
                        <td className="px-2 py-3.5 text-[12px]" title={p.careunit}>{shortUnit(p.careunit)}</td>
                        <td className="px-2 py-3.5 whitespace-nowrap text-[12px]">{p.stage}</td>
                        <td className="px-2 py-3.5"><RiskCell score={p.riskScore} stage={p.stage} /></td>
                        <td className="px-2 py-3.5 tabular-nums text-muted-foreground">
                          {(Math.max(p.pStage1, p.pStage2Plus) * 100).toFixed(0)}%
                        </td>
                        <td className="px-2 py-3.5" onClick={(e) => e.stopPropagation()}>
                          <PatientReportModal patient={p} metrics={metrics} />
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {!loadError && !loading && totalPages > 1 && (
            <div className="mt-3 flex flex-col items-center gap-2">
              <div className="flex items-center gap-1">
                <button onClick={() => setPage(0)} disabled={page === 0} className="rounded-md bg-secondary px-2 py-1 text-[11px] font-medium text-foreground hover:bg-primary hover:text-primary-foreground disabled:opacity-40">{"<<"}</button>
                <button onClick={() => setPage(page - 1)} disabled={page === 0} className="rounded-md bg-secondary px-2 py-1 text-[11px] font-medium text-foreground hover:bg-primary hover:text-primary-foreground disabled:opacity-40">{"<"}</button>
                {pageWindow.map((pnum) => (
                  <button
                    key={pnum}
                    onClick={() => setPage(pnum)}
                    className={cn(
                      "min-w-[28px] rounded-md px-2 py-1 text-[11px] font-medium",
                      pnum === page ? "bg-primary text-primary-foreground" : "bg-secondary text-foreground hover:bg-primary hover:text-primary-foreground",
                    )}
                  >
                    {pnum + 1}
                  </button>
                ))}
                <button onClick={() => setPage(page + 1)} disabled={page >= totalPages - 1} className="rounded-md bg-secondary px-2 py-1 text-[11px] font-medium text-foreground hover:bg-primary hover:text-primary-foreground disabled:opacity-40">{">"}</button>
                <button onClick={() => setPage(totalPages - 1)} disabled={page >= totalPages - 1} className="rounded-md bg-secondary px-2 py-1 text-[11px] font-medium text-foreground hover:bg-primary hover:text-primary-foreground disabled:opacity-40">{">>"}</button>
              </div>
              <span className="text-[11px] text-muted-foreground">
                {totalForFilter.toLocaleString()} / {(page * PAGE_SIZE + 1).toLocaleString()}-
                {Math.min((page + 1) * PAGE_SIZE, totalForFilter).toLocaleString()}
              </span>
            </div>
          )}

          <p className="mt-3 text-[11px] text-muted-foreground">{TEXT.newHelp}</p>
          <p className="mt-1 text-[11px] text-muted-foreground">{TEXT.disclaimer}</p>
        </CardContent>
      </Card>

      <PatientQuickView patient={quickViewPatient} metrics={metrics} onClose={() => setQuickViewPatient(null)} />
    </>
  );
}
