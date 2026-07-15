import { useEffect, useState } from "react";
import { HeartPulse, Activity, AlertTriangle, Users, Radio, Info } from "lucide-react";
import type { AlertRecord, IcuMonitorSummary } from "@/types";
import { icuMonitorService } from "@/services/icuMonitorService";
import { alertService } from "@/services/alertService";
import { PageHeader } from "@/components/common/PageHeader";
import { StatCard } from "@/components/common/StatCard";
import { StatusBadge } from "@/components/common/StatusBadge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { SYNTHETIC_NAME_NOTICE } from "@/lib/fakeName";
import { IcuPatientSearch } from "./IcuPatientSearch";
import { IcuRiskCohort } from "./IcuRiskCohort";
import { ColumnToggle, useColumnVisibility, type ColumnDef } from "./ColumnToggle";

// CDSS 알림 테이블 컬럼(키는 localStorage 보존 키).
const ALERT_COLUMNS: ColumnDef[] = [
  { key: "type", label: "유형" },
  { key: "patient", label: "환자", locked: true },
  { key: "priority", label: "우선순위" },
  { key: "score", label: "위험점수" },
  { key: "message", label: "근거" },
  { key: "status", label: "상태" },
];

const ALERT_TONE: Record<string, "danger" | "warning" | "success"> = {
  critical: "danger",
  warning: "warning",
  info: "success",
};

const STATUS_LABEL: Record<string, string> = {
  active: "신규",
  viewed: "확인중",
  acknowledged: "조치확인",
  dismissed: "해제",
  escalated: "에스컬레이션",
};

/**
 * ICU AKI 모니터링 — 실제 MIMIC-IV ICU 코호트에 학습 모델(LR→LGBM)을 적용한 위험순 목록.
 * 위험순 목록 자체는 공유 컴포넌트(IcuRiskCohort)로 분리 — 신장내과 대시보드와 동일 동작.
 * 환자 이름은 비식별 데이터에 부여한 합성 표시명이다(SYNTHETIC_NAME_NOTICE).
 */
export function IcuAkiMonitor() {
  const [summary, setSummary] = useState<IcuMonitorSummary | null>(null);
  const [alerts, setAlerts] = useState<AlertRecord[]>([]);
  const [simulating, setSimulating] = useState(false);
  // 값이 바뀌면 IcuRiskCohort 가 현재 페이지 목록을 다시 불러온다(CDSS 시뮬 후 갱신).
  const [refreshKey, setRefreshKey] = useState(0);

  const alertCols = useColumnVisibility("chym.cols.cdssAlerts", ALERT_COLUMNS);

  useEffect(() => {
    icuMonitorService.summary().then(setSummary).catch(() => {});
    void loadAlerts();
  }, []);

  function loadAlerts() {
    return alertService.list({ limit: 50 }).then(setAlerts).catch(() => {});
  }

  // CDSS 파이프라인 시뮬: LAB_EVENT 발행 → 엔진/Alert 처리 → WS 실시간 알림.
  // 파이프라인 구동 후 알림 + 위험순 목록(refreshKey)을 갱신한다.
  async function onSimulate() {
    setSimulating(true);
    try {
      await alertService.simulateIcu(8);
      await loadAlerts();
      setRefreshKey((k) => k + 1);
    } finally {
      setSimulating(false);
    }
  }

  return (
    <div className="mx-auto max-w-[1200px]">
      <PageHeader
        icon={HeartPulse}
        title="ICU AKI 모니터링"
        subtitle="실제 MIMIC-IV ICU 코호트 · 학습 모델(2-stage LR→LGBM) 예측 · 위험순"
      />

      {/* 합성 데이터 고지 — 이 화면의 환자/이름은 가짜임을 분명히 한다. */}
      <div className="mb-3 flex items-start gap-2 rounded-md border border-primary/30 bg-primary/5 px-3 py-2 text-[11px] text-foreground">
        <Info className="mt-0.5 size-3.5 shrink-0 text-primary" />
        <span>{SYNTHETIC_NAME_NOTICE}</span>
      </div>

      {/* 요약 KPI */}
      <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-5">
        <StatCard label="ICU 환자" value={summary?.total ?? "—"} unit="명" icon={Users} tone="primary" />
        <StatCard label="고위험" value={summary?.high ?? "—"} unit="명" icon={AlertTriangle} tone="danger" hint="Stage 2-3 예측" />
        <StatCard label="중등도" value={summary?.moderate ?? "—"} unit="명" icon={Activity} tone="warning" />
        <StatCard label="안정" value={summary?.low ?? "—"} unit="명" icon={Activity} tone="success" />
        <StatCard label="ICU 병동" value={summary?.nCareunits ?? "—"} unit="개" icon={HeartPulse} tone="accent" />
      </div>

      {/* 환자 검색 — 이름·Stay/Subject ID·병동 조회([검색]/[취소] 버튼) */}
      <IcuPatientSearch className="mb-4" />

      {/* CDSS 실시간 알림 파이프라인 (이벤트 기반: LAB→AKI→ALERT→WS) */}
      <Card className="mb-4">
        <CardHeader className="flex-row flex-wrap items-center justify-between gap-2">
          <CardTitle className="flex items-center gap-2">
            <Radio className="size-4 text-primary" /> CDSS 실시간 알림 파이프라인
          </CardTitle>
          <div className="flex items-center gap-2">
            <ColumnToggle columns={ALERT_COLUMNS} {...alertCols} />
            <Button size="sm" onClick={onSimulate} disabled={simulating}>
              {simulating ? "스트림 처리 중…" : "ICU 스트림 시뮬레이션"}
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <p className="mb-2 text-[11px] text-muted-foreground">
            LAB_EVENT → AKI 엔진(baseline 3계층·safety·guardrail) → Alert(dedup·우선순위) → WebSocket 푸시.
            지표 변화 시 백그라운드로 위험순 목록이 갱신됩니다(별도 컬럼 없이 값만 반영).
          </p>
          {alerts.length === 0 ? (
            <p className="py-3 text-center text-xs text-muted-foreground">
              아직 alert 가 없습니다. "ICU 스트림 시뮬레이션"을 눌러 파이프라인을 구동하세요.
            </p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[560px] text-sm">
                <thead>
                  <tr className="border-b border-border text-left text-[11px] text-muted-foreground">
                    {alertCols.isVisible("type") && <th className="px-2 py-2 font-medium">유형</th>}
                    {alertCols.isVisible("patient") && <th className="px-2 py-2 font-medium">환자</th>}
                    {alertCols.isVisible("priority") && <th className="px-2 py-2 font-medium">우선순위</th>}
                    {alertCols.isVisible("score") && <th className="px-2 py-2 font-medium">위험점수</th>}
                    {alertCols.isVisible("message") && <th className="px-2 py-2 font-medium">근거</th>}
                    {alertCols.isVisible("status") && <th className="px-2 py-2 font-medium">상태</th>}
                  </tr>
                </thead>
                <tbody>
                  {alerts.map((a) => (
                    <tr key={a.id} className="border-b border-border/50 hover:bg-muted/40">
                      {alertCols.isVisible("type") && (
                        <td className="px-2 py-2">
                          <StatusBadge label={a.type} tone={ALERT_TONE[a.severity] ?? "warning"} />
                        </td>
                      )}
                      {alertCols.isVisible("patient") && (
                        <td className="px-2 py-2 whitespace-nowrap tabular-nums text-muted-foreground">{a.patientId}</td>
                      )}
                      {alertCols.isVisible("priority") && <td className="px-2 py-2 tabular-nums">{a.priority}</td>}
                      {alertCols.isVisible("score") && (
                        <td className="px-2 py-2 tabular-nums">{a.akiScore != null ? a.akiScore.toFixed(2) : "—"}</td>
                      )}
                      {alertCols.isVisible("message") && (
                        <td className="px-2 py-2 text-[12px] text-muted-foreground" title={a.message}>
                          <span className="line-clamp-1 max-w-[280px]">{a.message}</span>
                        </td>
                      )}
                      {alertCols.isVisible("status") && (
                        <td className="px-2 py-2 whitespace-nowrap text-[12px]">{STATUS_LABEL[a.status] ?? a.status}</td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      {/* 위험순 환자 목록(공유 컴포넌트) — 병동 필터 + Stay ID 컬럼 + Quick View */}
      <IcuRiskCohort refreshKey={refreshKey} />
    </div>
  );
}
