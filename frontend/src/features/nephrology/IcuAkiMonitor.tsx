import { useEffect, useState } from "react";
import { Info } from "lucide-react";
import type { IcuMonitorSummary } from "@/types";
import { icuMonitorService } from "@/services/icuMonitorService";
import { alertService } from "@/services/alertService";
import { PageHeader } from "@/components/common/PageHeader";
import { StatCard } from "@/components/common/StatCard";
import { SYNTHETIC_NAME_NOTICE } from "@/lib/fakeName";
import { IcuPatientSearch } from "./IcuPatientSearch";
import { ClinicalPatientList } from "./ClinicalPatientList";
import { AlarmCenter } from "./AlarmCenter";
import { ConsultationStatus } from "./ConsultationStatus";

/**
 * ICU AKI 모니터링 대시보드 — 응급의학과 + 신장내과 공유
 * 70:30 화면 분할을 통해 좌측엔 환자 리스트, 우측엔 알람/협진 센터를 배치합니다.
 */
export function IcuAkiMonitor() {
  const [summary, setSummary] = useState<IcuMonitorSummary | null>(null);
  // 값이 바뀌면 ClinicalPatientList가 현재 페이지 목록을 다시 불러온다(CDSS 시뮬 후 갱신).
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    icuMonitorService.summary().then(setSummary).catch(() => {});
  }, []);

  useEffect(() => {
    // 데모 시뮬레이터(설정/시간경과/트리거) 완료 시 이 화면은 React Query를 구독하지 않으므로
    // 전역 이벤트로 직접 재조회한다.
    const onDemoRefresh = () => {
      icuMonitorService.summary().then(setSummary).catch(() => {});
      setRefreshKey((k) => k + 1);
    };
    window.addEventListener("chym:demo-refresh", onDemoRefresh);
    return () => window.removeEventListener("chym:demo-refresh", onDemoRefresh);
  }, []);

  return (
    <div className="w-full">
      <PageHeader
        title="ICU AKI 모니터링"
        subtitle="응급의학과 + 신장내과 공유 대시보드"
      />

      {/* 합성 데이터 고지 */}
      <div className="mb-3 flex items-start gap-2 rounded-md border border-primary/30 bg-primary/5 px-3 py-2 text-[11px] text-foreground">
        <Info className="mt-0.5 size-3.5 shrink-0 text-primary" />
        <span>{SYNTHETIC_NAME_NOTICE}</span>
      </div>

      {/* 요약 KPI */}
      <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-5">
        <StatCard label="ICU 환자" value={summary?.total ?? "—"} unit="명" tone="primary" />
        <StatCard label="고위험 환자" value={summary?.high ?? "—"} unit="명" tone="danger" hint="Stage 2-3 예측" />
        <StatCard label="중등도 위험" value={summary?.moderate ?? "—"} unit="명" tone="warning" />
        <StatCard label="안정" value={summary?.low ?? "—"} unit="명" tone="success" />
        <StatCard label="ICU 병동" value={summary?.nCareunits ?? "—"} unit="개" tone="accent" />
      </div>

      {/* 환자 검색 */}
      <IcuPatientSearch className="mb-4" />

      {/* 메인 레이아웃 (70:30 분할) */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[7fr_3fr] h-[calc(100vh-280px)] min-h-[600px]">
        {/* 좌측 70% : 환자 리스트 */}
        <div className="min-w-0 h-full">
          <ClinicalPatientList refreshKey={refreshKey} />
        </div>

        {/* 우측 30% : 알람 센터 & 협진 요청 현황 */}
        <div className="flex flex-col gap-4 min-w-0 h-full">
          <div className="flex-1 min-h-0">
            <AlarmCenter />
          </div>
          <div className="h-[300px] shrink-0">
            <ConsultationStatus />
          </div>
        </div>
      </div>
    </div>
  );
}
