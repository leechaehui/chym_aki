import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import type { IcuAkiPatient, PatientIdentity, ModelMetrics } from "@/types";
import { icuMonitorService } from "@/services/icuMonitorService";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LoadingSpinner } from "@/components/common/LoadingSpinner";
import { EmptyState } from "@/components/common/EmptyState";
import { fakeKoreanName } from "@/lib/fakeName";
import { riskBandFor } from "@/lib/riskLabel";
import { cn } from "@/lib/cn";
import { PatientQuickView } from "./PatientQuickView";
import { PatientReportModal } from "./PatientReportModal";

const PAGE_SIZE = 15;

function RiskBadge({ score, stage }: { score: number; stage?: string }) {
  // 등급은 점수가 아니라 모델의 실제 예측 stage(pred) 기준 — "단계" 열과 같은 소스를 써야
  // "Stage 1인데 고위험"처럼 두 열이 서로 다른 걸 말하는 모순이 안 생긴다.
  // 목록 정렬도(icu_monitor_service.list_patients) pred 우선이라 이 뱃지 기준과 일치한다.
  const band = riskBandFor(score, stage);
  const dotColor = band.tier === "high" ? "bg-red-500" : band.tier === "moderate" ? "bg-amber-400" : "bg-emerald-500";
  return (
    <span className={cn("inline-flex items-center gap-1.5 text-xs font-semibold tabular-nums", band.textClass)}>
      <span className={cn("w-2 h-2 rounded-full shrink-0", dotColor)} />
      {band.label} {score}
    </span>
  );
}

export function ClinicalPatientList({ refreshKey = 0 }: { refreshKey?: number }) {
  const [searchParams, setSearchParams] = useSearchParams();
  const [patients, setPatients] = useState<IcuAkiPatient[]>([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<PatientIdentity | null>(null);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  // 알림 바로가기(?patient=AKI-xxxx)로 들어오면 해당 환자의 검증 리포트를 자동으로 연다.
  const [reportPatient, setReportPatient] = useState<PatientIdentity | null>(null);

  useEffect(() => {
    setLoading(true);
    icuMonitorService
      .list({ limit: 1000 })
      .then(setPatients)
      .catch(() => setPatients([]))
      .finally(() => setLoading(false));
  }, [refreshKey]);

  // 알림 모달 "바로가기" 딥링크 처리 — 환자 목록이 로드된 뒤 subjectId 로 매칭해 검증 리포트 오픈.
  useEffect(() => {
    const raw = searchParams.get("patient");
    if (!raw || patients.length === 0) return;
    const sid = Number(raw.replace(/[^\d]/g, ""));
    if (!Number.isFinite(sid) || sid === 0) return;
    const match = patients.find((p) => p.subjectId === sid || p.stayId === sid);
    if (match) {
      setReportPatient({ stayId: match.stayId, subjectId: match.subjectId, age: match.age ?? undefined, gender: match.gender ?? undefined, careunit: match.careunit });
      icuMonitorService.modelMetrics().then(setMetrics).catch(() => {});
      // 파라미터 소비(중복 트리거 방지)
      searchParams.delete("patient");
      setSearchParams(searchParams, { replace: true });
    }
  }, [searchParams, patients, setSearchParams]);

  const totalPages = Math.ceil(patients.length / PAGE_SIZE);
  const visible = patients.slice(page * PAGE_SIZE, page * PAGE_SIZE + PAGE_SIZE);

  function handleRowClick(pt: IcuAkiPatient) {
    setSelected({ stayId: pt.stayId, subjectId: pt.subjectId, age: pt.age ?? undefined, gender: pt.gender ?? undefined });
    icuMonitorService.modelMetrics().then(setMetrics).catch(() => {});
  }

  return (
    <>
      <Card className="h-full flex flex-col">
        <CardHeader>
          <CardTitle>임상 환자 목록 (위험순)</CardTitle>
        </CardHeader>
        <CardContent className="flex-1 min-h-0 overflow-y-auto p-0">
          {loading ? (
            <LoadingSpinner label="환자 목록 불러오는 중" />
          ) : patients.length === 0 ? (
            <EmptyState title="환자 데이터 없음" />
          ) : (
            <>
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-card border-b border-border">
                  <tr>
                    {["#", "환자(합성)", "나이/성", "병동", "단계", "위험점수", "확률"].map((h) => (
                      <th key={h} className="px-3 py-2 text-left text-[11px] text-muted-foreground font-medium">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {visible.map((pt, i) => (
                    <tr
                      key={pt.stayId}
                      className="border-b border-border/40 hover:bg-muted/40 cursor-pointer"
                      onClick={() => handleRowClick(pt)}
                    >
                      <td className="px-3 py-2 text-muted-foreground">{page * PAGE_SIZE + i + 1}</td>
                      <td className="px-3 py-2 font-medium">{fakeKoreanName(pt.subjectId)}</td>
                      <td className="px-3 py-2">{pt.age ?? "—"}/{pt.gender ?? "—"}</td>
                      <td className="px-3 py-2 truncate max-w-[100px]">{pt.careunit}</td>
                      <td className="px-3 py-2">{pt.stage}</td>
                      <td className="px-3 py-2"><RiskBadge score={pt.riskScore} stage={pt.stage} /></td>
                      <td className="px-3 py-2 tabular-nums">{(pt.pStage2Plus * 100).toFixed(0)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {totalPages > 1 && (
                <div className="flex items-center justify-between px-3 py-2 border-t border-border">
                  <button
                    onClick={() => setPage((p) => Math.max(0, p - 1))}
                    disabled={page === 0}
                    className="text-xs text-muted-foreground disabled:opacity-40 hover:text-foreground"
                  >
                    ← 이전
                  </button>
                  <span className="text-[11px] text-muted-foreground">{page + 1}/{totalPages}</span>
                  <button
                    onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                    disabled={page >= totalPages - 1}
                    className="text-xs text-muted-foreground disabled:opacity-40 hover:text-foreground"
                  >
                    다음 →
                  </button>
                </div>
              )}
            </>
          )}
        </CardContent>
      </Card>

      <PatientQuickView
        patient={selected}
        metrics={metrics}
        onClose={() => setSelected(null)}
      />

      {/* 알림 바로가기 딥링크로 열리는 검증 리포트(controlled) */}
      {reportPatient && (
        <PatientReportModal
          patient={reportPatient}
          metrics={metrics}
          open={reportPatient !== null}
          onOpenChange={(o) => { if (!o) setReportPatient(null); }}
        />
      )}
    </>
  );
}
