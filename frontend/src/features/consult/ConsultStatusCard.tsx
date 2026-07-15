import { useEffect, useState } from "react";
import { ClipboardList, FileText } from "lucide-react";
import type { Consult, ConsultEvent } from "@/types";
import { CONSULT_STATUS_LABEL, URGENCY_LABEL } from "@/types";
import { useConsultStore } from "@/store/consultStore";
import { consultStatusTone, urgencyTone } from "@/lib/statusTone";
import { formatDateTime } from "@/lib/format";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { StatusBadge } from "@/components/common/StatusBadge";
import { Timeline, type TimelineItem } from "@/components/common/Timeline";
import { EmptyState } from "@/components/common/EmptyState";
import { PathologyReportModal } from "./PathologyReportModal";

/** 협진 단계 → 타임라인 표시 + 완료 여부(현재 상태까지 done 처리). */
const STAGE_ORDER = ["requested", "received", "analyzing", "read", "replied"];
function toTimeline(events: readonly ConsultEvent[]): TimelineItem[] {
  return events.map((e) => ({ id: e.id, title: e.label, meta: formatDateTime(e.at), description: e.actor, done: true }));
}

/**
 * 환자별 협진 현황 카드(신장내과 측). 해당 환자의 협진 진행 타임라인과
 * 병리과 회신(소견/진단/권고)을 표시 — 협진 요청→회신 루프를 닫는다.
 */
export function ConsultStatusCard({ patientMrn }: { patientMrn: string }) {
  const items = useConsultStore((s) => s.items);
  const load = useConsultStore((s) => s.load);
  const [reportConsult, setReportConsult] = useState<Consult | null>(null);

  useEffect(() => {
    if (items.length === 0) load();
  }, [items.length, load]);

  const consults = items.filter((c) => c.kind === "pathology" && c.patientMrn === patientMrn);

  return (
    <Card>
      <CardHeader className="flex-row items-center gap-2">
        <ClipboardList className="size-4 text-primary" />
        <CardTitle>병리 협진 현황</CardTitle>
      </CardHeader>
      <CardContent>
        {consults.length === 0 ? (
          <EmptyState title="진행 중인 협진이 없습니다" description="상단의 '병리 협진 요청'으로 협진을 의뢰할 수 있습니다." />
        ) : (
          <div className="flex flex-col gap-5">
            {consults.map((c) => (
              <div key={c.id}>
                <div className="mb-3 flex items-center gap-2">
                  <StatusBadge label={CONSULT_STATUS_LABEL[c.status]} tone={consultStatusTone[c.status]} dot />
                  <StatusBadge label={URGENCY_LABEL[c.urgency]} tone={urgencyTone[c.urgency]} />
                  <span className="text-[11px] text-muted-foreground">요청 {formatDateTime(c.requestedAt)}</span>
                </div>

                <Timeline items={toTimeline(c.timeline)} />

                {c.reply && (
                  <div className="mt-3 rounded-lg border border-success/30 bg-success/5 p-3">
                    <div className="mb-2 flex items-center justify-between gap-2">
                      <p className="text-[11px] font-semibold text-success">병리과 회신 · {c.reply.author}</p>
                      <Button size="sm" variant="subtle" onClick={() => setReportConsult(c)}>
                        <FileText className="size-3.5" /> 병리 리포트 열기
                      </Button>
                    </div>
                    <dl className="flex flex-col gap-2 text-xs">
                      <div>
                        <dt className="text-[11px] text-muted-foreground">소견</dt>
                        <dd className="line-clamp-2 whitespace-pre-wrap text-foreground">{c.reply.findings}</dd>
                      </div>
                      <div>
                        <dt className="text-[11px] text-muted-foreground">최종 진단</dt>
                        <dd className="line-clamp-2 whitespace-pre-wrap font-medium text-foreground">{c.reply.diagnosis}</dd>
                      </div>
                    </dl>
                    <p className="mt-2 text-[10px] text-muted-foreground">회신 {formatDateTime(c.reply.repliedAt)} · 전체는 "병리 리포트 열기"</p>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </CardContent>

      {reportConsult && (
        <PathologyReportModal
          consult={reportConsult}
          open={reportConsult !== null}
          onOpenChange={(o) => { if (!o) setReportConsult(null); }}
        />
      )}
    </Card>
  );
}
