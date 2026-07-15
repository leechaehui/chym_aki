import { useMemo, useState } from "react";
import { MessagesSquare, FileText } from "lucide-react";
import { useConsultStore } from "@/store/consultStore";
import { CONSULT_STATUS_LABEL, URGENCY_LABEL } from "@/types";
import type { Consult, ConsultStatus } from "@/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusBadge } from "@/components/common/StatusBadge";
import { EmptyState } from "@/components/common/EmptyState";
import { consultStatusTone, urgencyTone } from "@/lib/statusTone";
import { formatRelative } from "@/lib/format";
import { PathologyReportModal } from "@/features/consult/PathologyReportModal";

export function ConsultationStatus() {
  const items = useConsultStore((s) => s.items);
  const recent = useMemo(
    () => [...items].sort((a, b) => b.requestedAt.localeCompare(a.requestedAt)).slice(0, 8),
    [items],
  );
  // 병리 회신이 도착한 건은 클릭 시 정식 병리 리포트 모달로 연다.
  const [reportConsult, setReportConsult] = useState<Consult | null>(null);

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="flex-row items-center gap-2 pb-2">
        <MessagesSquare className="size-4 text-primary" />
        <CardTitle>협진 현황</CardTitle>
      </CardHeader>
      <CardContent className="flex-1 min-h-0 overflow-y-auto p-2">
        {recent.length === 0 ? (
          <EmptyState title="협진 요청 없음" />
        ) : (
          <ul className="flex flex-col gap-1.5">
            {recent.map((c) => {
              const replied = !!c.reply;
              return (
                <li
                  key={c.id}
                  className={replied ? "rounded-md border border-success/30 bg-success/5 px-2.5 py-1.5" : "rounded-md border border-border px-2.5 py-1.5"}
                >
                  <button
                    type="button"
                    onClick={() => replied && setReportConsult(c)}
                    className={replied ? "flex w-full flex-col gap-0.5 text-left hover:opacity-80" : "flex w-full flex-col gap-0.5 text-left"}
                    disabled={!replied}
                  >
                    <div className="flex items-center justify-between gap-1">
                      <span className="text-xs font-medium text-foreground">{c.patientName}</span>
                      <StatusBadge label={CONSULT_STATUS_LABEL[c.status as ConsultStatus]} tone={consultStatusTone[c.status]} dot />
                    </div>
                    <div className="flex items-center justify-between gap-1">
                      {replied ? (
                        <span className="inline-flex items-center gap-1 text-[10px] font-semibold text-success">
                          <FileText className="size-3" /> 병리 리포트 보기 →
                        </span>
                      ) : (
                        <StatusBadge label={URGENCY_LABEL[c.urgency]} tone={urgencyTone[c.urgency]} />
                      )}
                      <span className="text-[10px] text-muted-foreground/70">{formatRelative(c.requestedAt)}</span>
                    </div>
                  </button>
                </li>
              );
            })}
          </ul>
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
