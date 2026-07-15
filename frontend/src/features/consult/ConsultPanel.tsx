import { useEffect, useMemo } from "react";
import { Link } from "react-router-dom";
import { ChevronRight, Clock, CheckCircle2, PlayCircle } from "lucide-react";
import type { Consult, ConsultKind, ConsultStatus } from "@/types";
import { CONSULT_STATUS_LABEL, URGENCY_LABEL } from "@/types";
import { useConsultStore } from "@/store/consultStore";
import { consultStatusTone, urgencyTone } from "@/lib/statusTone";
import { formatRelative } from "@/lib/format";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusBadge } from "@/components/common/StatusBadge";
import { cn } from "@/lib/cn";

/** 종류별 협진을 한 줄 카운트 + 상위 3건 간략 목록으로 보여주는 컬럼. */
function ConsultColumn({
  kind,
  title,
}: {
  kind: ConsultKind;
  title: string;
}) {
  const items = useConsultStore((s) => s.items);
  const consults = useMemo(
    () => items.filter((c) => c.kind === kind).sort((a, b) => b.requestedAt.localeCompare(a.requestedAt)),
    [items, kind],
  );

  const counts = useMemo(() => {
    const c = { requested: 0, in_progress: 0, done: 0 };
    consults.forEach((x) => {
      if (x.status === "requested") c.requested += 1;
      else if (x.status === "in_progress") c.in_progress += 1;
      else c.done += 1;
    });
    return c;
  }, [consults]);

  const top = consults.slice(0, 3);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <span className="text-sm font-semibold text-foreground">{title}</span>
      </div>

      {/* 상태별 건수 — 대기중 / 진행중 / 완료 */}
      <div className="grid grid-cols-3 gap-2">
        {([
          { n: counts.requested, label: "대기중", Icon: Clock, cls: "text-warning" },
          { n: counts.in_progress, label: "진행중", Icon: PlayCircle, cls: "text-info" },
          { n: counts.done, label: "완료", Icon: CheckCircle2, cls: "text-success" },
        ]).map((s) => (
          <div key={s.label} className="rounded-md border border-border bg-muted/30 px-2 py-1.5 text-center">
            <p className={cn("flex items-center justify-center gap-1 text-base font-bold", s.cls)}>
              <s.Icon className="size-3.5" />
              {s.n}
            </p>
            <p className="text-[10px] text-muted-foreground">{s.label}</p>
          </div>
        ))}
      </div>

      {/* 간략 목록 — 최근 3건만 */}
      <div className="flex flex-col gap-1">
        {top.length === 0 ? (
          <p className="px-1 py-2 text-[11px] text-muted-foreground">협진 내역이 없습니다.</p>
        ) : (
          top.map((c: Consult) => (
            <div key={c.id} className="flex items-center justify-between gap-2 rounded-md px-1.5 py-1">
              <span className="flex min-w-0 items-center gap-1.5">
                <StatusBadge label={URGENCY_LABEL[c.urgency]} tone={urgencyTone[c.urgency]} />
                <span className="truncate text-xs text-foreground">{c.patientName}</span>
              </span>
              <span className="flex shrink-0 items-center gap-1.5">
                <StatusBadge label={CONSULT_STATUS_LABEL[c.status as ConsultStatus]} tone={consultStatusTone[c.status]} dot />
                <span className="text-[10px] text-muted-foreground/70">{formatRelative(c.requestedAt)}</span>
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

/**
 * 협진 패널(하단·보조) — 신장내과 메인 화면 맨 아래. 협진을 핵심 영역에서 내리고
 * 대기/진행/완료 현황과 최근 건만 간략히 보여준다. 상세는 협진 센터로 이동.
 */
export function ConsultPanel() {
  const items = useConsultStore((s) => s.items);
  const load = useConsultStore((s) => s.load);

  useEffect(() => {
    if (items.length === 0) load();
  }, [items.length, load]);

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between">
        <CardTitle>협진 패널</CardTitle>
        <Link
          to="/nephrology/consults"
          className="flex items-center gap-0.5 text-[11px] font-medium text-primary hover:underline"
        >
          협진 센터 <ChevronRight className="size-3.5" />
        </Link>
      </CardHeader>
      <CardContent className="grid gap-5 md:grid-cols-2">
        <ConsultColumn kind="nephrology" title="응급 협진 (ICU/ER 수신)" />
        <ConsultColumn kind="pathology" title="병리 협진 (→ 병리과 의뢰)" />
      </CardContent>
    </Card>
  );
}
