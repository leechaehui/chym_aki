import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Ambulance, Microscope, Reply, PlayCircle, CheckCircle2, ChevronDown, ChevronUp, Clock } from "lucide-react";
import type { Consult, ConsultKind } from "@/types";
import { CONSULT_STATUS_LABEL, URGENCY_LABEL } from "@/types";
import { useConsultStore } from "@/store/consultStore";
import { useAuthStore } from "@/store/authStore";
import { consultStatusTone, urgencyTone } from "@/lib/statusTone";
import { formatDateTime, formatRelative } from "@/lib/format";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { StatusBadge } from "@/components/common/StatusBadge";
import { Timeline, type TimelineItem } from "@/components/common/Timeline";
import { EmptyState } from "@/components/common/EmptyState";
import { cn } from "@/lib/cn";
import { ConsultReplyModal } from "./ConsultReplyModal";

function toTimeline(c: Consult): TimelineItem[] {
  return c.timeline.map((e) => ({ id: e.id, title: e.label, meta: formatDateTime(e.at), description: e.actor, done: true }));
}

/**
 * 협진 인박스(양방향·적응형). 같은 consultStore 를 응급/신장/병리 화면이 공유하므로
 * kind(협진 종류) 와 mode(보는 직군의 입장)만 달리해 한 컴포넌트로 세 화면을 덮는다.
 *
 * - mode="receiver" : 협진을 받는 측(예: 신장내과). 접수(진행 전이)·회신 작성을 한다.
 * - mode="requester": 협진을 보낸 측(예: 응급의학과). 상태·회신을 열람만 한다(읽기 전용).
 *
 * 적응형 레이아웃: 처리 대기(회신 전) 건이 있으면 펼쳐 강조하고, 모두 회신완료면
 * 한 줄 카운트 스트립으로 접어 주 업무 화면을 가리지 않는다(사용자가 수동 토글 가능).
 */
interface ConsultInboxProps {
  kind: ConsultKind;
  mode: "receiver" | "requester";
  title: string;
  emptyTitle: string;
  emptyDescription: string;
}

export function ConsultInbox({ kind, mode, title, emptyTitle, emptyDescription }: ConsultInboxProps) {
  const items = useConsultStore((s) => s.items);
  const load = useConsultStore((s) => s.load);
  const accept = useConsultStore((s) => s.accept);
  // 회신 부서명(회신 블록 라벨 + 접수자 기본값)은 협진 종류로 결정.
  const replyDept = kind === "nephrology" ? "신장내과" : "병리과";
  const userName = useAuthStore((s) => s.user?.name ?? replyDept);

  const Icon = kind === "nephrology" ? Ambulance : Microscope;

  const [searchParams, setSearchParams] = useSearchParams();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [replyOpen, setReplyOpen] = useState(false);
  // null = 자동(대기 유무로 결정), true/false = 사용자가 수동으로 토글.
  const [manualExpanded, setManualExpanded] = useState<boolean | null>(null);

  useEffect(() => {
    if (items.length === 0) load();
  }, [items.length, load]);

  const consults = useMemo(
    () => items.filter((c) => c.kind === kind).sort((a, b) => b.requestedAt.localeCompare(a.requestedAt)),
    [items, kind],
  );

  useEffect(() => {
    if (!selectedId && consults.length) setSelectedId(consults[0].id);
  }, [consults, selectedId]);

  // 알림 딥링크(?consult=<id>) → 해당 협진 선택 + 강제 펼침. 적용 후 파라미터 소비.
  useEffect(() => {
    const wanted = searchParams.get("consult");
    if (!wanted) return;
    if (consults.some((c) => c.id === wanted)) {
      setSelectedId(wanted);
      setManualExpanded(true);
    }
    searchParams.delete("consult");
    setSearchParams(searchParams, { replace: true });
  }, [searchParams, consults, setSearchParams]);

  const selected = consults.find((c) => c.id === selectedId) ?? null;

  const counts = useMemo(() => {
    const c = { requested: 0, in_progress: 0, replied: 0 };
    consults.forEach((x) => {
      if (x.status === "requested") c.requested += 1;
      else if (x.status === "in_progress") c.in_progress += 1;
      else if (x.status === "replied" || x.status === "read") c.replied += 1;
    });
    return c;
  }, [consults]);

  // 처리 대기(회신 전) 건과 응급도. 적응형 펼침/강조의 근거.
  const pending = consults.filter((c) => c.status !== "replied");
  const hasPending = pending.length > 0;
  const hasEmergency = pending.some((c) => c.urgency === "emergency");
  const expanded = manualExpanded ?? hasPending;

  // 접힘 상태 — 한 줄 카운트 스트립(클릭 시 펼침).
  if (!expanded) {
    return (
      <Card>
        <button
          onClick={() => setManualExpanded(true)}
          className="flex w-full items-center justify-between gap-2 px-6 py-3 text-left transition-colors hover:bg-muted/40"
        >
          <span className="flex items-center gap-2">
            <Icon className={cn("size-4", hasEmergency ? "text-destructive" : "text-muted-foreground")} />
            <CardTitle>{title}</CardTitle>
          </span>
          <span className="flex items-center gap-2 text-[11px] text-muted-foreground">
            {hasPending ? (
              <span className="flex items-center gap-1 font-medium text-warning">
                <Clock className="size-3.5" /> 대기 {counts.requested + counts.in_progress}
              </span>
            ) : (
              <span className="flex items-center gap-1 text-success">
                <CheckCircle2 className="size-3.5" /> 모두 회신완료
              </span>
            )}
            <span>· 총 {consults.length}건</span>
            <ChevronDown className="size-4" />
          </span>
        </button>
      </Card>
    );
  }

  return (
    <Card className={cn(hasEmergency && "ring-1 ring-destructive/30")}>
      <CardHeader className="flex-row flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Icon className={cn("size-4", hasEmergency ? "text-destructive" : "text-primary")} />
          <CardTitle>{title}</CardTitle>
        </div>
        <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
          <span>대기 {counts.requested}</span>·<span>진행 {counts.in_progress}</span>·<span>완료 {counts.replied}</span>
          {consults.length > 0 && (
            <button
              onClick={() => setManualExpanded(false)}
              className="ml-1 flex items-center gap-0.5 rounded px-1 py-0.5 hover:bg-muted/60"
              aria-label="접기"
            >
              <ChevronUp className="size-4" />
            </button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {consults.length === 0 ? (
          <EmptyState title={emptyTitle} description={emptyDescription} />
        ) : (
          <div className="grid gap-4 md:grid-cols-[260px_1fr]">
            {/* 목록 */}
            <div className="flex flex-col gap-1">
              {consults.map((c) => (
                <button
                  key={c.id}
                  onClick={() => setSelectedId(c.id)}
                  className={cn(
                    "flex w-full flex-col gap-1 rounded-md border px-3 py-2 text-left transition-colors",
                    c.id === selectedId ? "border-primary/40 bg-primary/5" : "border-transparent hover:bg-muted/60",
                  )}
                >
                  <div className="flex items-center justify-between gap-1">
                    <span className="text-sm font-medium text-foreground">
                      {c.patientName}{" "}
                      <span className="text-[11px] font-normal text-muted-foreground">{c.bedLabel ?? c.patientMrn}</span>
                    </span>
                    <StatusBadge label={CONSULT_STATUS_LABEL[c.status]} tone={consultStatusTone[c.status]} dot />
                  </div>
                  <span className="truncate text-[11px] text-muted-foreground">{c.diagnosis}</span>
                  <div className="flex items-center justify-between">
                    <StatusBadge label={URGENCY_LABEL[c.urgency]} tone={urgencyTone[c.urgency]} />
                    <span className="text-[10px] text-muted-foreground/70">{formatRelative(c.requestedAt)}</span>
                  </div>
                </button>
              ))}
            </div>

            {/* 상세 */}
            {selected ? (
              <div className="flex flex-col gap-3 rounded-lg border border-border bg-muted/20 p-3">
                <div className="flex flex-wrap items-center gap-2">
                  <StatusBadge label={CONSULT_STATUS_LABEL[selected.status]} tone={consultStatusTone[selected.status]} dot />
                  <StatusBadge label={URGENCY_LABEL[selected.urgency]} tone={urgencyTone[selected.urgency]} />
                  <span className="text-[11px] text-muted-foreground">
                    {selected.bedLabel ?? selected.patientMrn} · 요청 {formatDateTime(selected.requestedAt)} · {selected.requestedBy}
                  </span>
                </div>

                <div>
                  <p className="text-[11px] font-semibold text-muted-foreground">주요 검사</p>
                  <p className="text-sm text-foreground">{selected.keyLabs}</p>
                </div>
                <div>
                  <p className="text-[11px] font-semibold text-muted-foreground">요청 사유</p>
                  <p className="text-sm leading-relaxed text-foreground">{selected.reason}</p>
                </div>

                <Timeline items={toTimeline(selected)} />

                {selected.reply ? (
                  <div className="rounded-lg border border-success/30 bg-success/5 p-3">
                    <p className="mb-2 text-[11px] font-semibold text-success">{replyDept} 회신 · {selected.reply.author}</p>
                    <dl className="flex flex-col gap-2 text-xs">
                      <div>
                        <dt className="text-[11px] text-muted-foreground">평가 소견</dt>
                        <dd className="text-foreground">{selected.reply.findings}</dd>
                      </div>
                      <div>
                        <dt className="text-[11px] text-muted-foreground">임상 판단</dt>
                        <dd className="font-medium text-foreground">{selected.reply.diagnosis}</dd>
                      </div>
                      <div>
                        <dt className="text-[11px] text-muted-foreground">처치 권고</dt>
                        <dd className="text-foreground">{selected.reply.recommendation}</dd>
                      </div>
                    </dl>
                    <p className="mt-2 text-[10px] text-muted-foreground">회신 {formatDateTime(selected.reply.repliedAt)}</p>
                  </div>
                ) : (
                  // 요청자(읽기 전용)는 회신 대기 안내, 수신자는 아래 액션 버튼으로 처리.
                  mode === "requester" && (
                    <div className="flex items-center gap-2 rounded-lg border border-warning/30 bg-warning/5 p-3 text-[11px] text-warning">
                      <Clock className="size-3.5" /> {replyDept} 회신을 기다리는 중입니다.
                    </div>
                  )
                )}

                {mode === "receiver" && (
                  <div className="flex flex-wrap items-center justify-end gap-2">
                    {selected.status === "requested" && (
                      <Button variant="outline" size="sm" onClick={() => accept(selected.id, userName)}>
                        <PlayCircle className="size-3.5" /> 협진 접수
                      </Button>
                    )}
                    {selected.status === "replied" ? (
                      <span className="flex items-center gap-1 text-[11px] text-success">
                        <CheckCircle2 className="size-3.5" /> 회신 완료
                      </span>
                    ) : (
                      <Button size="sm" onClick={() => setReplyOpen(true)}>
                        <Reply className="size-3.5" /> 협진 회신
                      </Button>
                    )}
                  </div>
                )}
              </div>
            ) : (
              <EmptyState title="협진 건을 선택하세요" />
            )}
          </div>
        )}
      </CardContent>

      {mode === "receiver" && <ConsultReplyModal consult={selected} open={replyOpen} onOpenChange={setReplyOpen} />}
    </Card>
  );
}
