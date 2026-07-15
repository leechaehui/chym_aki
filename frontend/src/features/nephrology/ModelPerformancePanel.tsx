import { useState } from "react";
import { FileText } from "lucide-react";
import type { ModelMetrics, ModelStageMetrics } from "@/types";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { cn } from "@/lib/cn";

export const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

function Metric({ label, value, hint, tone }: { label: string; value: string; hint?: string; tone?: string }) {
  return (
    <div className="rounded-lg border border-border bg-background px-3 py-2">
      <div className="text-[11px] text-muted-foreground">{label}</div>
      <div className={cn("text-lg font-semibold tabular-nums", tone)}>{value}</div>
      {hint && <div className="text-[10px] text-muted-foreground">{hint}</div>}
    </div>
  );
}

/**
 * 모델 성능 패널 — AKI 발생 예측(Stage1)의 정확도/정밀도/민감도/위음성률을 보여준다.
 * 이 값들은 모델 단위 상수(테스트셋 기준)라 환자 행이 아니라 목록 상단에 배치한다.
 * 중증도(Stage2) 한계 설명은 패널이 아니라 검증 리포트에 담는다.
 */
export function ModelPerformancePanel({ metrics }: { metrics: ModelMetrics | null }) {
  if (!metrics) return null;
  const s1 = metrics.stage1;
  return (
    <Card className="mb-4">
      <CardContent className="pt-4">
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <div className="text-sm font-semibold">
            모델 성능
            <span className="ml-1 text-[11px] font-normal text-muted-foreground">
              · AKI 발생 예측(Stage1) · 테스트셋 {metrics.dataset.nRows.toLocaleString()}건
            </span>
          </div>
          <ValidationReportModal metrics={metrics} />
        </div>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Metric label="정확도 (Accuracy)" value={pct(s1.accuracy)} />
          <Metric label="정밀도 (Precision)" value={pct(s1.precision)} />
          <Metric
            label="민감도 (Recall)"
            value={pct(s1.recall)}
            hint="실제 AKI를 잡아내는 비율"
            tone="text-success"
          />
          <Metric
            label="위음성률 (FNR)"
            value={pct(s1.fnr)}
            hint="실제 AKI를 놓친 비율"
            tone={s1.fnr > 0.15 ? "text-danger" : "text-foreground"}
          />
        </div>
      </CardContent>
    </Card>
  );
}

function MetricRow({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <div className="flex items-center justify-between gap-2 border-b border-border/40 py-1 last:border-0">
      <span className="text-muted-foreground">{label}</span>
      <span className={cn("tabular-nums font-medium", tone)}>{value}</span>
    </div>
  );
}

function StageBlock({ m }: { m: ModelStageMetrics }) {
  return (
    <div className="rounded-lg border border-border p-3">
      <div className="mb-2 text-[12px] font-semibold">{m.label}</div>
      <div className="text-[12px]">
        <MetricRow label="정확도 (Accuracy)" value={pct(m.accuracy)} />
        <MetricRow label="정밀도 (Precision)" value={pct(m.precision)} />
        <MetricRow label="민감도 (Recall)" value={pct(m.recall)} tone="text-success" />
        <MetricRow label="위음성률 (FNR)" value={pct(m.fnr)} tone={m.fnr > 0.15 ? "text-danger" : undefined} />
        <MetricRow label="특이도 (Specificity)" value={pct(m.specificity)} />
        <MetricRow label="AUROC" value={m.auroc.toFixed(3)} />
        <MetricRow label="AUPRC" value={m.auprc.toFixed(3)} tone="text-success" />
        <MetricRow label="보정오차 (ECE)" value={m.ece.toFixed(3)} />
      </div>
      <div className="mt-2 text-[10px] text-muted-foreground">
        n={m.n.toLocaleString()} · TP {m.tp} / FP {m.fp} / FN {m.fn} / TN {m.tn}
      </div>
    </div>
  );
}

/** 검증 리포트 본문 — 모델 단위 성능. 모델 패널·환자별 리포트 양쪽이 공유한다. */
export function ValidationReportBody({ metrics }: { metrics: ModelMetrics }) {
  return (
    <div>
      <p className="mb-3 text-[11px] text-muted-foreground">
        테스트셋 {metrics.dataset.nRows.toLocaleString()}건 · AKI 유병률 {metrics.dataset.akiPrevalence} · 임계값
        Stage1 {metrics.dataset.stage1Threshold} / Stage2 {metrics.dataset.stage2Threshold}
        {metrics.generatedAt ? ` · 생성 ${metrics.generatedAt.slice(0, 10)}` : ""}
      </p>
      <div className="grid gap-3 sm:grid-cols-2">
        <StageBlock m={metrics.stage1} />
        <StageBlock m={metrics.stage2} />
      </div>

      <div className="mt-3 rounded-md border border-warning/40 bg-warning/10 px-3 py-2 text-[11px] text-foreground">
        ※ 중증도(Stage2) 변별력은 제한적입니다 (AUROC {metrics.stage2.auroc.toFixed(2)} · 위음성률{" "}
        {pct(metrics.stage2.fnr)}). 보조 지표로만 사용하고 단독 임상판단에 쓰지 마세요.
      </div>

      <div className="mt-3">
        <div className="mb-1 text-[12px] font-semibold">알려진 한계 (Known limitations)</div>
        <ul className="list-disc space-y-1 pl-4 text-[11px] text-muted-foreground">
          {metrics.knownFailures.map((f, i) => (
            <li key={i}>{f}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function ValidationReportModal({ metrics }: { metrics: ModelMetrics }) {
  const [open, setOpen] = useState(false);
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <Button size="sm" variant="subtle" onClick={() => setOpen(true)}>
        <FileText className="mr-1 size-3.5" /> 검증 리포트
      </Button>
      <DialogContent className="max-h-[85vh] max-w-2xl overflow-y-auto">
        <DialogHeader>
          <DialogTitle>AKI 2-Stage 모델 검증 리포트</DialogTitle>
        </DialogHeader>
        <ValidationReportBody metrics={metrics} />
      </DialogContent>
    </Dialog>
  );
}
