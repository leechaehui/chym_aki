import { AlertTriangle, Info } from "lucide-react";
import type { RrtAssessment, RrtLevel } from "@/types";
import { cn } from "@/lib/cn";

/** 단계별 표시 색/배경(NONE 안정 → URGENT 위급). */
const LEVEL_STYLE: Record<RrtLevel, { dot: string; text: string; box: string; ko: string }> = {
  NONE: { dot: "🟢", text: "text-success", box: "border-success/40 bg-success/10", ko: "신호 없음" },
  DISCUSSION: { dot: "🟡", text: "text-amber-500", box: "border-amber-500/40 bg-amber-500/10", ko: "논의 권장" },
  CONSIDERATION: { dot: "🟠", text: "text-warning", box: "border-warning/40 bg-warning/10", ko: "고려 단계" },
  URGENT: { dot: "🔴", text: "text-destructive", box: "border-destructive/50 bg-destructive/10", ko: "긴급 평가" },
};

const TREND_KO: Record<string, string> = {
  stable: "안정",
  worsening: "악화",
  "rapidly deteriorating": "빠른 악화",
};

/**
 * RRT(신대체요법) 트리거 — 임상 의사결정 보조(예측 아님).
 * KDIGO+ICU 규칙으로 '논의/고려/긴급 평가' 단계를 분류만 한다. "RRT 필요/확률" 단정 금지.
 */
export function RrtTriggerPanel({ assessment }: { assessment: RrtAssessment }) {
  const style = LEVEL_STYLE[assessment.rrtLevel];
  return (
    <div className="space-y-2">
      {/* 단계 배지 + 행동 */}
      <div className={cn("flex items-start justify-between gap-3 rounded-md border px-3 py-2", style.box)}>
        <div>
          <div className={cn("text-sm font-bold", style.text)}>
            {style.dot} {assessment.label}
          </div>
          <div className="mt-0.5 text-[11px] text-foreground">{assessment.clinicalAction}</div>
        </div>
        <div className="shrink-0 text-right text-[10px] text-muted-foreground">
          <div>추세 {TREND_KO[assessment.trendSignal] ?? assessment.trendSignal}</div>
          <div>신뢰도 {(assessment.confidence * 100).toFixed(0)}%</div>
        </div>
      </div>

      {/* 근거 */}
      <div>
        <div className="mb-1 text-[11px] font-medium">트리거 근거</div>
        <ul className="list-disc space-y-0.5 pl-4 text-[11px] text-foreground">
          {assessment.keyDrivers.map((d, i) => <li key={i}>{d}</li>)}
        </ul>
      </div>

      {/* 정직성: 평가 못한 기준 */}
      {assessment.unavailableCriteria.length > 0 && (
        <div className="rounded-md border border-border bg-muted/30 px-3 py-2 text-[10px] text-muted-foreground">
          <span className="font-medium">평가 제외 기준(데이터 미가용):</span>
          <ul className="mt-0.5 list-disc pl-4">
            {assessment.unavailableCriteria.map((c, i) => <li key={i}>{c}</li>)}
          </ul>
        </div>
      )}

      {/* 안전 문구 */}
      <div className="flex items-start gap-2 rounded-md border border-warning/40 bg-warning/10 px-3 py-2 text-[11px] text-foreground">
        <AlertTriangle className="mt-0.5 size-3.5 shrink-0 text-warning" />
        <span>{assessment.warningNote}</span>
      </div>

      {/* 엔진 정체성 한 줄 요약 */}
      <div className="flex items-start gap-2 rounded-md border border-primary/30 bg-primary/5 px-3 py-2 text-[11px] text-foreground">
        <Info className="mt-0.5 size-3.5 shrink-0 text-primary" />
        <span>
          이 시스템은 <b>"투석을 예측하는 AI"</b>가 아니라{" "}
          <b>"투석이 논의되어야 하는 순간을 잡아주는 임상 트리거 엔진"</b>입니다 (KDIGO + ICU 규칙 기반).
        </span>
      </div>
    </div>
  );
}
