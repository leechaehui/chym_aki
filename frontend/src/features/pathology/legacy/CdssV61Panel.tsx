import { ChevronDown, ShieldAlert, ShieldCheck, ShieldQuestion, Activity } from "lucide-react";
import type { WsiAnalysisResult, WsiStain } from "@/types/wsi";
import type { CdssDescriptors, CdssV61Decision, V61Axis, V61Confidence, V61Level } from "@/types/cdss";
import { buildCdssV61 } from "./cdssV61";
import { EmptyState } from "@/components/common/EmptyState";
import { cn } from "@/lib/cn";

const LEVEL_TONE: Record<V61Level, { bar: string; text: string; bg: string }> = {
  HIGH: { bar: "bg-destructive", text: "text-destructive", bg: "bg-destructive/10" },
  MODERATE: { bar: "bg-warning", text: "text-warning", bg: "bg-warning/10" },
  LOW: { bar: "bg-success", text: "text-success", bg: "bg-success/10" },
};

const CONF_META: Record<V61Confidence, { icon: typeof ShieldCheck; tone: string; label: string }> = {
  CONFIDENT: { icon: ShieldCheck, tone: "text-success", label: "CONFIDENT · 신뢰" },
  SHADOW: { icon: ShieldQuestion, tone: "text-warning", label: "SHADOW · 주의 해석" },
  ABSTAIN: { icon: ShieldAlert, tone: "text-destructive", label: "ABSTAIN · 판단 보류" },
};

/** L2/L3 공용 행 — 라벨(+합성/미측정 note) + level 칩 + score 막대. */
function AxisRow({ a }: { a: V61Axis }) {
  const tone = LEVEL_TONE[a.level];
  return (
    <div className="mb-2.5">
      <div className="mb-1 flex items-center justify-between gap-2 text-[11px]">
        <span className="flex items-center gap-1 text-foreground">
          {a.label}
          {!a.real && a.note && (
            <span className="rounded bg-muted px-1 text-[9px] text-muted-foreground" title={a.note}>
              {a.note.includes("미측정") ? "미측정" : "합성"}
            </span>
          )}
        </span>
        <span className={cn("rounded px-1.5 py-0.5 text-[10px] font-bold", tone.bg, tone.text)}>{a.level}</span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
        <div className={cn("h-full rounded-full transition-all", tone.bar)} style={{ width: `${Math.min(100, Math.max(2, a.score * 100))}%` }} />
      </div>
    </div>
  );
}

/**
 * CHYM v6.1 CDSS 결과 패널 (§5 3-layer, 실데이터 전용).
 *
 * 별도 트리거 없음 — viewer 탭의 'AI 분석'(실제 ABMIL 추론) 결과 wsiResult 를 그대로 소비한다.
 * L1 AKI risk 는 환자의 실제 EMR aiRiskScore(있으면)로, 없으면 WSI 파생(라벨 표기).
 */
export function CdssV61Panel({
  wsiResult,
  aiRiskScore,
  stain,
  descriptors = null,
}: {
  wsiResult: WsiAnalysisResult | null;
  aiRiskScore: number | null;
  stain: WsiStain;
  descriptors?: CdssDescriptors | null;
}) {
  const decision: CdssV61Decision | null = buildCdssV61(wsiResult, aiRiskScore, stain, descriptors);

  if (!decision) {
    return <EmptyState icon={Activity} title="AI 분석 대기" description="‘AI 분석’을 실행하면 실제 ABMIL 모델 결과를 v6.1 3-layer(AKI 위험도·핵심 3축·상세)로 표시합니다." />;
  }

  const tone = LEVEL_TONE[decision.riskLevel];
  const conf = CONF_META[decision.confidence];
  const ConfIcon = conf.icon;

  return (
    <div className="flex flex-col gap-4">
      {/* 🟢 LAYER 1 — 임상 결론 */}
      <div className={cn("rounded-lg border border-border p-4", tone.bg)}>
        <div className="flex items-center justify-between">
          <p className="text-[11px] font-medium text-muted-foreground">AKI Progression Risk</p>
          <span className="rounded bg-card px-1.5 py-0.5 text-[9px] font-semibold text-muted-foreground">
            {decision.riskSource === "EMR" ? "EMR 실모델" : "WSI 파생"}
          </span>
        </div>
        <div className="mt-0.5 flex items-center justify-between gap-2">
          <span className={cn("text-2xl font-extrabold tracking-tight", tone.text)}>{decision.riskLevel}</span>
          <span className={cn("flex items-center gap-1 text-[11px] font-semibold", conf.tone)}>
            <ConfIcon className="size-3.5" />
            {conf.label}
          </span>
        </div>
        <p className="mt-1.5 text-[11px] leading-relaxed text-foreground/80">{decision.interpretation}</p>
        <p className="mt-0.5 text-[10px] text-muted-foreground/70">신뢰 근거: {decision.confidenceReason}</p>
      </div>

      {/* 🟡 LAYER 2 — 핵심 3축 */}
      <div className="rounded-lg border border-border bg-card p-3">
        <p className="mb-2 text-xs font-semibold text-foreground">핵심 임상 축</p>
        {decision.axes.map((a) => (
          <AxisRow key={a.key} a={a} />
        ))}
      </div>

      {/* 🔵 LAYER 3 — 상세 descriptor(접힘) */}
      <details className="group rounded-lg border border-border bg-card">
        <summary className="flex cursor-pointer list-none items-center justify-between p-3 text-xs font-semibold text-foreground">
          상세 Descriptors (실측 헤드)
          <ChevronDown className="size-4 text-muted-foreground transition-transform group-open:rotate-180" />
        </summary>
        <div className="border-t border-border p-3">
          <p className="mb-2 text-[10px] text-muted-foreground">
            실측 헤드(간질 섬유화·세뇨관 위축·간질 염증)의 Banff 기반 상세 정량입니다.
          </p>
          {decision.descriptors.map((a) => (
            <AxisRow key={a.key} a={a} />
          ))}
        </div>
      </details>

      {/* model trace (MFDS §11) */}
      <p className="text-[10px] text-muted-foreground/70">
        model {decision.modelLabel} · patches {decision.nPatches} · 본 결과는 진단이 아닌 임상 의사결정 보조입니다.
      </p>
    </div>
  );
}
