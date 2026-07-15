import { useState, useEffect } from "react";
import type { WsiAnalysisResult, WsiStain } from "@/types/wsi";
import { EmptyState } from "@/components/common/EmptyState";

type ModelKey = "he" | "mt" | "pas";

/** meter — 채움 색이 심각도(accent→warning→danger)를 나타낸다. */
function severityFill(pct: number): string {
  if (pct >= 50) return "bg-destructive";
  if (pct >= 25) return "bg-warning";
  return "bg-success";
}

/** 가로 막대 — 라벨 + 값. 끝만 4px 라운드, 시작(baseline)은 각지게. */
function Bar({
  label, value, suffix, pct, banff,
}: {
  label: string; value: string; suffix?: string; pct: number; banff?: number | null;
}) {
  const clamped = Math.min(100, Math.max(0, pct));
  return (
    <div className="mb-2.5">
      <div className="mb-1 flex items-center justify-between text-[11px]">
        <span className="text-foreground">{label}</span>
        <span className="font-semibold text-foreground">
          {value}
          {suffix && <span className="ml-0.5 font-normal text-muted-foreground">{suffix}</span>}
          {banff != null && (
            <span className="ml-1.5 rounded bg-muted px-1 py-px text-[10px] font-medium text-muted-foreground">
              Banff G{banff}
            </span>
          )}
        </span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded bg-muted">
        <div
          className={`h-full rounded-r transition-all ${severityFill(clamped)}`}
          style={{ width: `${Math.max(2, clamped)}%` }}
        />
      </div>
    </div>
  );
}

function metricVal(result: WsiAnalysisResult, key: string): number | null {
  return result.metrics.find((m) => m.key === key)?.value ?? null;
}

/** 키별 한글 라벨 폴백 — 모델이 해당 지표를 미예측해 metric.label 이 없을 때도 영문 키 대신 한글 노출. */
const KO_LABEL: Record<string, string> = {
  fibrosisRatio: "간질 섬유화",
  atrophyRatio: "세뇨관 위축",
  tubularInjury: "세뇨관 손상",
  inflammation: "간질 염증",
};

function metricLabel(result: WsiAnalysisResult, key: string): string {
  return result.metrics.find((m) => m.key === key)?.label ?? KO_LABEL[key] ?? key;
}

function metricUnit(result: WsiAnalysisResult, key: string): string {
  return result.metrics.find((m) => m.key === key)?.unit ?? "";
}

/** Banff 등급(0–3) — PAS(CdssEngine) 응답의 raw 는 ordinal 등급. 없으면 null(뱃지 미표시).
 *  ABMIL(HE/MT)은 raw 가 연속 회귀값이라 이 헬퍼는 PAS 패널에서만 호출한다. */
function metricBanff(result: WsiAnalysisResult, key: string): number | null {
  const m = result.metrics.find((mm) => mm.key === key);
  return m && m.raw != null ? Math.round(m.raw) : null;
}

/** HE 모델 결과 — 5개 타겟. */
function HEPanel({ result }: { result: WsiAnalysisResult }) {
  const keys = ["fibrosisRatio", "atrophyRatio", "tubularInjury", "inflammation"];

  return (
    <div className="flex flex-col gap-4">
      <p className="text-[11px] text-muted-foreground">
        모델: <span className="font-medium text-foreground">{result.model_label}</span>
        &nbsp;· 패치 수: <span className="font-medium text-foreground">{result.n_patches.toLocaleString()}</span>
      </p>
      <div>
        <p className="mb-2 text-xs font-semibold text-foreground">정량 지표</p>
        {keys.map((key) => {
          const val = metricVal(result, key);
          const unit = metricUnit(result, key);
          return (
            <Bar
              key={key}
              label={metricLabel(result, key)}
              value={val !== null ? val.toFixed(1) : "—"}
              suffix={unit}
              pct={val ?? 0}
            />
          );
        })}
      </div>
    </div>
  );
}

/** MT 모델 결과 — 2개 타겟. */
function MTPanel({ result }: { result: WsiAnalysisResult }) {
  const keys = ["fibrosisRatio", "atrophyRatio"];

  return (
    <div className="flex flex-col gap-4">
      <p className="text-[11px] text-muted-foreground">
        모델: <span className="font-medium text-foreground">{result.model_label}</span>
        &nbsp;· 패치 수: <span className="font-medium text-foreground">{result.n_patches.toLocaleString()}</span>
      </p>
      <div>
        <p className="mb-2 text-xs font-semibold text-foreground">정량 지표</p>
        {keys.map((key) => {
          const val = metricVal(result, key);
          const unit = metricUnit(result, key);
          return (
            <Bar
              key={key}
              label={metricLabel(result, key)}
              value={val !== null ? val.toFixed(1) : "—"}
              suffix={unit}
              pct={val ?? 0}
            />
          );
        })}
      </div>
    </div>
  );
}

/** PAS 모델 결과 — 3개 타겟(stain-agnostic ctranspath, STAIN_KEEP에 PAS 포함). */
function PASPanel({ result }: { result: WsiAnalysisResult }) {
  const keys = ["fibrosisRatio", "atrophyRatio", "inflammation"];
  return (
    <div className="flex flex-col gap-4">
      <p className="text-[11px] text-muted-foreground">
        모델: <span className="font-medium text-foreground">{result.model_label}</span>
        &nbsp;· 패치 수: <span className="font-medium text-foreground">{result.n_patches.toLocaleString()}</span>
      </p>
      <div>
        <p className="mb-2 text-xs font-semibold text-foreground">정량 지표</p>
        {keys.map((key) => {
          const val = metricVal(result, key);
          return (
            <Bar key={key} label={metricLabel(result, key)}
              value={val !== null ? val.toFixed(1) : "—"} suffix={metricUnit(result, key)}
              pct={val ?? 0} banff={metricBanff(result, key)} />
          );
        })}
      </div>
    </div>
  );
}

export function VunoResultPanel({
  result,
  selectedStain,
}: {
  result: WsiAnalysisResult | null;
  selectedStain?: WsiStain;
}) {
  const [model, setModel] = useState<ModelKey>("he");

  // 외부 stain 변경 시 내부 model 드롭다운 동기화
  useEffect(() => {
    if (!selectedStain) return;
    const key = selectedStain.toLowerCase() as ModelKey;
    if (key === "he" || key === "mt" || key === "pas") setModel(key);
  }, [selectedStain]);

  return (
    <div className="flex flex-col gap-4">
      {/* 분석 결과 없으면 안내, 있으면 stain별 결과 표시 */}
      {!result && (
        <EmptyState title="AI 분석 대기" description="슬라이드를 선택하고 AI 분석을 실행하면 결과가 표시됩니다." />
      )}

      {model === "he" && result && result.stain === "HE" && <HEPanel result={result} />}
      {model === "mt" && result && result.stain === "MT" && <MTPanel result={result} />}
      {model === "pas" && result && result.stain === "PAS" && <PASPanel result={result} />}

      {/* 선택 모델과 분석 결과 stain이 불일치할 때 */}
      {model !== "pas" && result && result.stain !== model.toUpperCase() && (
        <p className="text-[11px] text-muted-foreground">
          현재 분석 결과는 <strong>{result.stain}</strong> 모델입니다. 위에서 모델을 맞춰 선택하세요.
        </p>
      )}
    </div>
  );
}
