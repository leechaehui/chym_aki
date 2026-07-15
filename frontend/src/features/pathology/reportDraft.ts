/**
 * 표준화 병리 보고서 초안 생성 — **실제 ABMIL/CDSS 모델이 예측하는 descriptor 만** 사용한다.
 *
 * 실제 모델 출력(backend api/wsi.py TARGET_META · mil_model.py):
 *   HE : fibrosisRatio·atrophyRatio·tubularInjury·inflammation(%) + artHyalinosis(0~3)
 *   MT : fibrosisRatio·atrophyRatio(%) + artHyalinosis(0~3)
 *   PAS: fibrosisRatio·atrophyRatio·inflammation(%)
 *
 * 모델이 평가하지 않는 항목(사구체 개수·global sclerosis·tubulitis(t)·vasculitis(v)·
 * ptc·cg·mm 등)은 **지어내지 않고 소견 문구에서 아예 제외**한다(placeholder 문구 미기재).
 * → 실제로 정량된 구획(Tubules·Interstitium·Vessels)만 MICROSCOPIC 블록에 표시된다.
 * %→Banff 점수는 표준 컷오프(ci/ct/i: <6→0, 6–25→1, 26–50→2, >50→3)를 따른다.
 */
import type { WsiAnalysisResult, WsiStain } from "@/types/wsi";

type StainResults = Partial<Record<WsiStain, WsiAnalysisResult>>;

const STAIN_LABEL: Record<WsiStain, string> = { HE: "H&E", MT: "MT", PAS: "PAS" };

/** 영문 중증도(점수 인덱스 0~3) — 병리 보고서 표준 표현. */
const SEV_EN = ["No significant", "Mild", "Moderate", "Severe"] as const;

interface Band {
  score: number;
  /** 표준 Banff 컷오프에 따라 판정된 중증도(영문). */
  word: (typeof SEV_EN)[number];
}

/** 표준 Banff ci/ct/i 스타일: 침범 면적(%) → 점수 0~3. */
function bandFromPercent(pct: number): Band {
  if (pct < 6) return { score: 0, word: SEV_EN[0] };
  if (pct <= 25) return { score: 1, word: SEV_EN[1] };
  if (pct <= 50) return { score: 2, word: SEV_EN[2] };
  return { score: 3, word: SEV_EN[3] };
}

/** artHyalinosis 등 0~3 연속 점수 → 점수 0~3(반올림 밴드). */
function bandFromScore(v: number): Band {
  if (v < 0.5) return { score: 0, word: SEV_EN[0] };
  if (v < 1.5) return { score: 1, word: SEV_EN[1] };
  if (v < 2.5) return { score: 2, word: SEV_EN[2] };
  return { score: 3, word: SEV_EN[3] };
}

/** descriptor 별 신뢰 stain 우선순위(콜라겐=삼색/PAS 우선 등). */
const PREFER: Record<string, WsiStain[]> = {
  fibrosisRatio: ["MT", "PAS", "HE"],
  atrophyRatio: ["PAS", "HE", "MT"],
  tubularInjury: ["HE"],
  inflammation: ["HE", "PAS"],
  artHyalinosis: ["HE", "MT"],
};

interface Picked {
  value: number;
  stain: WsiStain;
}

/** 우선 stain 순서대로 해당 descriptor 값을 찾는다(없으면 null). */
function pick(res: StainResults, key: string): Picked | null {
  for (const s of PREFER[key] ?? (["HE", "MT", "PAS"] as WsiStain[])) {
    const m = res[s]?.metrics.find((mm) => mm.key === key);
    if (m && m.value !== null && m.value !== undefined && Number.isFinite(m.value)) {
      return { value: m.value, stain: s };
    }
  }
  return null;
}

export interface DraftInput {
  patientName: string;
  patientMrn: string;
  results: StainResults;
}

export interface DraftOutput {
  findings: string;
  diagnosis: string;
}

/**
 * 실제 모델 descriptor → 표준 병리 보고서 초안(소견 / 진단+Banff).
 * 소견 = Specimen·Gross·Microscopic(구획별), 진단 = Diagnosis·Banff·Comment.
 */
export function buildStandardizedReport({ patientName, patientMrn, results }: DraftInput): DraftOutput {
  const fib = pick(results, "fibrosisRatio");
  const atr = pick(results, "atrophyRatio");
  const ati = pick(results, "tubularInjury");
  const inf = pick(results, "inflammation");
  const ah = pick(results, "artHyalinosis");

  const ci = fib ? bandFromPercent(fib.value) : null; // interstitial fibrosis
  const ct = atr ? bandFromPercent(atr.value) : null; // tubular atrophy
  const iB = inf ? bandFromPercent(inf.value) : null; // interstitial inflammation
  const atiB = ati ? bandFromPercent(ati.value) : null; // acute tubular injury
  const ahB = ah ? bandFromScore(ah.value) : null; // arteriolar hyalinosis

  const src = (p: Picked | null) => (p ? ` [${STAIN_LABEL[p.stain]}]` : "");
  const pct = (p: Picked | null) => (p ? `~${p.value.toFixed(0)}%` : "");

  // ── 구획별 현미경 소견 — 모델이 실제로 정량한 항목만 기재(미정량 구획은 문구에서 제외) ──
  const tubules: string[] = [];
  if (atiB) {
    tubules.push(
      atiB.score === 0
        ? `No significant acute tubular injury${src(ati)}.`
        : `${atiB.word} acute tubular injury (${pct(ati)})${src(ati)}.`,
    );
  }
  if (ct) tubules.push(`${ct.word} tubular atrophy (Banff ct ${ct.score}/3, ${pct(atr)})${src(atr)}.`);

  const interstitium: string[] = [];
  if (ci) interstitium.push(`${ci.word} interstitial fibrosis (Banff ci ${ci.score}/3, ${pct(fib)})${src(fib)}.`);
  if (iB) {
    interstitium.push(
      iB.score === 0
        ? `No significant interstitial inflammation${src(inf)}.`
        : `${iB.word} interstitial inflammation (Banff i ${iB.score}/3, ${pct(inf)})${src(inf)}.`,
    );
  }

  const vessels = ahB
    ? ahB.score === 0
      ? `No significant arteriolar hyalinosis${src(ah)}.`
      : `Arteriolar hyalinosis (Banff ah ${ahB.score}/3)${src(ah)}.`
    : null;

  // 정량된 구획만 MICROSCOPIC 블록에 포함(Glomeruli·Gross 등 모델 미정량 항목은 제외).
  const micro: string[] = [];
  if (tubules.length > 0) micro.push(`Tubules: ${tubules.join(" ")}`);
  if (interstitium.length > 0) micro.push(`Interstitium: ${interstitium.join(" ")}`);
  if (vessels) micro.push(`Vessels: ${vessels}`);

  const findings = [
    `SPECIMEN: Kidney, needle biopsy — ${patientName} (${patientMrn})`,
    ...(micro.length > 0 ? [``, `MICROSCOPIC DESCRIPTION`, ...micro] : []),
  ].join("\n");

  // ── 진단 + Banff + Comment ──────────────────────────────────────────────────
  const dxLines: string[] = [];
  if (ci || ct) {
    const iftaScore = Math.max(ci?.score ?? 0, ct?.score ?? 0);
    dxLines.push(`- ${SEV_EN[iftaScore]} interstitial fibrosis and tubular atrophy (IFTA), ${pct(fib ?? atr)}`);
  }
  if (atiB && atiB.score > 0) dxLines.push(`- ${atiB.word} acute tubular injury`);
  if (iB && iB.score > 0) dxLines.push(`- ${iB.word} interstitial inflammation`);
  if (ahB && ahB.score > 0) dxLines.push(`- Arteriolar hyalinosis`);
  if (dxLines.length === 0) dxLines.push(`- No significant AI-quantified chronic/acute injury`);

  // Banff 점수는 모델이 실제로 낸 축(ci·ct·i·ah)만 기재.
  const banff: string[] = [];
  if (ci) banff.push(`ci${ci.score}`);
  if (ct) banff.push(`ct${ct.score}`);
  if (iB) banff.push(`i${iB.score}`);
  if (ahB) banff.push(`ah${ahB.score}`);

  const models = Array.from(
    new Set((["HE", "MT", "PAS"] as WsiStain[]).map((s) => results[s]?.model_label).filter(Boolean)),
  ).join(" · ");

  const diagnosis = [
    `DIAGNOSIS — Kidney, needle biopsy:`,
    ...dxLines,
    ``,
    `BANFF SCORES (AI 정량 기반 · 잠정): ${banff.join(" ") || "N/A"}`,
    `(ci=간질 섬유화, ct=세뇨관 위축, i=간질 염증, ah=세동맥 유리질화 · 각 0~3점)`,
    `(g·t·v·ptc·cg·mm 등은 본 AI 모델이 평가하지 않음 — 병리 전문의 확인 필요)`,
    ``,
    `COMMENT`,
    `AI 정량 분석(${models || "ABMIL"}) 기반 초안입니다. AI는 핵심 소견의 정량 평가를 보조하며,`,
    `최종 병리 진단은 병리 전문의가 현미경 소견과 임상 정보를 종합해 확정합니다. 검토·승인 후 판독 완료하세요.`,
  ].join("\n");

  return { findings, diagnosis };
}
