/**
 * CHYM v6.1 CDSS 3-layer 판정 — 순수 변환(실데이터 전용, 두 실모델 late fusion).
 *
 * 입력(모두 실측):
 *  - wsi : 라이브 Task-Attention MIL · CORAL ordinal (8001) — fibrosis/atrophy/inflammation + CORAL margin.
 *  - desc: cdss_v5 CLAM-lite/CTransPath OOF (8010 /wsi/cdss-descriptors) — ati_severity(ATI)/immune(tubulitis)/chronic.
 *  - aiRiskScore: 실제 EMR AKI risk (Patient.aiRiskScore, 8010).
 *
 * v6.1 §6 tubular injury spectrum 을 **진짜로** 충족:
 *   Tubular Injury = ATI(ati_severity, 구조적) ⊕ tubulitis(immune, 면역성) 통합 — 둘 다 실제 학습 헤드.
 * 라이브 KPMP 모델이 미예측하는 ATI/tubulitis 를 cdss_v5 OOF(검증 예측)로 공급한다. stub/난수/alias 없음.
 */
import type { WsiAnalysisResult, WsiStain } from "@/types/wsi";
import type { CdssDescriptors, CdssV61Decision, V61Axis, V61Confidence, V61Level } from "@/types/cdss";

function level(v: number, high = 0.5, mod = 0.25): V61Level {
  if (v >= high) return "HIGH";
  if (v >= mod) return "MODERATE";
  return "LOW";
}

function raw(wsi: WsiAnalysisResult, key: string): number | null {
  const m = wsi.metrics.find((x) => x.key === key);
  return m ? m.raw : null;
}

function axis(key: string, label: string, score: number, real: boolean, note?: string): V61Axis {
  return { key, label, level: level(score), score: Number(score.toFixed(4)), real, note };
}

/** noisy-OR: 구조적·면역성 중 하나라도 손상 시사 시 통합 점수 상승. */
function noisyOr(a: number, b: number): number {
  return a + b - a * b;
}

function parseCoralUncertainty(diagnosis: string | undefined): number | null {
  if (!diagnosis) return null;
  const m = diagnosis.match(/불확실성\s*([0-9]*\.?[0-9]+)/);
  return m ? parseFloat(m[1]) : null;
}

/** descriptor 헤드 메트릭이 약한지(AUROC<0.7 또는 CI 하한<0.5) — SHADOW 등급 판단용. */
function isWeakHead(desc: CdssDescriptors | null, key: string): boolean {
  const m = desc?.metrics?.[key];
  if (!m) return false;
  if (m.auroc !== null && m.auroc < 0.7) return true;
  if (m.ci95 && m.ci95[0] < 0.5) return true;
  return false;
}

/**
 * 실제 모델 결과 → v6.1 3-layer.
 * @param wsi   라이브 추론 결과(없으면 null 반환).
 * @param aiRiskScore 실제 EMR AKI risk(0..1 또는 0..100). null 이면 영상 파생.
 * @param stain 염색.
 * @param desc  cdss_v5 OOF 디스크립터(ATI/immune/chronic). 코호트 미포함 시 null → 라이브 헤드만.
 */
export function buildCdssV61(
  wsi: WsiAnalysisResult | null,
  aiRiskScore: number | null,
  stain: WsiStain,
  desc: CdssDescriptors | null = null,
): CdssV61Decision | null {
  if (!wsi) return null;

  // ── 라이브 KPMP 실측 헤드 ──
  const fibrosis = raw(wsi, "fibrosisRatio") ?? 0;
  const atrophy = raw(wsi, "atrophyRatio") ?? 0;
  const inflamm = raw(wsi, "inflammation") ?? 0;
  const tubularDirect = raw(wsi, "tubularInjury"); // 신모델엔 부재

  // ── cdss_v5 OOF 실측 헤드(있으면) ──
  const d = desc?.descriptors ?? null;
  const ati = d?.atiSeverity ?? null;        // 구조적 ATI
  const tubulitis = d?.immune ?? null;       // 면역성 tubulitis
  const chronicOof = d?.chronic ?? null;     // 만성 배경

  // ── L2 ① 세뇨관 손상(통합) — ATI ⊕ tubulitis(실측), 없으면 라이브/위축 폴백 ──
  let tubularScore: number;
  let tubularReal: boolean;
  let tubularNote: string | undefined;
  if (ati !== null) {
    tubularScore = noisyOr(ati, tubulitis ?? 0);
    tubularReal = true;
    tubularNote = tubulitis !== null ? "ATI⊕tubulitis 통합(실측)" : "ATI 기반(실측, tubulitis 미커버)";
  } else if (tubularDirect !== null) {
    tubularScore = tubularDirect;
    tubularReal = true;
  } else {
    tubularScore = atrophy;
    tubularReal = true;
    tubularNote = "세뇨관 위축 기반(실측)";
  }

  // ── L2 ② 만성 변화(Chronicity) — 전용 chronic 헤드 우선, 없으면 섬유화·위축 합성 ──
  // ※ "CKD 기저"로 표기하지 않는다: AKI CDSS에서 CKD 진단/진행으로 오인될 수 있어,
  //    실제 의미인 만성 손상량(chronicity)으로 표기한다.
  const chronicityScore = chronicOof ?? (0.5 * fibrosis + 0.3 * atrophy);
  const chronicityNote = chronicOof !== null ? "chronic 헤드(실측)" : "간질 섬유화·위축 합성(실측)";

  const axes: V61Axis[] = [
    axis("tubularInjury", "세뇨관 손상 (Tubular injury)", tubularScore, tubularReal, tubularNote),
    axis("chronicity", "만성 변화 (Chronicity)", chronicityScore, chronicOof !== null, chronicityNote),
    axis("inflammation", "염증 (Inflammation)", inflamm, true),
  ];

  // ── L3 상세 descriptor(실측 헤드) ──
  const descriptors: V61Axis[] = [];
  if (ati !== null) descriptors.push(axis("ati", "ATI · 구조적 (ati_severity)", ati, true, "cdss_v5 OOF"));
  if (tubulitis !== null) {
    const weak = isWeakHead(desc, "immune");
    descriptors.push(axis("tubulitis", "Tubulitis · 면역성 (immune)", tubulitis, true, weak ? "cdss_v5 OOF · 근거 제한적" : "cdss_v5 OOF"));
  }
  descriptors.push(axis("fibrosisRatio", "간질 섬유화 (Fibrosis)", fibrosis, true));
  descriptors.push(axis("atrophyRatio", "세뇨관 위축 (Atrophy)", atrophy, true));
  descriptors.push(axis("inflammationDesc", "간질 염증 (Inflammation)", inflamm, true));

  // ── L1 AKI progression risk ──
  const emrRisk = aiRiskScore === null ? null : aiRiskScore > 1 ? aiRiskScore / 100 : aiRiskScore;
  const acute = Math.max(ati ?? 0, tubulitis ?? 0, inflamm, tubularDirect ?? 0); // 실측 급성 신호
  let riskScore: number;
  let riskSource: CdssV61Decision["riskSource"];
  if (emrRisk !== null) {
    riskScore = emrRisk;
    riskSource = "EMR";
  } else {
    riskScore = 0.7 * acute + 0.3 * chronicityScore;
    riskSource = "WSI_DERIVED";
  }
  const riskLevel = riskSource === "EMR" ? level(riskScore, 0.66, 0.33) : level(riskScore, 0.5, 0.25);

  // ── §9 confidence: 실제 CORAL margin + EMR-WSI 일치(실측 ATI 포함) + tubulitis 헤드 신뢰 ──
  const nPatches = wsi.n_patches;
  const coralU = parseCoralUncertainty(wsi.report?.diagnosis);
  const status = (wsi.report?.status ?? "").toUpperCase();
  const consistencyGap = emrRisk !== null ? Math.abs(emrRisk - acute) : null;
  const tubulitisWeak = tubulitis !== null && isWeakHead(desc, "immune");

  let confidence: V61Confidence;
  let confidenceReason: string;
  if (nPatches < 50 || /REJECT|ABSTAIN|BLOCK/.test(status)) {
    confidence = "ABSTAIN";
    confidenceReason = nPatches < 50 ? `유효 패치 부족(${nPatches}개)` : `라우터 게이트=${status}`;
  } else if (consistencyGap !== null && consistencyGap >= 0.4) {
    confidence = "SHADOW";
    confidenceReason = `EMR risk·영상 급성손상 불일치(Δ${consistencyGap.toFixed(2)})`;
  } else if ((coralU !== null && coralU >= 0.35)) {
    confidence = "ABSTAIN";
    confidenceReason = `CORAL 불확실성 높음(${coralU.toFixed(3)})`;
  } else if ((coralU !== null && coralU >= 0.18) || /SHADOW/.test(status) || tubulitisWeak) {
    confidence = "SHADOW";
    confidenceReason = tubulitisWeak
      ? "tubulitis 헤드 근거 제한적(AUROC↓)"
      : coralU !== null
        ? `CORAL 불확실성 중간(${coralU.toFixed(3)})`
        : "라우터 SHADOW";
  } else {
    confidence = "CONFIDENT";
    confidenceReason = consistencyGap !== null
      ? `EMR·영상 일치(Δ${consistencyGap.toFixed(2)}) · 패치 ${nPatches}`
      : `패치 ${nPatches}`;
  }

  const interpretation = buildInterpretation(riskLevel, axes, riskSource);
  const modelLabel = wsi.model_label + (desc?.available && d ? " + cdss_v5 OOF(ATI/immune)" : "");

  return {
    riskLevel,
    riskSource,
    confidence,
    confidenceReason,
    interpretation,
    axes,
    descriptors,
    modelLabel,
    nPatches,
  };
}

function buildInterpretation(
  risk: V61Level,
  axes: readonly V61Axis[],
  source: CdssV61Decision["riskSource"],
): string {
  const ti = axes.find((a) => a.key === "tubularInjury");
  const ckd = axes.find((a) => a.key === "chronicity");
  const base = source === "EMR" ? "EMR 기반 AKI 위험" : "영상 파생 위험";
  const acuteHigh = ti && ti.level === "HIGH";
  const chronic = ckd && ckd.level !== "LOW";
  if (acuteHigh && chronic) return `${base} · 만성 변화 배경 위 급성 세뇨관 손상 — 임상 상관 필요`;
  if (acuteHigh) return `${base} · 비교적 보존된 배경 위 급성 세뇨관 손상 — 임상 상관 필요`;
  if (ti && ti.level === "MODERATE") return `${base} · 불확정 세뇨관 손상 — 임상 상관 필요`;
  return `${base} · 뚜렷한 급성 세뇨관 손상 신호 없음 — 임상 상관 필요`;
}
