"""
CDSS Engine v2 — 비진단 임상 의사결정 보조 + 리포트 생성 엔진 (4단)

STEP1 QC(rule-based 게이트) → STEP2 Router(CORAL margin 위험도) →
STEP3 Inference strategy(조건부 ensemble: HIGH_RISK=ensemble ON, LOW_RISK=latency/batch 따라) →
STEP4 Clinical Report(QC상태·위험도·전략·예측·신뢰도설명·소견요약·stain기여·한계·권고).

설계 정합성(확정):
- **QC = 학습 head 아님, 규칙기반 측정 모듈**(QC 라벨 없음→supervision 불가).
- **Router = CORAL margin 단독**(router_exp.py에서 selection 작동 확인). **MC-dropout/entropy 금지.**
- **Ensemble = variance 안정화**(5-seed, 예측 QWK 0.37→0.48). 예측 정확도용.
- 학습 대상은 ordinal head 뿐. 병리 범위 = 신성(intrinsic) 내부 소견 정량(fibrosis/atrophy/inflammation).
⚠️ SHADOW 전용·탐색적(QWK~0.48, N≈60, 외부검증0). 확정 판정/치료 제안 금지. abstain-first.

사용:
  학습+저장:  python cdss_engine.py --fit
  데모(추론): python cdss_engine.py --demo
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# 패키지 루트(이 파일 기준) — 코드/모델/설정은 패키지와 함께 이동해도 따라옴(절대경로 하드코딩 제거).
PKG_ROOT = Path(__file__).resolve().parent.parent          # .../Pathology_model
sys.path.insert(0, str(PKG_ROOT))
from mil.train import ROOT, SEED, load_bags, seed_all        # ROOT=데이터 루트(코드와 분리)

ORD = ["fibrosis", "atrophy", "inflammation"]
KBINS = 4
SEEDS = [42, 43, 44, 45, 46]   # 5-seed deep ensemble
BANFF = {"fibrosis": [5, 25, 50], "atrophy": [5, 25, 50], "inflammation": [10, 25, 50]}
STAIN_KEEP = {"HE", "PAS", "MT", "SILVER"}
REQUIRED_STAINS = {"HE", "PAS"}
MODEL_PATH = PKG_ROOT / "models/cdss_shadow/ordinal_ms.pt"          # baseline(원본·fallback)
LN_PATH = PKG_ROOT / "models/cdss_shadow/ordinal_ms_ln.pt"          # LayerNorm 후보(+temperature)
PAS_PATH = PKG_ROOT / "models/cdss_shadow/ordinal_pas.pt"           # PAS 단일 stain 변종(멀티 융합 제거)


def _resolve_model():
    """A/B 선택 — env CDSS_MODEL_VARIANT(ln|baseline|pas). 기본 ln(파일 있으면), 없으면 baseline.
    pas=PAS 단일 stain 변종. 로드 실패 시 __init__ 에서 baseline fallback. 롤백=env 한 줄로 즉시."""
    v = os.environ.get("CDSS_MODEL_VARIANT", "ln").lower()
    if v == "pas" and PAS_PATH.exists():
        return PAS_PATH, "pas"
    if v == "ln" and LN_PATH.exists():
        return LN_PATH, "ln"
    if v == "pas" and LN_PATH.exists():          # pas 파일 없으면 ln 로 폴백
        return LN_PATH, "ln"
    return MODEL_PATH, "baseline"
AUDIT = PKG_ROOT / "results/09_cdss_shadow/cdss_engine.jsonl"
BANNED = ["확진", "거부반응", "rejection", "diagnosis", "진단 결과", "진단됨", "처방", "치료 결정"]
# Router 임계 — config.json[router](calibrate_router.py가 risk-coverage로 보정)에서 로드, 없으면 기본값
_RT = json.loads((PKG_ROOT / "config.json").read_text(encoding="utf-8")).get("router", {})
U_ABSTAIN = _RT.get("uncertainty_abstain", 0.70)
QC_ABSTAIN = _RT.get("qc_abstain", 0.40)
ORD_CONF_MIN = _RT.get("ord_conf_min", 0.40)


def to_ord(v, edges):
    return np.nan if v != v else float(sum(v > e for e in edges))


def _build_labels():
    import pandas as pd
    sm = pd.read_csv(ROOT / "Pathology_model/artifacts/split_manifest.csv").drop_duplicates("patient_id")
    sm["patient_id"] = sm["patient_id"].astype(str)
    desc = pd.read_csv(ROOT / "Pathology_model/artifacts/descriptor_labels.csv")
    desc["patient_id"] = desc["patient_id"].astype(str)
    src = {"fibrosis": "interstitial_fibrosis_pct", "atrophy": "tubular_atrophy",
           "inflammation": "interstitial_mononuclear_wbc_pct"}
    lab = pd.DataFrame({"patient_id": desc["patient_id"]})
    for t, c in src.items():
        v = pd.to_numeric(desc[c], errors="coerce"); v = v.where(v < 999)
        lab[t] = v.map(lambda x: to_ord(x, BANFF[t]))
    return lab


def _new_model(embed_dim, layernorm=False):
    from mil.model import TaskAttentionMIL, STAINS
    m = TaskAttentionMIL(in_dim=embed_dim, silver_mode="prediction",
                         tasks=ORD, out_dims={t: KBINS - 1 for t in ORD})
    if layernorm:                       # LN 후보: encoder 에 LayerNorm 추가(dim 은 원 구조에서 도출)
        import torch.nn as nn
        d = m.encoders[STAINS[0]][0].out_features
        m.encoders = nn.ModuleDict(
            {s: nn.Sequential(nn.Linear(embed_dim, d), nn.ReLU(), nn.Dropout(0.25), nn.LayerNorm(d))
             for s in STAINS})
    return m


# ---------------- 학습+저장 (전체 코호트, 시각화/배포용 단일 모델) ----------------
def fit():
    import torch, torch.nn.functional as F, pandas as pd
    seed_all(SEED); device = "cuda" if torch.cuda.is_available() else "cpu"
    idx = pd.read_csv(ROOT / "data/embeddings/ctranspath/index.csv", dtype={"magnification": str})
    idx = idx[idx["magnification"].isin(["10", "40"]) & idx["stain"].isin(STAIN_KEEP)]
    embed_dim = int(idx["embed_dim"].iloc[0])
    lab = _build_labels()
    bags = load_bags(lab["patient_id"].tolist(), idx, keep=STAIN_KEEP)
    lab = lab[lab["patient_id"].isin(bags) & lab[ORD].notna().any(axis=1)].reset_index(drop=True)

    def train_seed(seed):
        torch.manual_seed(seed); np.random.seed(seed)
        model = _new_model(embed_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        for ep in range(40):
            model.train()
            for r in lab.sample(frac=1, random_state=seed + ep).itertuples(index=False):
                rd = dict(zip(lab.columns, r))
                bag = {s: torch.from_numpy(v).to(device) for s, v in bags[rd["patient_id"]].items()}
                out = model(bag); loss = 0.0; nt = 0
                for t in ORD:
                    yv = rd[t]
                    if yv == yv:
                        lv = torch.tensor([1.0 if yv > k else 0.0 for k in range(KBINS - 1)], device=device)
                        loss = loss + F.binary_cross_entropy_with_logits(out[t], lv); nt += 1
                if nt:
                    opt.zero_grad(); loss.backward(); opt.step()
        return {k: v.cpu() for k, v in model.state_dict().items()}

    ensemble = [train_seed(s) for s in SEEDS]   # 5-seed deep ensemble (예측 QWK 0.37→0.48)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ensemble": ensemble, "seeds": SEEDS, "embed_dim": embed_dim,
                "tasks": ORD, "kbins": KBINS, "banff": BANFF,
                "trained": datetime.now().isoformat(timespec="seconds"),
                "note": "SHADOW only; exploratory(QWK~0.48,N61); 5-seed ensemble + CORAL margin router"},
               MODEL_PATH)
    print(f"저장 -> {MODEL_PATH} (cohort={len(lab)}, embed_dim={embed_dim}, ensemble={len(ensemble)})")


class CdssEngine:
    def __init__(self, device="cpu"):
        # Intent(requested) vs Execution(resolved) 분리 — fallback 시 로그/캐시가 실제 로드모델을 반영.
        self.device = device
        req = os.environ.get("CDSS_MODEL_VARIANT", "ln").lower()
        self.requested_variant = req if req in ("ln", "baseline", "pas") else "ln"
        path, variant = _resolve_model()                     # 1차: 파일 존재 기준
        try:
            self._load(path, variant, device)                # torch.load + 구성 + state_dict 전체
            same = self.model_variant == self.requested_variant
            self.load_status = "ok" if same else "fallback"
            self.load_reason = "" if same else "ln_checkpoint_missing"
        except Exception as e:                               # LN 로드/구성 실패 → baseline(원본) fallback
            self._load(MODEL_PATH, "baseline", device)
            self.load_status = "fallback"
            self.load_reason = f"ln_load_failed:{type(e).__name__}"

    def _load(self, path, variant, device):
        import torch
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.model_variant = variant                         # resolved(실제 로드된 모델)
        self.model_path = str(path)
        self.temperature = {t: float(ckpt.get("temperature", {}).get(t, 1.0)) for t in ORD}
        layernorm = bool(ckpt.get("layernorm", False))
        # 단일 stain 변종(ckpt["stains"])이면 QC 필수/집계 stain 을 그걸로. 없으면 기존 멀티(불변).
        _st = set(ckpt.get("stains") or [])
        self.required_stains = _st or REQUIRED_STAINS
        self._qc_denom = _st or STAIN_KEEP
        sds = ckpt.get("ensemble") or [ckpt["state_dict"]]   # 신규 ensemble / 구형 단일 호환
        self.models = []
        for sd in sds:
            m = _new_model(ckpt["embed_dim"], layernorm=layernorm).to(device)
            m.load_state_dict(sd); m.eval(); self.models.append(m)

    @property
    def variant_info(self) -> dict:
        """Intent/Execution 상태 — 캐시 키·A/B 로그·리포트의 단일 소스."""
        return {"requested": self.requested_variant, "resolved": self.model_variant,
                "status": getattr(self, "load_status", "ok"), "reason": getattr(self, "load_reason", "")}

    @property
    def model_stains(self) -> set:
        """이 모델이 학습·사용하는 stain 집합(멀티=STAIN_KEEP, pas=단일 {PAS}). adapter 의 bag 필터용."""
        return getattr(self, "_qc_denom", STAIN_KEEP)

    def _qc(self, bag):
        """QC Module (rule-based / deterministic) — 학습 아님. signal integrity 측정.
        qc = 0.3·tissue_coverage + 0.2·patch_density + 0.2·stain_completeness
             + 0.2·embedding_stability + 0.1·numeric_sanity
        """
        present = {s for s, v in bag.items() if v is not None and v.shape[0] > 0}
        if not present:                                       # [infer-single] 표시 슬라이드 stain 1개(피처)만 있으면 통과
            return 0.0, "stain 피처 없음"
        allv = np.concatenate([v for v in bag.values() if v is not None and v.shape[0] > 0], 0)
        numeric_sanity = 1.0 if np.isfinite(allv).all() else 0.0
        if numeric_sanity == 0:
            return 0.0, "임베딩 NaN/Inf(초점/아티팩트 의심)"
        total = sum(v.shape[0] for v in bag.values() if v is not None)
        tissue_coverage = min(1.0, total / 300.0)             # 조직 샘플 충분도
        patch_density = float(np.mean([min(1.0, (bag[s].shape[0] / 20.0))
                                       for s in present & STAIN_KEEP]))  # stain별 패치 충실도
        stain_completeness = 1.0 if present else 0.0          # [infer-single] 표시 슬라이드 1장이면 완비(멀티 4-stain 페널티 제거)
        norm = float(np.linalg.norm(allv, axis=1).mean())
        embedding_stability = 1.0 if 0.1 <= norm <= 100 else 0.0
        score = round(0.3 * tissue_coverage + 0.2 * patch_density + 0.2 * stain_completeness
                      + 0.2 * embedding_stability + 0.1 * numeric_sanity, 3)
        return score, (f"cov={tissue_coverage:.2f},dens={patch_density:.2f},"
                       f"stain={stain_completeness:.2f},emb={embedding_stability},num={numeric_sanity}")

    def _predict(self, bag_np, use_ensemble: bool):
        """CORAL 등급 + margin 불확실성 + stain 기여. use_ensemble면 5-seed 누적확률 평균.
        router/예측은 CORAL margin만 사용(MC-dropout/entropy 금지)."""
        import torch
        models = self.models if use_ensemble else self.models[:1]
        bag = {s: torch.from_numpy(v).to(self.device) for s, v in bag_np.items()
               if v is not None and v.shape[0] > 0}
        grades, margins, contrib, calconf = {}, {}, {}, {}
        with torch.no_grad():
            outs = [m(bag) for m in models]
            for t in ORD:
                probs = np.mean([torch.sigmoid(o[t]).cpu().numpy() for o in outs], axis=0)  # RAW: 등급/margin/router 운영점 보존
                grades[t] = int((probs > 0.5).sum())
                margins[t] = float(np.mean(1.0 - np.abs(2 * probs - 1.0)))  # 0.5 근접=불확실
                T = self.temperature.get(t, 1.0)              # temperature 는 '보고용 보정 신뢰도'에만(router 미변경 → 과-abstain 방지)
                pc = np.mean([torch.sigmoid(o[t] / T).cpu().numpy() for o in outs], axis=0)
                calconf[t] = round(float(pc.max()), 3)        # 보정된 최상위 임계확률(신뢰도 보고)
                c = outs[0].get("task_stain_contrib_norm", {}).get(t, {})
                if c:
                    contrib[t] = max(c, key=c.get)            # 최대 기여 stain(설명)
        return grades, margins, contrib, calconf

    # ---- STEP 3: inference strategy (조건부 ensemble) ----
    @staticmethod
    def _strategy(risk: str, latency_critical: bool, batch: bool):
        if risk == "HIGH_RISK":
            return "SELECTIVE"          # variance 안정화 위해 ensemble ON
        if latency_critical:
            return "OFF"                # 단일 모델
        if batch:
            return "FULL"              # offline 배치 → 5-seed
        return "OFF"                    # SELECTIVE default OFF

    def analyze(self, bag_np, *, slide_id="unknown", latency_critical=False, batch=False):
        # STEP 1: QC gate
        qc, qc_note = self._qc(bag_np)
        if qc <= QC_ABSTAIN:
            return self._report(decision="ABSTAIN", reason="QC_FAIL", qc=qc, qc_note=qc_note,
                                slide_id=slide_id)
        # STEP 2: Router — 단일 모델 margin으로 위험도 라우팅
        _, m0, _, _ = self._predict(bag_np, use_ensemble=False)
        margin0 = float(np.mean(list(m0.values())))
        risk = "HIGH_RISK" if margin0 >= U_ABSTAIN else "LOW_RISK"
        # STEP 3: strategy → ensemble 여부
        strategy = self._strategy(risk, latency_critical, batch)
        use_ens = strategy in ("FULL", "SELECTIVE")
        grades, margins, contrib, calconf = self._predict(bag_np, use_ensemble=use_ens)
        unc = round(float(np.mean(list(margins.values()))), 3)
        # HIGH_RISK이고 ensemble 후에도 margin 과대 → 추가 abstain(안전)
        if risk == "HIGH_RISK" and unc >= U_ABSTAIN + 0.15:
            return self._report(decision="ABSTAIN", reason="HIGH_UNCERTAINTY", qc=qc, qc_note=qc_note,
                                risk=risk, strategy=strategy, unc=unc, slide_id=slide_id)
        # STEP 4: clinical report
        return self._report(decision="ALLOW", qc=qc, qc_note=qc_note, risk=risk, strategy=strategy,
                            unc=unc, grades=grades, contrib=contrib, calconf=calconf, slide_id=slide_id)

    # ---- STEP 4: clinical report generation (비진단·보조) ----
    def _report(self, *, decision, qc, qc_note, slide_id, reason=None, risk=None,
                strategy=None, unc=None, grades=None, contrib=None, calconf=None):
        rep = {"mode": "SHADOW", "decision": decision,
               "modelVariant": getattr(self, "model_variant", "baseline"),   # resolved(실제 로드) — A/B 로그
               "modelRequested": getattr(self, "requested_variant", "ln"),   # intent(요청)
               "modelLoadStatus": getattr(self, "load_status", "ok"),        # ok | fallback
               "qcStatus": "PASS" if qc > QC_ABSTAIN else "FAIL", "qcScore": qc,
               "riskStratification": risk, "modelStrategy": strategy,
               "ensemble": "ON" if strategy in ("FULL", "SELECTIVE") else "OFF"}
        if decision == "ABSTAIN":
            rep["abstainReason"] = reason
            rep["uncertaintyLimitation"] = (
                f"QC 미달({qc_note})로 보고 보류." if reason == "QC_FAIL"
                else f"CORAL margin 불확실성 {unc} 과대 → 보류.")
            rep["clinicalRecommendation"] = "신뢰도 부족 — 병리의 직접 검토 필요(재촬영/재염색 고려)."
            rep["finalPrediction"] = {t: None for t in ORD}
        else:
            rep["finalPrediction"] = grades                      # Banff 0–3 (신성 내부 소견 정량)
            rep["uncertainty"] = unc
            rep["confidenceInterpretation"] = (                  # margin 기반(entropy 금지)
                f"CORAL margin 불확실성 {unc} (낮을수록 확신). 신뢰수준 "
                + ("상대적 높음" if unc < 0.2 else "중간" if unc < U_ABSTAIN else "낮음"))
            rep["pathologyFindingsSummary"] = self._findings(grades)
            rep["stainContribution"] = contrib                   # task별 최대기여 stain
            rep["calibratedConfidence"] = calconf                # temperature 보정 P(등급 임계) — 보고용(router 미영향)
            rep["uncertaintyLimitation"] = (
                "탐색적·임상 미검증(QWK~0.48, N≈60, 외부검증 없음). "
                + (f"고위험 라우팅({risk})." if risk == "HIGH_RISK" else ""))
            rep["clinicalRecommendation"] = self._recommend(grades, risk)
        rep["disclaimer"] = "비진단 보조 정보(확정적 판정 아님). 최종 판단은 병리의/신장내과."
        blob = json.dumps(rep, ensure_ascii=False)
        hit = [w for w in BANNED if w in blob]
        if hit:
            raise ValueError(f"비진단 표현 위반{hit}")
        AUDIT.parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIT, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"),
                                "slideId": slide_id, **rep}, ensure_ascii=False) + "\n")
        return rep

    @staticmethod
    def _findings(grades):
        lv = {0: "없음/경미", 1: "경도", 2: "중등도", 3: "고도"}
        name = {"fibrosis": "간질 섬유화", "atrophy": "세뇨관 위축", "inflammation": "간질 염증"}
        return [f"{name[t]} {lv.get(g, '?')} 가능성(Banff {g})" for t, g in grades.items()]

    @staticmethod
    def _recommend(grades, risk):
        ci, ct, i = grades.get("fibrosis", 0), grades.get("atrophy", 0), grades.get("inflammation", 0)
        recs = []
        if ci >= 2 and ct >= 2:
            recs.append("만성 비가역 손상(섬유화+위축) 동반 가능성 → 예후/acute-on-chronic 관점 신장내과 검토")
        if i >= 2:
            recs.append("활동성 간질 염증 패턴 가능성 → 약물성/면역성(AIN) 등 임상 상관 권고")
        if risk == "HIGH_RISK":
            recs.append("불확실성 높음 → 추가 검토 또는 재검 고려")
        recs.append("현재 결과는 보조적 참고용(SHADOW) — 병리의 최종 판독 필요")
        return recs


def demo():
    import pandas as pd, torch  # noqa
    idx = pd.read_csv(ROOT / "data/embeddings/ctranspath/index.csv", dtype={"magnification": str})
    idx = idx[idx["magnification"].isin(["10", "40"]) & idx["stain"].isin(STAIN_KEEP)]
    eng = CdssEngine()
    lab = _build_labels()
    bags = load_bags(lab["patient_id"].tolist(), idx, keep=STAIN_KEEP)  # 보유 환자만 resolve
    pids = [p for p in lab["patient_id"] if p in bags][:3]
    print(f"(보유 환자 {len(bags)}명 / ensemble {len(eng.models)} 모델)")
    print("=== 정상 환자 임상 리포트(v2, batch=True) ===")
    for pid in pids:
        r = eng.analyze(bags[pid], slide_id=pid, batch=True)
        print(f"\n[{pid}] {json.dumps(r, ensure_ascii=False, indent=1)}")
    print("\n=== QC 미달(PAS 누락) → ABSTAIN, 리포트 미생성 ===")
    one = next(b for b in bags.values() if b.get("HE") is not None)
    print(json.dumps(eng.analyze({"HE": one["HE"]}, slide_id="noPAS"), ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.fit:
        fit()
    if a.demo:
        demo()
    if not (a.fit or a.demo):
        print("--fit 또는 --demo")
