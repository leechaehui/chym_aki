"""배율 train/serve 스큐 실측 — CdssEngine 을 T(10+40, 학습배율) vs S(20, 서빙배율)로 비교.

배율 변수만 격리(같은 인코더 ctranspath·같은 stain norm·같은 top-k·같은 모델). 기존
data_processed/embeddings/ctranspath 임베딩에 10/20/40x 가 이미 있어 추출 불필요.

질문: 서빙이 학습에 없던 20x 를 넣으면 CdssEngine 예측이 얼마나 달라지고,
      정답(gold Banff ordinal) 대비 정확도가 실제로 떨어지나?

출력: 콘솔 요약 + scratchpad json. AUDIT jsonl 은 임시 경로로 리다이렉트(운영 오염 방지).
사용: python -m mil.diag_mag_serve_skew
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mil.cdss_engine as ce
from mil.cdss_engine import CdssEngine, STAIN_KEEP, ORD, BANFF, to_ord
from mil.train import ROOT, load_bags
from mil.cdss_paths import p as _p

OUT = Path(r"C:\Users\301-4\AppData\Local\Temp\claude\C--team-chym-aki\427e56b3-dd2b-4697-997b-433d9ab3f226\scratchpad")
REPO = Path(r"C:\team\chym_aki")   # cdss_engine._build_labels 는 stale ROOT(repo가정) 사용 → 여기서 직접


def _build_labels():
    """gold Banff ordinal (cdss_engine._build_labels 를 올바른 repo-root 경로로 재현)."""
    desc = pd.read_csv(REPO / "Pathology_model/artifacts/descriptor_labels.csv")
    desc["patient_id"] = desc["patient_id"].astype(str)
    src = {"fibrosis": "interstitial_fibrosis_pct", "atrophy": "tubular_atrophy",
           "inflammation": "interstitial_mononuclear_wbc_pct"}
    lab = pd.DataFrame({"patient_id": desc["patient_id"]})
    for t, c in src.items():
        v = pd.to_numeric(desc[c], errors="coerce"); v = v.where(v < 999)
        lab[t] = v.map(lambda x: to_ord(x, BANFF[t]))
    return lab


def qwk(y, p, K=4):
    """quadratic weighted kappa (0..K-1)."""
    y = np.asarray(y, int); p = np.asarray(p, int)
    O = np.zeros((K, K))
    for a, b in zip(y, p):
        O[a, b] += 1
    w = np.array([[(i - j) ** 2 / (K - 1) ** 2 for j in range(K)] for i in range(K)])
    act = O.sum(1); pred = O.sum(0); E = np.outer(act, pred) / O.sum()
    denom = (w * E).sum()
    return float(1 - (w * O).sum() / denom) if denom > 0 else float("nan")


def main():
    ce.AUDIT = OUT / "_mag_skew_audit.jsonl"        # 운영 audit 오염 방지

    idx = pd.read_csv(_p("embeddings") / "ctranspath" / "index.csv", dtype={"magnification": str})
    idx = idx[idx["stain"].isin(STAIN_KEEP)]
    idx_T = idx[idx["magnification"].isin(["10", "40"])]
    idx_S = idx[idx["magnification"] == "20"]

    lab = _build_labels()
    pids_all = lab["patient_id"].tolist()
    bags_T = load_bags(pids_all, idx_T, keep=STAIN_KEEP)
    bags_S = load_bags(pids_all, idx_S, keep=STAIN_KEEP)

    # 두 조건 모두에서 HE&PAS 보유(=ALLOW 가능·apples-to-apples)한 환자만 비교
    def has_req(b):
        return b is not None and {"HE", "PAS"} <= set(b.keys())
    pids = [p for p in pids_all
            if has_req(bags_T.get(p)) and has_req(bags_S.get(p))]
    print(f"비교 대상 환자 n={len(pids)} (T,S 모두 HE&PAS 보유)", flush=True)

    eng = CdssEngine()
    print(f"모델 variant={eng.model_variant} ensemble={len(eng.models)}", flush=True)

    rows = []
    for pid in pids:
        rT = eng.analyze(bags_T[pid], slide_id=f"{pid}_T", batch=True)
        rS = eng.analyze(bags_S[pid], slide_id=f"{pid}_S", batch=True)
        rows.append({"pid": pid,
                     "decT": rT["decision"], "decS": rS["decision"],
                     "qcT": rT["qcScore"], "qcS": rS["qcScore"],
                     "uncT": rT.get("uncertainty"), "uncS": rS.get("uncertainty"),
                     "gT": rT.get("finalPrediction"), "gS": rS.get("finalPrediction"),
                     "nT": {s: int(v.shape[0]) for s, v in bags_T[pid].items()},
                     "nS": {s: int(v.shape[0]) for s, v in bags_S[pid].items()}})

    # ---- 요약 ----
    df = pd.DataFrame(rows)
    n = len(df)
    same_dec = int((df["decT"] == df["decS"]).sum())
    both_allow = df[(df["decT"] == "ALLOW") & (df["decS"] == "ALLOW")]
    print(f"\n=== 결정 일치 ===")
    print(f"  decision 동일: {same_dec}/{n}  (T ALLOW={int((df.decT=='ALLOW').sum())}, "
          f"S ALLOW={int((df.decS=='ALLOW').sum())}, both ALLOW={len(both_allow)})")
    print(f"  QC score  mean T={df.qcT.mean():.3f}  S={df.qcS.mean():.3f}")

    print(f"\n=== ① T vs S 예측 divergence (both ALLOW n={len(both_allow)}) ===")
    for t in ORD:
        gt = both_allow["gT"].map(lambda d: d[t]); gs = both_allow["gS"].map(lambda d: d[t])
        diff = (gt - gs).abs()
        exact = float((diff == 0).mean()); w1 = float((diff <= 1).mean())
        print(f"  {t:12} 등급 정확일치 {exact:.0%} | ±1 {w1:.0%} | MAE {diff.mean():.3f} "
              f"| meanGrade T={gt.mean():.2f} S={gs.mean():.2f}")
    if len(both_allow):
        du = (both_allow["uncT"] - both_allow["uncS"]).abs()
        print(f"  uncertainty |ΔT-S| mean={du.mean():.3f}  (T={both_allow.uncT.mean():.3f} S={both_allow.uncS.mean():.3f})")

    # ---- ② gold 대비 정확도(라벨 있는 환자) ----
    goldmap = {r.patient_id: r for r in lab.itertuples()}
    print(f"\n=== ② gold Banff 대비 정확도 (T,S 각각) ===")
    for t in ORD:
        recs = []
        for _, r in both_allow.iterrows():
            g = getattr(goldmap.get(r["pid"]), t, np.nan)
            if g == g:  # not NaN
                recs.append((int(g), int(r["gT"][t]), int(r["gS"][t])))
        if len(recs) < 5:
            print(f"  {t:12} 라벨 {len(recs)}명 — 스킵"); continue
        y, pT, pS = zip(*recs)
        maeT = np.mean(np.abs(np.array(y) - np.array(pT)))
        maeS = np.mean(np.abs(np.array(y) - np.array(pS)))
        print(f"  {t:12} n={len(recs):2} | QWK  T={qwk(y,pT):.3f}  S={qwk(y,pS):.3f}"
              f"   | MAE  T={maeT:.3f}  S={maeS:.3f}   (S가 나쁘면 배율스큐 실재)")

    # 결정 바뀐 환자
    flip = df[df["decT"] != df["decS"]]
    if len(flip):
        print(f"\n=== 결정 뒤집힌 환자 {len(flip)} ===")
        for _, r in flip.iterrows():
            print(f"  {r['pid']}: T={r['decT']} S={r['decS']} (qc {r['qcT']}->{r['qcS']}, nT={r['nT']} nS={r['nS']})")

    (OUT / "mag_skew_result.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n-> {OUT/'mag_skew_result.json'}")


if __name__ == "__main__":
    main()
