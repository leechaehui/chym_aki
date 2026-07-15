"""
Exp9 — Robustness 분석 (설계 §10, CKD OOD 폐기 → AKI 내부 분석으로 재정의)

CKD 외부코호트가 없으므로 OOD 대신 'AKI 내부' robustness 를 본다:
  A. chronic vs acute phenotype separation  — chronic task(DKD/HTN=1 vs ATI/AIN=0) OOF 분리도
  B. fibrosis-heavy stress test             — 만성(섬유화 과다) 환자에서 acute/severity 예측 왜곡
  C. attention entropy 변화 (옵션, --attn)  — SILVER 제약 유무가 attention 안정성에 미치는 영향

A·B 는 기존 OOF csv 만으로 즉시 산출(토치 불필요). C 는 모델 패스가 필요해 토치/임베딩 가용 시에만.

사용:
  python exp9_robustness.py --tag exp10main --encoder ctranspath          # A,B
  python exp9_robustness.py --tag exp10main --encoder ctranspath --attn   # +C(무거움)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "Pathology_model")))
ROOT = Path(__file__).resolve().parents[2]
SEED = 42

ACUTE = {"Acute Tubular Injury", "Acute Interstitial Nephritis"}
CHRONIC = {"Diabetic Kidney Disease", "Hypertensive Kidney Disease"}


def _auroc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); p = np.asarray(p)
    return round(float(roc_auc_score(y, p)), 3) if len(np.unique(y)) > 1 else None


def phenotype_separation(oof: pd.DataFrame) -> dict:
    """A — chronic task OOF 로 chronic(DKD/HTN) vs acute(ATI/AIN) 분리도."""
    ch = oof[oof["task"] == "chronic"]
    if ch.empty:
        return {"note": "chronic OOF 없음"}
    pos = ch[ch["y"] == 1]["p"]; neg = ch[ch["y"] == 0]["p"]
    return {
        "auroc": _auroc(ch["y"], ch["p"]),
        "n_chronic": int((ch["y"] == 1).sum()), "n_acute": int((ch["y"] == 0).sum()),
        "mean_score_chronic": round(float(pos.mean()), 3) if len(pos) else None,
        "mean_score_acute": round(float(neg.mean()), 3) if len(neg) else None,
        "score_gap": round(float(pos.mean() - neg.mean()), 3) if len(pos) and len(neg) else None,
    }


def fibrosis_stress(oof: pd.DataFrame, dx: dict) -> dict:
    """B — 섬유화 과다(만성 dx) 환자에서 severity/stage3 예측이 왜곡되는지."""
    def grp(pid):
        c = dx.get(str(pid), "")
        return "chronic" if c in CHRONIC else ("acute" if c in ACUTE else "other")
    out = {}
    for task in ("ati_severity", "stage3"):
        t = oof[oof["task"] == task].copy()
        if t.empty:
            continue
        t["grp"] = t["patient_id"].map(grp)
        stats = {}
        for g in ("acute", "chronic"):
            sub = t[t["grp"] == g]
            if sub.empty:
                continue
            stats[g] = {"n": int(len(sub)),
                        "mean_pred": round(float(sub["p"].mean()), 3),
                        "mean_true": round(float(sub["y"].mean()), 3),
                        "mean_abs_err": round(float((sub["p"] - sub["y"]).abs().mean()), 3)}
        out[task] = stats
    return out


def attention_entropy(encoder: str, dx: dict) -> dict:
    """C — SILVER off vs consistency_attn 로 full-cohort 학습 후 그룹별 attention 엔트로피."""
    import torch
    from mil.model import StainAwareMIL
    from mil.train import load_bags

    emb_dir = ROOT / "data/embeddings" / encoder
    idx = pd.read_csv(emb_dir / "index.csv", dtype={"magnification": str})
    idx = idx[(idx["magnification"] == "10") |
              ((idx["stain"] == "IF") & (idx["magnification"] == "native"))].copy()
    embed_dim = int(idx["embed_dim"].iloc[0])
    sm = pd.read_csv(ROOT / "split_manifest.csv").drop_duplicates("patient_id")
    coh = sm[sm["fold"] >= 0].copy(); coh["patient_id"] = coh["patient_id"].astype(str)
    bags = load_bags(coh["patient_id"].tolist(), idx, keep=None)
    coh = coh[coh["patient_id"].isin(bags.keys())]

    def group_entropy(mode):
        torch.manual_seed(SEED); np.random.seed(SEED)
        model = StainAwareMIL(in_dim=embed_dim, silver_mode=mode)
        model.eval()
        ent = {"acute": [], "chronic": []}
        with torch.no_grad():
            for pid, bag in bags.items():
                tb = {s: torch.from_numpy(v) for s, v in bag.items()}
                out = model(tb)
                e = out.get("main_attn_entropy")
                if e is None:  # off 모드는 엔트로피 미반환 → 직접 계산
                    ents = []
                    for a in out["patch_attn"].values():
                        n = a.shape[0]
                        if n > 1:
                            ents.append(float(-(a * (a + 1e-8).log()).sum() / np.log(n)))
                    e = np.mean(ents) if ents else None
                else:
                    e = float(e)
                c = dx.get(str(pid), "")
                g = "chronic" if c in CHRONIC else ("acute" if c in ACUTE else None)
                if g and e is not None:
                    ent[g].append(e)
        return {g: round(float(np.mean(v)), 3) if v else None for g, v in ent.items()}

    return {"note": "정규화 엔트로피(0~1, 1=완전분산). 미학습 init 비교 — 안정성 상대지표.",
            "silver_off": group_entropy("off"),
            "silver_consistency_attn": group_entropy("consistency_attn")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="exp10main")
    ap.add_argument("--encoder", default="ctranspath")
    ap.add_argument("--attn", action="store_true", help="attention 엔트로피(C) 포함(토치 필요)")
    args = ap.parse_args()

    oof_path = ROOT / f"oof_{args.tag}_{args.encoder}.csv"
    if not oof_path.exists():
        sys.exit(f"OOF 없음: {oof_path}")
    oof = pd.read_csv(oof_path)
    sm = pd.read_csv(ROOT / "split_manifest.csv").drop_duplicates("patient_id")
    dx = {str(r.patient_id): str(r.primary_adjudicated_category)
          for r in sm.itertuples()}

    report = {"tag": args.tag, "encoder": args.encoder,
              "A_phenotype_separation": phenotype_separation(oof),
              "B_fibrosis_stress": fibrosis_stress(oof, dx)}
    if args.attn:
        report["C_attention_entropy"] = attention_entropy(args.encoder, dx)

    out = ROOT / f"exp9_robustness_{args.tag}_{args.encoder}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    # 콘솔은 cp949 환경 깨짐 방지를 위해 ascii-safe 요약만 출력(전체는 파일 참조)
    a = report["A_phenotype_separation"]
    print(f"[A] chronic/acute separation AUROC={a.get('auroc')} "
          f"gap={a.get('score_gap')} (n_chronic={a.get('n_chronic')}/n_acute={a.get('n_acute')})")
    print(f"[B] fibrosis stress -> {json.dumps(report['B_fibrosis_stress'], ensure_ascii=True)}")
    if "C_attention_entropy" in report:
        c = report["C_attention_entropy"]
        print(f"[C] attn entropy off={c.get('silver_off')} "
              f"consistency_attn={c.get('silver_consistency_attn')}")
    print(f"-> {out.name}")


if __name__ == "__main__":
    main()
