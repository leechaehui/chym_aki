"""
실험6 — Patient Embedding UMAP (데이터 자체 분리가능성 검증)

성능 한계가 '모델' 때문인지 '데이터' 때문인지 가른다: 학습 없이 raw patient embedding
(환자별 전 패치 평균; ctranspath, 비-SILVER MAIN)을 2D 투영해 진단별 cluster 분리를 본다.
DKD/AIN 이 ATI 와 raw feature 공간에서 안 갈라지면 → 모델이 아니라 데이터(분리신호 부족) 문제.

출력: results/repeated_cv/umap_patient_embedding.png  (+ 좌표 csv)
사용: python umap_patient_embedding.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Malgun Gothic"; matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path("c:/team/chym_aki/Pathology_model")))
ROOT = Path("c:/team/chym_aki")
OUT = ROOT / "Pathology_model/results/06_data_limitation"
ENC = "ctranspath"; MAG = "10"; MAIN = ["HE", "PAS", "MT"]
DX_COLOR = {"Acute Tubular Injury": "tab:gray", "Acute Interstitial Nephritis": "tab:blue",
            "Diabetic Kidney Disease": "tab:red", "Hypertensive Kidney Disease": "tab:orange"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(ROOT / "data/embeddings" / ENC / "index.csv", dtype={"magnification": str})
    idx = idx[(idx["magnification"] == MAG) & (idx["stain"].isin(MAIN))]
    sm = pd.read_csv(ROOT / "split_manifest.csv").drop_duplicates("patient_id")
    sm["patient_id"] = sm["patient_id"].astype(str)
    dx = {str(r.patient_id): str(r.primary_adjudicated_category) for r in sm.itertuples()}

    feats, pids, labels = [], [], []
    for pid in sm[sm["primary_adjudicated_category"].isin(DX_COLOR)]["patient_id"]:
        rows = idx[idx["patient_id"].astype(str) == pid]
        arrs = [np.load(ROOT / r.npy_path) for r in rows.itertuples()]
        arrs = [a for a in arrs if a.shape[0] > 0]
        if not arrs:
            continue
        feats.append(np.concatenate(arrs, 0).mean(0))   # 환자 = 전 MAIN 패치 평균(raw)
        pids.append(pid); labels.append(dx[pid])
    X = np.vstack(feats)
    print(f"환자 {len(X)}명 raw embedding (dim {X.shape[1]}) — 진단 분포: "
          f"{pd.Series(labels).value_counts().to_dict()}", flush=True)

    try:
        import umap
        emb = umap.UMAP(n_neighbors=min(15, len(X) - 1), min_dist=0.1,
                        random_state=42).fit_transform(X)
        method = "UMAP"
    except Exception as e:
        from sklearn.decomposition import PCA
        emb = PCA(n_components=2, random_state=42).fit_transform(X)
        method = f"PCA(UMAP실패:{e})"

    fig, ax = plt.subplots(figsize=(8, 7))
    for cat, col in DX_COLOR.items():
        m = [i for i, l in enumerate(labels) if l == cat]
        if m:
            ax.scatter(emb[m, 0], emb[m, 1], c=col, label=f"{cat.split()[0]} (n={len(m)})",
                       s=60, alpha=0.8, edgecolors="k", linewidths=0.4)
    ax.set_title(f"Patient raw embedding {method}\n(데이터 자체 분리가능성 — cluster 안 갈리면 데이터 한계)")
    ax.legend(); ax.set_xlabel(f"{method}-1"); ax.set_ylabel(f"{method}-2")
    fig.tight_layout(); fig.savefig(OUT / "umap_patient_embedding.png", dpi=120); plt.close(fig)
    pd.DataFrame({"patient_id": pids, "dx": labels, "x": emb[:, 0], "y": emb[:, 1]}).to_csv(
        OUT / "umap_coords.csv", index=False, encoding="utf-8")
    print(f"-> {OUT}\\umap_patient_embedding.png ({method})")


if __name__ == "__main__":
    main()
