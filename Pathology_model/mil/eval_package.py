"""
STEP 4 — 평가 패키지: ROC/PR/calibration/confusion + error analysis

oof_{tag}_{encoder}.csv 의 OOF 예측으로 task별(immune/chronic/stage3) 그림과
오분류 표를 생성. 작은 N(탐색적)임을 그림에 명시.

사용: python eval_package.py exp5 ctranspath  (여러 개 인자 가능)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Malgun Gothic"   # 한글 글리프(폰트문제 해결, UTF-8 무관)
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, average_precision_score,
                             confusion_matrix, roc_auc_score)
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/eval"
OUT.mkdir(parents=True, exist_ok=True)
CLF = ["immune", "chronic", "stage3"]
META = pd.read_csv(ROOT / "manifest_aki_full.csv").drop_duplicates("redcap_id").set_index("redcap_id")


def panels(tag, enc):
    oof = pd.read_csv(ROOT / f"oof_{tag}_{enc}.csv")
    fig, ax = plt.subplots(len(CLF), 4, figsize=(16, 4 * len(CLF)))
    err_rows = []
    for i, t in enumerate(CLF):
        d = oof[oof["task"] == t]
        y, p = d["y"].to_numpy().astype(int), d["p"].to_numpy()
        if len(np.unique(y)) < 2:
            for j in range(4):
                ax[i, j].text(0.5, 0.5, f"{t}\n(단일 클래스)", ha="center"); ax[i, j].axis("off")
            continue
        a = roc_auc_score(y, p)
        # ROC
        fpr, tpr, _ = roc_curve(y, p)
        ax[i, 0].plot(fpr, tpr, label=f"AUROC={a:.2f}"); ax[i, 0].plot([0, 1], [0, 1], "k--", lw=0.6)
        ax[i, 0].set_title(f"{t} ROC"); ax[i, 0].set_xlabel("FPR"); ax[i, 0].set_ylabel("TPR"); ax[i, 0].legend(loc="lower right")
        # PR
        pr, rc, _ = precision_recall_curve(y, p); ap = average_precision_score(y, p)
        ax[i, 1].plot(rc, pr, label=f"AP={ap:.2f}"); ax[i, 1].axhline(y.mean(), ls="--", c="grey", lw=0.6)
        ax[i, 1].set_title(f"{t} PR"); ax[i, 1].set_xlabel("Recall"); ax[i, 1].set_ylabel("Precision"); ax[i, 1].legend()
        # calibration
        try:
            fp, mp = calibration_curve(y, p, n_bins=5, strategy="quantile")
            ax[i, 2].plot(mp, fp, "o-"); ax[i, 2].plot([0, 1], [0, 1], "k--", lw=0.6)
        except Exception:
            pass
        ax[i, 2].set_title(f"{t} Calibration"); ax[i, 2].set_xlabel("pred prob"); ax[i, 2].set_ylabel("obs freq")
        # confusion
        pred = (p > 0.5).astype(int); cm = confusion_matrix(y, pred, labels=[0, 1])
        ax[i, 3].imshow(cm, cmap="Blues"); ax[i, 3].set_title(f"{t} Confusion@0.5")
        ax[i, 3].set_xticks([0, 1]); ax[i, 3].set_yticks([0, 1])
        ax[i, 3].set_xlabel("pred"); ax[i, 3].set_ylabel("true")
        for (r_, c_), v in np.ndenumerate(cm):
            ax[i, 3].text(c_, r_, int(v), ha="center", va="center")
        # error analysis: 오분류 환자
        for _, row in d.iterrows():
            pr_ = int(row["p"] > 0.5)
            if pr_ != int(row["y"]):
                pid = row["patient_id"]
                cat = META.loc[pid, "primary_adjudicated_category"] if pid in META.index else "?"
                err_rows.append({"task": t, "patient_id": pid, "y": int(row["y"]),
                                 "p": round(row["p"], 3), "pred": pr_, "adjudication": cat})
    fig.suptitle(f"{tag} / {enc}  (N 작음 — 탐색적)", y=1.0)
    fig.tight_layout()
    fp_png = OUT / f"{tag}_{enc}_eval.png"
    fig.savefig(fp_png, dpi=110, bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(err_rows).to_csv(OUT / f"{tag}_{enc}_errors.csv", index=False, encoding="utf-8")
    print(f"  {tag}/{enc}: {fp_png.name}, errors {len(err_rows)}건")


def main():
    pairs = []
    args = sys.argv[1:]
    for i in range(0, len(args), 2):
        pairs.append((args[i], args[i + 1]))
    if not pairs:
        pairs = [("exp10main", "ctranspath"), ("exp5", "ctranspath"), ("exp6", "ctranspath")]
    print("STEP4 평가 패키지 생성:")
    for tag, enc in pairs:
        if (ROOT / f"oof_{tag}_{enc}.csv").exists():
            panels(tag, enc)
        else:
            print(f"  {tag}/{enc}: oof 없음 skip")


if __name__ == "__main__":
    main()
