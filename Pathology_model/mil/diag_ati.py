"""
ATI severity 음의 Spearman 진단

oof_exp1_resnet50.csv(task==ati_severity)의 OOF 예측으로:
 - ground truth 분포(0=Stage1, 0.5=Stage2, 1.0=Stage3; 클수록 중증)
 - 예측 통계(붕괴 여부)
 - fold별 Spearman
 - 부호반전 시 Spearman
 - scatter / histogram PNG 저장
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
TAG, ENC = "exp1", "resnet50"
OUT = ROOT / "data/diag"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(ROOT / f"oof_{TAG}_{ENC}.csv")
    a = df[df["task"] == "ati_severity"].copy()
    y, p = a["y"].to_numpy(), a["p"].to_numpy()
    print(f"n={len(a)}")
    print("\n[1] ground truth 분포 (0=St1,0.5=St2,1.0=St3; 클수록 중증)")
    print(a["y"].value_counts().sort_index().to_string())
    print("\n[2] 예측 통계 (붕괴 점검)")
    print(f"  pred min={p.min():.3f} max={p.max():.3f} mean={p.mean():.3f} std={p.std():.3f} range={p.max()-p.min():.3f}")
    print(f"  gt   min={y.min():.3f} max={y.max():.3f} mean={y.mean():.3f} std={y.std():.3f}")

    print("\n[3] fold별 Spearman")
    for f, g in a.groupby("fold"):
        if g["y"].nunique() > 1:
            rho = spearmanr(g["y"], g["p"]).correlation
            print(f"  fold {f}: rho={rho:+.3f} (n={len(g)}, y분포={dict(g['y'].value_counts().sort_index())})")
        else:
            print(f"  fold {f}: y 단일값 -> Spearman 불가 (n={len(g)}, y={g['y'].iloc[0]})")

    rho_all = spearmanr(y, p).correlation
    print(f"\n[4] 전체 Spearman = {rho_all:+.3f}")
    print(f"[5] 부호반전(pred=-p) Spearman = {spearmanr(y, -p).correlation:+.3f} (반전은 단순 부호변경)")
    # 그룹 평균 예측(단조성): y레벨별 평균 pred — 중증일수록 pred가 커야 정상
    print("\n[6] y레벨별 평균 예측 (정상이면 단조증가)")
    print(a.groupby("y")["p"].agg(["mean", "std", "count"]).round(3).to_string())

    # 산점도
    plt.figure(figsize=(5, 4))
    jitter = (np.random.default_rng(0).random(len(y)) - 0.5) * 0.04
    plt.scatter(y + jitter, p, alpha=0.6, s=25)
    plt.xlabel("ground truth (severity)"); plt.ylabel("OOF prediction")
    plt.title(f"ATI severity OOF (rho={rho_all:+.3f})")
    plt.xticks([0, 0.5, 1.0]); plt.tight_layout()
    plt.savefig(OUT / "ati_scatter.png", dpi=110); plt.close()

    # 히스토그램
    plt.figure(figsize=(5, 4))
    plt.hist(p, bins=20, alpha=0.8)
    plt.axvline(p.mean(), color="r", ls="--", label=f"mean={p.mean():.3f}")
    plt.xlabel("OOF prediction"); plt.ylabel("count"); plt.legend()
    plt.title("ATI severity prediction histogram"); plt.tight_layout()
    plt.savefig(OUT / "ati_hist.png", dpi=110); plt.close()
    print(f"\n저장: {OUT/'ati_scatter.png'} , {OUT/'ati_hist.png'}")


if __name__ == "__main__":
    main()
