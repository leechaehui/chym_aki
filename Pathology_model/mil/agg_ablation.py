"""stain ablation 결과를 stain x task 행렬로 집계. 사용: agg_ablation.py [exp1 resnet50]"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "exp1"
ENC = sys.argv[2] if len(sys.argv) > 2 else "resnet50"
ROWS = [("multi", PREFIX)] + [(s, f"{PREFIX}_{s}") for s in ("HE", "PAS", "MT", "SILVER", "IF")]


def load(tag):
    p = ROOT / f"mil_cv_{tag}_{ENC}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def auroc(d, t):
    m = d["tasks"].get(t, {})
    a = m.get("auroc")
    if a is None:
        return "  -  "
    ci = m.get("auroc_ci95", [None, None])
    return f"{a:.2f}[{ci[0]},{ci[1]}] n{m.get('n')}({m.get('n_pos')})"


def spear(d):
    m = d["tasks"].get("ati_severity", {})
    s = m.get("spearman")
    return f"{s:+.2f} n{m.get('n')}" if s is not None else "  -  "


print("=" * 100)
print(f"{PREFIX} ({ENC}, 512/10x) — stain별 단독 vs multi  [AUROC(CI) / Spearman]")
print("=" * 100)
hdr = f"{'stain':<8}{'immune(AIN/ATI)':<26}{'stage3(급성)':<26}{'chronic(만성)':<26}{'ati_sev'}"
print(hdr)
print("-" * 100)
for name, tag in ROWS:
    d = load(tag)
    if d is None:
        print(f"{name:<8}(없음)"); continue
    print(f"{name:<8}{auroc(d,'immune'):<26}{auroc(d,'stage3'):<26}{auroc(d,'chronic'):<26}{spear(d)}")

# multi-stain fusion attention
m = load("exp1")
if m and "stain_contribution" in m:
    print("\n[Multi-stain fusion attention (평균, 클수록 모델이 중시)]")
    sc = m["stain_contribution"]
    for s in sorted(sc, key=lambda k: -sc[k]):
        print(f"   {s:<8} {sc[s]:.3f}")
