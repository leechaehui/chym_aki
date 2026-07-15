"""Exp1/2/3 인코더 비교: task별 (stain x encoder) AUROC/Spearman 행렬."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENCS = [("ResNet50", "exp1", "resnet50"), ("CTransPath", "exp2", "ctranspath"),
        ("DINOv2", "exp3", "dinov2")]
STAINS = ["multi", "HE", "PAS", "MT", "SILVER", "IF"]


def load(prefix, enc, stain):
    tag = prefix if stain == "multi" else f"{prefix}_{stain}"
    p = ROOT / f"mil_cv_{tag}_{enc}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def cell_auroc(d, t):
    if not d:
        return "  na   "
    m = d["tasks"].get(t, {})
    a = m.get("auroc")
    return f"{a:.2f}" if a is not None else "  -  "


def cell_spear(d):
    if not d:
        return " na "
    s = d["tasks"].get("ati_severity", {}).get("spearman")
    return f"{s:+.2f}" if s is not None else "  -  "


for t, label, fn in [("immune", "immune (AIN vs ATI) [급성]", cell_auroc),
                     ("stage3", "stage3 (급성 중증 이진)", cell_auroc),
                     ("chronic", "chronic (DKD/HTN vs acute) [만성]", cell_auroc),
                     ("ati_severity", "ati_severity (Spearman)", None)]:
    print(f"\n=== {label} ===")
    print(f"{'stain':<8}" + "".join(f"{e[0]:>12}" for e in ENCS))
    for st in STAINS:
        cells = []
        for _, pre, enc in ENCS:
            d = load(pre, enc, st)
            cells.append((cell_spear(d) if t == "ati_severity" else cell_auroc(d, t)))
        print(f"{st:<8}" + "".join(f"{c:>12}" for c in cells))

print("\n=== multi-stain fusion attention (encoder별) ===")
for name, pre, enc in ENCS:
    d = load(pre, enc, "multi")
    sc = d.get("stain_contribution", {}) if d else {}
    s = " ".join(f"{k}:{v:.2f}" for k, v in sorted(sc.items(), key=lambda x: -x[1]))
    print(f"  {name:<11} {s}")
