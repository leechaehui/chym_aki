"""10x vs 40x vs 멀티스케일: stain별(HE/PAS/MT) + multi + multiscale, task별 AUROC."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(tag):
    p = ROOT / f"mil_cv_{tag}_ctranspath.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def au(d, t):
    if not d:
        return "  -   "
    m = d["tasks"].get(t, {})
    a = m.get("auroc")
    return f"{a:.2f}" if a is not None else "  -   "


def sp(d):
    if not d:
        return "  -  "
    s = d["tasks"].get("ati_severity", {}).get("spearman")
    return f"{s:+.2f}" if s is not None else "  -  "


# (label, 10x tag, 40x tag)
ROWS = [("HE",   "exp2_HE",  "exp5_HE"),
        ("PAS",  "exp2_PAS", "exp5_PAS"),
        ("MT",   "exp2_MT",  "exp5_MT"),
        ("MULTI(HE+PAS+MT)", "exp10main", "exp5")]

for task, label in [("immune", "immune (급성: AIN vs ATI)"),
                    ("stage3", "stage3 (급성 중증)"),
                    ("chronic", "chronic (만성)")]:
    print(f"\n=== {label}  [AUROC] ===")
    print(f"{'stain':<18}{'10x':>8}{'40x':>8}{'Δ(40-10)':>10}")
    for name, t10, t40 in ROWS:
        d10, d40 = load(t10), load(t40)
        a10 = d10["tasks"].get(task, {}).get("auroc") if d10 else None
        a40 = d40["tasks"].get(task, {}).get("auroc") if d40 else None
        delta = f"{a40-a10:+.2f}" if (a10 is not None and a40 is not None) else "  -  "
        print(f"{name:<18}{au(d10,task):>8}{au(d40,task):>8}{delta:>10}")

print("\n=== ati_severity (Spearman) ===")
print(f"{'stain':<18}{'10x':>8}{'40x':>8}")
for name, t10, t40 in ROWS:
    print(f"{name:<18}{sp(load(t10)):>8}{sp(load(t40)):>8}")

print("\n=== 멀티스케일 (10x+40x, MULTI) Exp6 ===")
e6 = load("exp6")
if e6:
    for t in ("immune", "stage3", "chronic"):
        m = e6["tasks"].get(t, {})
        print(f"  {t:<10} AUROC {m.get('auroc')} CI{m.get('auroc_ci95')}")
    print(f"  ati_severity Spearman {e6['tasks'].get('ati_severity',{}).get('spearman')}")
    print(f"  scale contribution: {e6.get('scale_contribution')}")
