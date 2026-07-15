"""
#7 Missing Modality — 환자별 stain 보유(modality availability) 마스크 아티팩트

split_manifest.csv에서 환자 x stain 보유 여부를 집계해 modality_mask.csv 생성.
이 마스크는 학습 시 missing-aware fusion의 입력(존재하는 stain만 융합)과
stain availability encoding(존재 여부 자체가 임상 정보)으로 사용된다.

원칙: missing stain을 0/평균으로 '대체'하지 않는다. 마스크로만 표현한다.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "split_manifest.csv"
OUT = ROOT / "modality_mask.csv"
REPORT = ROOT / "modality_mask_report.json"
STAINS = ["HE", "PAS", "MT", "SILVER", "IF"]


def main():
    df = pd.read_csv(SPLIT)
    piv = df.pivot_table(index="patient_id", columns="stain",
                         values="file_id", aggfunc="count", fill_value=0)
    for s in STAINS:
        if s not in piv:
            piv[s] = 0
    piv = piv[STAINS]
    avail = (piv > 0).astype(int)
    avail.columns = [f"has_{s}" for s in STAINS]
    avail["n_stains"] = avail.sum(1)
    avail["combo"] = piv.gt(0).apply(lambda r: "+".join(s for s in STAINS if r[s]), axis=1)
    # 슬라이드 수도 함께(계보)
    for s in STAINS:
        avail[f"n_{s}"] = piv[s]
    # fold/label 등급 join(환자단위)
    pat_meta = df.drop_duplicates("patient_id").set_index("patient_id")[
        ["fold", "label_grade", "stratify_key", "task_immune",
         "task_ati_severity", "task_chronic"]]
    out = avail.join(pat_meta).reset_index()
    out.to_csv(OUT, index=False, encoding="utf-8")

    report = {
        "n_patients": int(len(out)),
        "availability_rate": {s: round(float(avail[f"has_{s}"].mean()), 3) for s in STAINS},
        "n_stains_distribution": avail["n_stains"].value_counts().sort_index().to_dict(),
        "complete_5stain_patients": int((avail["n_stains"] == 5).sum()),
        "top_combos": avail["combo"].value_counts().head(8).to_dict(),
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("modality_mask.csv 생성:", len(out), "환자")
    print(json.dumps(report["availability_rate"], ensure_ascii=False))
    print("완전 5종 보유:", report["complete_5stain_patients"], "명")


if __name__ == "__main__":
    main()
