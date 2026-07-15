"""
Phase A — 환자단위 분할 + 라벨 등급화
거버넌스 우선순위 #1(데이터 누수 방지), #2(라벨 신뢰성), 기반 #5(재현성)·#6(계보)

원칙:
 - redcap_id(환자) 단위로만 분할. 동일 환자의 모든 slide/stain은 같은 fold에 귀속.
 - slide 단위 분할 금지. split_manifest는 file-level이지만 fold는 '환자'에서 상속.
 - 라벨 등급(gold/silver/unlabeled) 구분. unlabeled는 지도학습 제외(SSL 풀).
 - CV도 환자단위 StratifiedKFold. 모든 seed/version 고정·기록.

라벨은 모두 'weak/proxy'임에 유의(KPMP는 etiology 3-class GT가 없음):
 - immune       : AIN=1 vs ATI=0  (head2 / AIN proxy)
 - ati_severity : kdigo 단계 ordinal proxy, 급성(ATI/AIN) 한정 (head1)
 - chronic      : DKD/HTN=1 vs ATI/AIN=0  (head3 / 섬유화 burden proxy)
"""
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
MANI = ROOT / "manifest_aki_full.csv"
SEL = ROOT / "selected_manifest.csv"
BASE = ROOT / "data/raw"
OUT = ROOT / "split_manifest.csv"
REPORT = ROOT / "split_report.json"

CONFIG = {
    "dataset_version": "v1",
    "split_seed": 42,
    "n_folds": 5,
    "split_scheme": "patient-level StratifiedKFold (gold only)",
    "stratify_levels": ["ati", "ain", "chronic"],
    "created": str(date.today()),
}

GOLD_CATS = {"Acute Tubular Injury", "Acute Interstitial Nephritis",
             "Diabetic Kidney Disease", "Hypertensive Kidney Disease"}


def grade(cat):
    if pd.isna(cat):
        return "unlabeled"
    c = str(cat)
    if ";" in c or "Cannot be determined" in c:
        return "unlabeled"
    if c in GOLD_CATS:
        return "gold"
    if c == "Other":
        return "silver"
    return "unlabeled"


def stratify_key(cat):
    if cat == "Acute Tubular Injury":
        return "ati"
    if cat == "Acute Interstitial Nephritis":
        return "ain"
    if cat in ("Diabetic Kidney Disease", "Hypertensive Kidney Disease"):
        return "chronic"
    return None


def task_immune(cat):
    if cat == "Acute Interstitial Nephritis":
        return 1
    if cat == "Acute Tubular Injury":
        return 0
    return np.nan


def task_chronic(cat):
    if cat in ("Diabetic Kidney Disease", "Hypertensive Kidney Disease"):
        return 1
    if cat in ("Acute Tubular Injury", "Acute Interstitial Nephritis"):
        return 0
    return np.nan


def kdigo_ord(k):
    if pd.isna(k):
        return np.nan
    s = str(k)
    if ";" in s:          # 다단계 동시기재 = 모호 -> 제외
        return np.nan
    for n in (3, 2, 1):
        if f"Stage {n}" in s:
            return n
    return np.nan


def task_stage3(kord):
    # 보조 이진: Stage3(중증) vs Non-Stage3. 단일 kdigo 있는 전체 환자.
    if pd.isna(kord):
        return np.nan
    return 1 if kord == 3 else 0


def dest_path(r):
    pid = str(r["redcap_id"]).replace(";", "_")
    suf = r["_grp"].replace("wsi_", "").upper()
    fid = str(r["file_name"]).split("_")[0][:8]
    ext = ".tif" if r["_grp"] == "wsi_if" else ".svs"
    return f"data/raw/{r['_grp']}/{pid}_{suf}_{fid}{ext}"


def main():
    full = pd.read_csv(MANI)
    sel = pd.read_csv(SEL)

    # 환자단위 라벨/등급 테이블
    pat = full.drop_duplicates("redcap_id").set_index("redcap_id")
    pat_tbl = pd.DataFrame(index=pat.index)
    pat_tbl["primary_adjudicated_category"] = pat["primary_adjudicated_category"]
    pat_tbl["kdigo_stage"] = pat["kdigo_stage"]
    pat_tbl["label_grade"] = pat_tbl["primary_adjudicated_category"].apply(grade)
    pat_tbl["stratify_key"] = pat_tbl["primary_adjudicated_category"].apply(stratify_key)
    pat_tbl["task_immune"] = pat_tbl["primary_adjudicated_category"].apply(task_immune)
    pat_tbl["task_chronic"] = pat_tbl["primary_adjudicated_category"].apply(task_chronic)
    kord = pat_tbl["kdigo_stage"].apply(kdigo_ord)
    # ati_severity = 전체 단일-kdigo 환자(병리 adjudication과 무관한 임상 gold 지표)
    pat_tbl["task_ati_severity"] = kord.to_numpy()
    pat_tbl["task_stage3"] = kord.apply(task_stage3).to_numpy()
    pat_tbl["fold"] = -1

    skf = StratifiedKFold(n_splits=CONFIG["n_folds"], shuffle=True,
                          random_state=CONFIG["split_seed"])
    # (1) gold 환자: stratify_key(ati/ain/chronic)로 fold 배정 → immune/chronic 용(불변)
    gold = pat_tbl[pat_tbl["label_grade"] == "gold"]
    gids = gold.index.to_numpy()
    for fold, (_, va) in enumerate(skf.split(gids, gold["stratify_key"].to_numpy())):
        pat_tbl.loc[gids[va], "fold"] = fold
    # (2) 비-gold이지만 단일 kdigo 있는 환자: stage3로 stratify해 fold 추가(severity/stage3 용)
    extra = pat_tbl[(pat_tbl["fold"] == -1) & pat_tbl["task_stage3"].notna()]
    if len(extra):
        eids = extra.index.to_numpy()
        for fold, (_, va) in enumerate(skf.split(eids, extra["task_stage3"].to_numpy())):
            pat_tbl.loc[eids[va], "fold"] = fold

    # file-level split_manifest (slide마다 환자 fold 상속) + 계보(lineage) 컬럼
    sel = sel.copy()
    sel["stain"] = sel["_grp"].str.replace("wsi_", "", regex=False).str.upper()
    sel["slide_path"] = sel.apply(dest_path, axis=1)
    sel["file_id"] = sel["file_name"].str.split("_").str[0]
    # pat_tbl이 라벨/판독의 단일 출처 -> sel의 중복 컬럼 제거 후 join
    sel = sel.drop(columns=[c for c in ("primary_adjudicated_category", "kdigo_stage")
                            if c in sel.columns])
    join = sel.join(pat_tbl, on="redcap_id")

    cols = ["redcap_id", "stain", "_grp", "file_id", "package_id", "file_name",
            "slide_path", "file_size", "label_grade", "stratify_key", "fold",
            "task_immune", "task_ati_severity", "task_stage3", "task_chronic",
            "primary_adjudicated_category", "kdigo_stage"]
    out = join[cols].rename(columns={"redcap_id": "patient_id"})
    out.to_csv(OUT, index=False, encoding="utf-8")

    # ---- 리포트 (#2 로그: gold/silver/unlabeled, #1 분포) ----
    npat = pat_tbl.shape[0]
    grade_pat = pat_tbl["label_grade"].value_counts().to_dict()
    report = {
        "config": CONFIG,
        "patient_counts": {
            "total": int(npat),
            "gold": int(grade_pat.get("gold", 0)),
            "silver": int(grade_pat.get("silver", 0)),
            "unlabeled": int(grade_pat.get("unlabeled", 0)),
        },
        "slide_counts": {
            "total": int(len(out)),
            "by_grade": out.groupby("label_grade")["file_id"].size().to_dict(),
        },
        "fold_patient_distribution": {},
        "fold_label_distribution": {},
    }
    for f in range(CONFIG["n_folds"]):
        fp = pat_tbl[pat_tbl["fold"] == f]
        report["fold_patient_distribution"][f] = fp["stratify_key"].value_counts().to_dict()
        report["fold_label_distribution"][f] = {
            "immune_pos": int((fp["task_immune"] == 1).sum()),
            "immune_neg": int((fp["task_immune"] == 0).sum()),
            "chronic_pos": int((fp["task_chronic"] == 1).sum()),
        }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 콘솔 요약 ----
    print("=" * 60)
    print(" Phase A — 환자단위 분할 + 라벨 등급화")
    print("=" * 60)
    print(f"환자: 총 {npat}  | gold {grade_pat.get('gold',0)} "
          f"silver {grade_pat.get('silver',0)} unlabeled {grade_pat.get('unlabeled',0)}")
    print(f"slide(file): {len(out)}  ->  {OUT.name}")
    print("\n[CV fold별 환자 층화 분포] (gold만, fold -1=제외)")
    print(pat_tbl[pat_tbl.fold >= 0].groupby(["fold", "stratify_key"]).size()
          .unstack(fill_value=0).to_string())
    print("\n[누수 점검] 같은 환자가 2개 이상 fold에 존재? ", end="")
    leak = out.groupby("patient_id")["fold"].nunique()
    print("없음 ✅" if (leak <= 1).all() else f"누수! {leak[leak>1].index.tolist()}")
    print(f"\n리포트: {REPORT.name}")


if __name__ == "__main__":
    main()
