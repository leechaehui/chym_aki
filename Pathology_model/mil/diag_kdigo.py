from pathlib import Path
"""gold 필터가 Stage1/2를 배제했는지 추적."""
import pandas as pd

pd.set_option("display.width", 200)
full = pd.read_csv(str(Path(__file__).resolve().parents[2] / "manifest_aki_full.csv")).drop_duplicates("redcap_id")
sm = pd.read_csv(str(Path(__file__).resolve().parents[2] / "split_manifest.csv")).drop_duplicates("patient_id")


def kd(s):
    s = str(s)
    if ";" in s:
        return "multi"
    for n in (1, 2, 3):
        if f"Stage {n}" in s:
            return f"St{n}"
    return "NaN"


def cat(s):
    s = str(s)
    if ";" in s or "Cannot" in s:
        return "mixed/undet"
    if s == "nan":
        return "NaN판독"
    return s


full["kd"] = full["kdigo_stage"].apply(kd)
full["cat"] = full["primary_adjudicated_category"].apply(cat)

print("[A] 전체 AKI 코호트(95) kdigo")
print(full["kd"].value_counts().to_string())

print("\n[B] kdigo x 판독카테고리 교차표")
print(pd.crosstab(full["cat"], full["kd"]).to_string())

g = sm[sm.label_grade == "gold"].merge(full[["redcap_id", "kd"]],
                                       left_on="patient_id", right_on="redcap_id", how="left")
print("\n[C] gold subset kdigo")
print(g["kd"].value_counts().to_string())

gg = sm[(sm.label_grade == "gold") & (sm.task_ati_severity.notna())].merge(
    full[["redcap_id", "kd"]], left_on="patient_id", right_on="redcap_id", how="left")
print("\n[D] ati_severity 라벨대상(gold ATI/AIN & 단일kdigo) kdigo")
print(gg["kd"].value_counts().to_string())

ng = sm[sm.label_grade != "gold"].merge(full[["redcap_id", "kd", "cat"]],
                                        left_on="patient_id", right_on="redcap_id", how="left")
print("\n[E] 비-gold 환자 kdigo (St1/2가 여기 숨었나)")
print(ng["kd"].value_counts().to_string())
print("\n[E2] 비-gold 환자 판독카테고리")
print(ng["cat"].value_counts().to_string())
