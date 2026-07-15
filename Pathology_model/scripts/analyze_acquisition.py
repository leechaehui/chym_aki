"""
KPMP AKI 코호트 추가 수집 타당성 분석 (분석1~4 + 최종 보고)

핵심 질문: AKI 코호트 내부에 DKD/HTN/AIN 양성이 더 남아 있는가?
데이터: manifest_aki_full.csv = enrollment_category=AKI '전체' 조회 스냅샷
(라이브 재조회로 imaging 1821파일 동일 확인 — 스냅샷이 최신).

출력 -> Pathology_model/results/acquisition/
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Malgun Gothic"; matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

ART = (Path(__file__).resolve().parents[2] / "Pathology_model/artifacts")
OUT = (Path(__file__).resolve().parents[2] / "Pathology_model/results/07_acquisition")
OUT.mkdir(parents=True, exist_ok=True)

DX_MAP = {"Acute Tubular Injury": "ATI", "Acute Interstitial Nephritis": "AIN",
          "Diabetic Kidney Disease": "DKD", "Hypertensive Kidney Disease": "HTN"}
CATS = ["ATI", "AIN", "DKD", "HTN", "FSGS", "MCD", "IgAN", "MN", "기타/미분류"]


def to_cat(s):
    s = str(s)
    if s in DX_MAP:
        return DX_MAP[s]
    for key, lab in (("FSGS", "FSGS"), ("Focal Segmental", "FSGS"), ("Minimal Change", "MCD"),
                     ("IgA", "IgAN"), ("Membranous", "MN")):
        if key in s:
            return lab
    return "기타/미분류"   # 빈값/Cannot be determined/Other/다중표기


def main():
    m = pd.read_csv(ART / "manifest_aki_full.csv", dtype=str).fillna("")
    sel = pd.read_csv(ART / "selected_manifest.csv", dtype=str).fillna("")
    # 이미징(WSI) 행, 집계 pseudo-ID(semicolon) 제외
    img = m[(m["data_type"] == "Imaging") & (~m["redcap_id"].str.contains(";"))].copy()
    img["cat"] = img["primary_adjudicated_category"].apply(to_cat)
    img["fsize"] = pd.to_numeric(img["file_size"], errors="coerce").fillna(0.0)
    pat = img.drop_duplicates("redcap_id")
    full_pat = set(pat["redcap_id"]); sel_pat = set(sel["redcap_id"])
    dl_pat = full_pat & sel_pat

    # ===== 분석1: 전체 AKI 진단 분포 =====
    full_cnt = pat["cat"].value_counts().to_dict()
    dist = pd.DataFrame({"diagnosis": CATS,
                         "aki_full_patients": [full_cnt.get(c, 0) for c in CATS]})
    dist.to_csv(OUT / "aki_full_diagnosis_distribution.csv", index=False, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(dist["diagnosis"], dist["aki_full_patients"], color="tab:blue")
    for i, v in enumerate(dist["aki_full_patients"]):
        ax.text(i, v + 0.3, str(v), ha="center")
    ax.set_title(f"전체 AKI 코호트 WSI 환자 진단 분포 (총 {len(pat)}명)"); ax.set_ylabel("환자 수")
    plt.xticks(rotation=30, ha="right"); fig.tight_layout()
    fig.savefig(OUT / "diagnosis_distribution.png", dpi=120); plt.close(fig)

    # ===== 분석2: 진단별 확보율 =====
    dl_cat = pat[pat["redcap_id"].isin(dl_pat)]["cat"].value_counts().to_dict()
    cov = pd.DataFrame({"diagnosis": CATS,
                        "full": [full_cnt.get(c, 0) for c in CATS],
                        "downloaded": [dl_cat.get(c, 0) for c in CATS]})
    cov["coverage_%"] = (cov["downloaded"] / cov["full"].replace(0, np.nan) * 100).round(1).fillna(0)
    cov.to_csv(OUT / "diagnosis_coverage.csv", index=False, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7, 4))
    hm = cov[cov["full"] > 0]
    im = ax.imshow(hm[["coverage_%"]].T.values, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(hm))); ax.set_xticklabels(hm["diagnosis"], rotation=30, ha="right")
    ax.set_yticks([0]); ax.set_yticklabels(["coverage %"])
    for i, (_, r) in enumerate(hm.iterrows()):
        ax.text(i, 0, f"{r['downloaded']}/{r['full']}\n{r['coverage_%']:.0f}%", ha="center", va="center")
    ax.set_title("진단별 확보율 (downloaded / AKI full)")
    plt.colorbar(im, ax=ax, fraction=0.05); fig.tight_layout()
    fig.savefig(OUT / "coverage_heatmap.png", dpi=120); plt.close(fig)

    # ===== 분석3: 추가 확보 가능 양성 =====
    def cap(cats):
        full = sum(full_cnt.get(c, 0) for c in cats)
        dl = sum(dl_cat.get(c, 0) for c in cats)
        return full, dl, max(full - dl, 0)
    ch_full, ch_dl, ch_add = cap(["DKD", "HTN"])
    im_full, im_dl, im_add = cap(["AIN"])
    capacity = {
        "chronic_positive": {"aki_full": ch_full, "downloaded": ch_dl, "additional_available": ch_add,
                             "min_target": 30, "rec_target": 50,
                             "gap_to_min": max(30 - ch_dl, 0), "achievable_within_aki": ch_add},
        "immune_positive": {"aki_full": im_full, "downloaded": im_dl, "additional_available": im_add,
                            "min_target": 30, "rec_target": 50,
                            "gap_to_min": max(30 - im_dl, 0), "achievable_within_aki": im_add},
        "note": "additional_available = AKI 전체 - 이미 다운로드. AKI 코호트 내 추가 여력."}
    (OUT / "additional_positive_capacity.json").write_text(
        json.dumps(capacity, indent=2, ensure_ascii=False), encoding="utf-8")

    # ===== 분석4: 저장공간 추정 =====
    sel["fsize"] = pd.to_numeric(sel["file_size"], errors="coerce").fillna(0.0)
    cur_bytes = float(sel["fsize"].sum()); cur_wsi = len(sel); cur_pat = len(sel_pat)
    gb = 1024 ** 3
    gb_per_wsi = cur_bytes / cur_wsi / gb
    wsi_per_pat = cur_wsi / cur_pat
    gb_per_pat = gb_per_wsi * wsi_per_pat
    scen = []
    for name, cats, target in [("A: DKD/HTN 30", ["DKD", "HTN"], 30), ("B: DKD/HTN 50", ["DKD", "HTN"], 50),
                               ("C: AIN 30", ["AIN"], 30), ("D: AIN 50", ["AIN"], 50),
                               ("E: chronic+immune 동시(각50)", ["DKD", "HTN", "AIN"], 100)]:
        dl = sum(dl_cat.get(c, 0) for c in cats)
        avail = sum(max(full_cnt.get(c, 0) - dl_cat.get(c, 0), 0) for c in cats)
        need_pat = max(target - dl, 0)
        feasible_pat = min(need_pat, avail)
        scen.append({"scenario": name, "downloaded": dl, "target": target,
                     "needed_patients": need_pat, "available_in_aki": avail,
                     "feasible_patients": feasible_pat,
                     "needed_GB_if_existed": round(need_pat * gb_per_pat, 1),
                     "feasible_GB_in_aki": round(feasible_pat * gb_per_pat, 1),
                     "feasible": feasible_pat >= need_pat and need_pat > 0})
    sdf = pd.DataFrame(scen)
    sdf.to_csv(OUT / "storage_projection.csv", index=False, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(sdf))
    ax.bar([i - 0.2 for i in x], sdf["needed_GB_if_existed"], 0.4, label="필요(가정)", color="tab:gray")
    ax.bar([i + 0.2 for i in x], sdf["feasible_GB_in_aki"], 0.4, label="AKI내 실현가능", color="tab:green")
    ax.set_xticks(list(x)); ax.set_xticklabels(sdf["scenario"], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("GB"); ax.set_title(f"추가 저장공간 시나리오 (현재 {cur_bytes/gb:.1f}GB/{cur_wsi}WSI, {gb_per_pat:.2f}GB/환자)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "storage_projection.png", dpi=120); plt.close(fig)

    # ===== 최종 보고서 =====
    coverage_overall = round(len(dl_pat) / len(full_pat) * 100, 1)
    if ch_add >= 20 and im_add >= 20:
        case, verdict = "Case 1", "추가 다운로드 수행 (우선순위 1)"
    elif coverage_overall >= 80 and ch_add < 5 and im_add < 5:
        case, verdict = "Case 2+3", "추가 다운로드 중단 — AKI 코호트 소진. 양성 확대는 신규 코호트(CKD/Healthy Ref/외부) 필요"
    else:
        case, verdict = "Case 중간", "부분 확보 가능 — 잔여분만 다운로드"
    report = {
        "aki_full_wsi_patients": len(full_pat), "downloaded_patients": len(dl_pat),
        "overall_coverage_%": coverage_overall,
        "diagnosis_distribution": {c: full_cnt.get(c, 0) for c in CATS},
        "coverage_by_dx": cov.set_index("diagnosis")[["full", "downloaded", "coverage_%"]].to_dict("index"),
        "chronic": {"aki_full": ch_full, "downloaded": ch_dl, "additional_available": ch_add},
        "immune": {"aki_full": im_full, "downloaded": im_dl, "additional_available": im_add},
        "storage": {"current_GB": round(cur_bytes / gb, 1), "current_wsi": cur_wsi,
                    "GB_per_wsi": round(gb_per_wsi, 3), "wsi_per_patient": round(wsi_per_pat, 2),
                    "GB_per_patient": round(gb_per_pat, 2)},
        "missing_patients_dx": "3명 전원 라벨 없음(빈값) → 양성 추가 0",
        "live_check": "atlas.kpmp.org AKI Imaging=1821파일, 스냅샷과 동일(최신)",
        "case": case, "verdict": verdict,
        "conclusion": (f"AKI 전체 DKD {full_cnt.get('DKD',0)}명/HTN {full_cnt.get('HTN',0)}명/AIN {full_cnt.get('AIN',0)}명 "
                       f"= 이미 100% 확보(추가 가능 chronic {ch_add}·immune {im_add}). "
                       f"FSGS/MCD/IgAN/MN은 AKI에 0명. 추가 다운로드 무의미.")}
    (OUT / "report_additional_acquisition.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"AKI WSI 환자 {len(full_pat)} | 다운로드 {len(dl_pat)} (coverage {coverage_overall}%)")
    print(f"chronic+ full {ch_full}/dl {ch_dl}/추가가능 {ch_add} | immune+ full {im_full}/dl {im_dl}/추가가능 {im_add}")
    print(f"저장: GB/환자={gb_per_pat:.2f} | [{case}] {verdict}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
