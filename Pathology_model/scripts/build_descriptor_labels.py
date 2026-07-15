"""
KPMP TIV Descriptor Scores → 병리 정량 라벨 테이블 (RQ2 염증 / RQ3 섬유화·위축)

기존 진단 카테고리(primary_adjudicated_category, chronic 양성 10명뿐) 대신, KPMP 병리위원회
반정량 점수(연속 % / ordinal)를 우리 코호트(83명)에 결합한다. 섬유화/위축/염증을 회귀·서열
타깃으로 쓰면 chronic 이진(10명)보다 표본·통계력이 크게 개선된다.

입력 : artifacts/kpmp_descriptor_scores.xlsx (Data Table 시트, 멀티행 헤더)
출력 : artifacts/descriptor_labels.csv  + artifacts/descriptor_labels_coverage.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path(__file__).resolve().parent.parent / "artifacts"   # 코드 상대경로(이식성)

# 큐레이션: (열 인덱스, 출력 컬럼명, 그룹). 열 인덱스는 Data Table(header=None) 기준.
CURATED = [
    (3, "cortex_pct", "global"), (4, "medulla_pct", "global"),
    # --- RQ3 만성(섬유화/위축/손상) ---
    (5, "interstitial_fibrosis_pct", "chronic"),
    (12, "tubular_atrophy", "chronic"),
    (18, "tubular_injury_pct", "chronic"),
    (125, "arteriosclerosis", "chronic"),
    # --- ATI 세부 소견(급성세관손상 specific, presence 0/1) — 멀티태스크 보조헤드용 ---
    (19, "ti_simplification", "ati"),
    (20, "ti_cell_sloughing", "ati"),
    (21, "ti_necrosis_apoptosis", "ati"),
    (22, "ti_detachment_denudation", "ati"),
    # --- RQ2 염증 ---
    (31, "tubulitis", "inflammation"),
    (32, "tubulitis_lymphocytic", "inflammation"),
    (33, "tubulitis_neutrophilic", "inflammation"),
    (60, "interstitial_mononuclear_wbc_pct", "inflammation"),
    (67, "interstitial_eosinophils", "inflammation"),
    (71, "interstitial_neutrophils", "inflammation"),
    (75, "interstitial_granulomas", "inflammation"),
    (140, "vascular_inflammation", "inflammation"),
]


def main():
    raw = pd.read_excel(ART / "kpmp_descriptor_scores.xlsx", sheet_name="Data Table", header=None)
    units = raw.iloc[2].astype(str)            # 행2: descriptor + 단위
    data = raw.iloc[4:].reset_index(drop=True)  # 데이터: 행4부터
    pid = data.iloc[:, 0].astype(str).str.strip()

    out = pd.DataFrame({"patient_id": pid})
    label_units = {}
    for ci, name, grp in CURATED:
        out[name] = pd.to_numeric(data.iloc[:, ci], errors="coerce")
        label_units[name] = {"group": grp, "source_header": str(units.iloc[ci]).strip()}
    out = out[out["patient_id"].str.match(r"^\d+-\d+")]   # 유효 participant 행만

    # 우리 코호트 결합 표시
    sel = pd.read_csv(ART / "selected_manifest.csv", dtype=str).fillna("")
    sm = pd.read_csv(ART / "split_manifest.csv", dtype=str).fillna("")
    downloaded = set(sel["redcap_id"].str.strip())
    trained = set(sm[pd.to_numeric(sm["fold"], errors="coerce") >= 0]["patient_id"].str.strip())
    out["in_downloaded_83"] = out["patient_id"].isin(downloaded)
    out["in_trained_cv"] = out["patient_id"].isin(trained)
    out.to_csv(ART / "descriptor_labels.csv", index=False, encoding="utf-8")

    # 커버리지 요약(우리 다운로드 83명 기준)
    ours = out[out["in_downloaded_83"]]
    cov = {"descriptor_participants_total": int(out["patient_id"].nunique()),
           "downloaded_83_with_scores": int(len(ours)),
           "trained_cv_with_scores": int(out["in_trained_cv"].sum()),
           "labels": {}}
    for ci, name, grp in CURATED:
        v = ours[name]
        nn = int(v.notna().sum())
        info = {"group": grp, "n_labeled_in_83": nn, "header": label_units[name]["source_header"]}
        if nn:
            info.update({"min": round(float(v.min()), 1), "max": round(float(v.max()), 1),
                         "mean": round(float(v.mean()), 1),
                         "nunique": int(v.nunique())})
        cov["labels"][name] = info
    (ART / "descriptor_labels_coverage.json").write_text(
        json.dumps(cov, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"descriptor 참가자 {cov['descriptor_participants_total']} | "
          f"우리 83명 중 점수보유 {cov['downloaded_83_with_scores']} | "
          f"CV대상 중 {cov['trained_cv_with_scores']}")
    print("\n[그룹별 라벨 가용(우리 83명)]")
    for name, info in cov["labels"].items():
        rng = f" range {info.get('min')}~{info.get('max')} mean {info.get('mean')}" if "mean" in info else ""
        print(f"  [{info['group']:12s}] {name:34s} n={info['n_labeled_in_83']}{rng}")
    print(f"\n-> descriptor_labels.csv, descriptor_labels_coverage.json")


if __name__ == "__main__":
    main()
