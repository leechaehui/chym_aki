"""
KPMP Atlas Enriched Manifest Fetcher (AKI cohort)

atlas.kpmp.org 의 Elastic App Search 엔진(atlas-repository)을 직접 호출하여
enrollment_category=AKI 코호트의 전체 파일 목록을 '판독/임상 메타데이터까지 포함해'
manifest_aki_full.csv 로 저장한다.

기존 pathology_img.csv(11컬럼)에는 없는 핵심 필드:
 - primary_adjudicated_category : KPMP 병리 판독 진단명 (etiology 라벨의 근거)
 - kdigo_stage, baseline_egfr, proteinuria, sample_type ...
 - redcap_id : 환자 단위 그룹핑(train/test 누수 방지)용 ID
"""
import csv
import sys
import time
import requests
from pathlib import Path

ENDPOINT = "https://atlas.kpmp.org/spatial-viewer/search/api/as/v1/engines/atlas-repository/search"
SEARCH_KEY = "search-vwz67uj2sf8h83h4y8i8j6g3"  # 번들에 공개된 read-only 검색키
OUT = (Path(__file__).resolve().parents[2] / "manifest_aki_full.csv")

FILTERS = {
    "all": [
        {"enrollment_category": ["AKI"]},
        {"data_type": ["Imaging", "Biomarker Data", "Clinical Study Data", "Pathology Study Data"]},
    ]
}

# 저장할 필드(있으면 raw 값을, 리스트면 ';'로 join)
FIELDS = [
    "file_id", "package_id", "file_name", "access", "data_format", "data_category",
    "data_type", "file_size", "workflow_type", "experimental_strategy", "sample_type",
    "redcap_id", "enrollment_category", "primary_adjudicated_category", "kdigo_stage",
    "baseline_egfr", "proteinuria", "albuminuria", "a1c", "diabetes_history",
    "hypertension_history", "age_binned", "sex", "race", "tissue_source", "protocol",
]


def flat(v):
    if v is None:
        return ""
    if isinstance(v, list):
        return ";".join(str(x) for x in v)
    return str(v)


def fetch_page(session, page, size=100):
    body = {"query": "", "filters": FILTERS, "page": {"size": size, "current": page}}
    r = session.post(
        ENDPOINT,
        headers={"Authorization": f"Bearer {SEARCH_KEY}", "Content-Type": "application/json"},
        json=body,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def main():
    session = requests.Session()
    first = fetch_page(session, 1)
    total = first["meta"]["page"]["total_results"]
    pages = -(-total // 100)
    print(f"AKI 코호트 총 파일: {total}건  ({pages} 페이지)")

    rows = []
    for p in range(1, pages + 1):
        data = first if p == 1 else fetch_page(session, p)
        for res in data["results"]:
            row = {f: flat(res.get(f, {}).get("raw")) for f in FIELDS}
            rows.append(row)
        print(f"  page {p}/{pages} -> 누적 {len(rows)}건")
        time.sleep(0.2)

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n저장 완료: {OUT}  ({len(rows)}건, {len(FIELDS)}컬럼)")


if __name__ == "__main__":
    main()
