"""
#3 데이터 무결성 + #4 정합성 검증 (거버넌스)

검증 항목:
 [무결성 #3]
  - 파일 존재 / 0바이트 / 잘림(매직넘버 TIFF·BigTIFF)
  - SHA256 checksum 계산 -> checksums.csv (데이터 버전 baseline·향후 변조 탐지)
  - slide_id(file_id) 중복
  - metadata(환자 판독/임상) 존재 여부
  - manifest 일치(디스크 ↔ split_manifest 양방향)
 [정합성 #4]
  - 파일명 stain ↔ manifest workflow_type ↔ 폴더(_grp) 일치
  - stain 태그 불일치 자동 리포트

실패 항목은 quarantine 리스트로 남겨 학습에서 제외 가능하게 한다.
"""
import csv
import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "split_manifest.csv"
MANI = ROOT / "manifest_aki_full.csv"
CHECKSUMS = ROOT / "checksums.csv"
REPORT = ROOT / "integrity_report.json"
TIFF_MAGICS = (b"\x49\x49\x2a\x00", b"\x4d\x4d\x00\x2a",
               b"\x49\x49\x2b\x00", b"\x4d\x4d\x00\x2b")

# 파일명 stain 약어 -> 정규 workflow_type
STAIN_TO_WORKFLOW = {
    "HE": "H&E stain", "PAS": "PAS stain", "MT": "TRI stain",
    "SILVER": "SIL stain", "IF": "RGB max proj of 8-ch IF",
}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    df = pd.read_csv(SPLIT)
    full = pd.read_csv(MANI)
    pat_ids = set(full["redcap_id"].astype(str))

    issues = {"missing": [], "zero": [], "bad_magic": [], "dup_slide_id": [],
              "no_metadata": [], "stain_mismatch": [], "not_in_manifest": []}
    checksum_rows = []

    # slide_id(file_id) 중복
    dup = df["file_id"][df["file_id"].duplicated(keep=False)].unique().tolist()
    issues["dup_slide_id"] = dup

    # 디스크 ↔ manifest 양방향
    manifest_paths = set(df["slide_path"])
    disk_paths = set()
    for g in ("wsi_he", "wsi_pas", "wsi_mt", "wsi_silver", "wsi_if"):
        d = ROOT / "data/raw" / g
        if d.exists():
            for p in d.iterdir():
                if p.suffix in (".svs", ".tif"):
                    disk_paths.add(f"data/raw/{g}/{p.name}")
    issues["not_in_manifest"] = sorted(disk_paths - manifest_paths)

    n_ok = 0
    for _, r in df.iterrows():
        p = ROOT / r["slide_path"]
        # 무결성
        if not p.exists():
            issues["missing"].append(r["slide_path"]); continue
        sz = p.stat().st_size
        if sz == 0:
            issues["zero"].append(r["slide_path"]); continue
        with open(p, "rb") as f:
            head = f.read(4)
        if head not in TIFF_MAGICS:
            issues["bad_magic"].append(r["slide_path"]); continue
        # 정합성: 파일명 stain ↔ folder ↔ manifest workflow_type
        stain = r["stain"]
        folder_stain = r["_grp"].replace("wsi_", "").upper()
        exp_wf = STAIN_TO_WORKFLOW.get(stain)
        man_row = full[full["file_id"] == r["file_id"]]
        wf_ok = (not man_row.empty) and (man_row.iloc[0]["workflow_type"] == exp_wf)
        if (stain != folder_stain) or (not wf_ok):
            issues["stain_mismatch"].append(
                f"{r['slide_path']} (folder={folder_stain}, "
                f"wf={'' if man_row.empty else man_row.iloc[0]['workflow_type']})")
            continue
        # metadata 존재
        if str(r["patient_id"]) not in pat_ids:
            issues["no_metadata"].append(r["slide_path"]); continue
        # checksum
        checksum_rows.append({"slide_path": r["slide_path"], "file_id": r["file_id"],
                              "size": sz, "sha256": sha256(p)})
        n_ok += 1
        if n_ok % 50 == 0:
            print(f"  ...checksum {n_ok} done", flush=True)

    # checksums.csv 저장
    with open(CHECKSUMS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["slide_path", "file_id", "size", "sha256"])
        w.writeheader(); w.writerows(checksum_rows)

    total_issues = sum(len(v) for v in issues.values())
    report = {
        "dataset_version": "v1",
        "checked": str(date.today()),
        "slides_total": int(len(df)),
        "slides_passed": n_ok,
        "issues_total": total_issues,
        "issues": {k: (len(v) if isinstance(v, list) else v) for k, v in issues.items()},
        "issue_detail": {k: v[:20] for k, v in issues.items() if v},
        "checksums_file": CHECKSUMS.name,
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 60)
    print(" #3 무결성 + #4 정합성 검증 결과")
    print("=" * 60)
    print(f"통과: {n_ok}/{len(df)}   총 이슈: {total_issues}")
    for k, v in issues.items():
        mark = "OK" if not v else "FAIL"
        print(f"  [{mark}] {k}: {len(v) if isinstance(v,list) else v}")
    print(f"\nchecksums -> {CHECKSUMS.name}   리포트 -> {REPORT.name}")


if __name__ == "__main__":
    main()
