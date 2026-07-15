"""
KPMP AKI 병리 WSI - 구조 균형 50GB 데이터셋 빌더 (MIL용)

manifest_aki_full.csv(enrollment_category=AKI 전체)에서
stain 비율 강제규칙 + 환자 단위 다양성(누수/중복 최소화)을 만족하도록
파일을 선정하고, 선택적으로 실제 다운로드까지 수행한다.

사용:
    python build_balanced_dataset.py            # dry-run: 선정 + 리포트만 (다운로드 X)
    python build_balanced_dataset.py --download  # 선정 후 실제 다운로드까지

핵심 규칙(검증 포함):
    H&E 30-40% / PAS 20-30% / Trichrome 15-25% / Silver 10-15% / IF 5-10%
    PAS <= H&E,  Silver+IF >= 15%,  Trichrome >= 15%
"""
import argparse
import sys
import requests
from pathlib import Path

import pandas as pd
from tqdm import tqdm

MANIFEST = (Path(__file__).resolve().parents[2] / "manifest_aki_full.csv")
BASE_DIR = (Path(__file__).resolve().parents[2] / "data/raw")
SELECTED_OUT = (Path(__file__).resolve().parents[2] / "selected_manifest.csv")
DOWNLOAD_URL = "https://atlas.kpmp.org/api/v1/file/download/{package_id}/{file_name}"

GiB = 1024 ** 3

# stain 그룹별: (출력 디렉토리, 목표 비율, 파일 필터 함수)
# 목표 총량 50GB 기준 비율 (합 = 1.00)
TARGET_TOTAL_GB = 50.0
STAIN_PLAN = {
    "wsi_he":     {"ratio": 0.35, "label": "H&E"},
    "wsi_pas":    {"ratio": 0.25, "label": "PAS"},
    "wsi_mt":     {"ratio": 0.18, "label": "Trichrome"},
    "wsi_silver": {"ratio": 0.13, "label": "Silver"},
    "wsi_if":     {"ratio": 0.09, "label": "IF"},
}


def classify(row):
    """행을 stain 그룹 디렉토리명으로 분류. 대상 아니면 None."""
    fmt = str(row["data_format"])
    wf = str(row["workflow_type"])
    # IF: 진단용은 아니지만 2D RGB max projection만 사용(3D/8ch는 과대용량 제외)
    if fmt == "tif" and wf == "RGB max proj of 8-ch IF":
        return "wsi_if"
    if fmt != "svs":
        return None
    # 순수 svs stain. Frozen H&E는 품질 사유로 제외(정규 H&E만)
    if wf == "H&E stain":
        return "wsi_he"
    if wf == "PAS stain":
        return "wsi_pas"
    if wf == "TRI stain":
        return "wsi_mt"
    if wf == "SIL stain":
        return "wsi_silver"
    # TOL / Frozen H&E / Other stain 등은 제외
    return None


def select_patient_diverse(df_grp, target_bytes):
    """
    환자 단위 라운드로빈 선정:
    각 환자에서 1장씩(작은 파일 우선) 돌아가며 골라 최대한 많은 환자를 커버.
    누적이 target_bytes를 넘기 직전까지 채운다.
    """
    # 환자별로 파일을 작은 것부터 정렬한 큐 구성
    buckets = {}
    for pid, sub in df_grp.groupby("redcap_id"):
        buckets[pid] = sub.sort_values("file_size").to_dict("records")
    # 환자 순서는 보유 슬라이드가 적은 환자 우선(희소 환자 보호) 후 ID순
    pid_order = sorted(buckets, key=lambda p: (len(buckets[p]), str(p)))

    selected = []
    acc = 0
    progressed = True
    while progressed:
        progressed = False
        for pid in pid_order:
            if not buckets[pid]:
                continue
            rec = buckets[pid][0]
            sz = rec["file_size"]
            if acc + sz > target_bytes:
                # 이 그룹 목표 도달: 더 못 넣으면 종료 판단
                continue
            selected.append(rec)
            acc += sz
            buckets[pid].pop(0)
            progressed = True
        # 한 바퀴 돌아도 아무도 못 넣었으면 종료
        if acc >= target_bytes:
            break
    return selected, acc


def validate_rules(ratios):
    he, pas, tri, sil, iff = (ratios[k] for k in
                              ["wsi_he", "wsi_pas", "wsi_mt", "wsi_silver", "wsi_if"])
    checks = [
        ("H&E 30-40%",       0.30 <= he <= 0.40),
        ("PAS 20-30%",       0.20 <= pas <= 0.30),
        ("Trichrome 15-25%", 0.15 <= tri <= 0.25),
        ("Silver 10-15%",    0.10 <= sil <= 0.15),
        ("IF 5-10%",         0.05 <= iff <= 0.10),
        ("PAS <= H&E",       pas <= he),
        ("Silver+IF >= 15%", (sil + iff) >= 0.15),
        ("Trichrome >= 15%", tri >= 0.15),
    ]
    return checks


def download(records):
    ok, fail = 0, 0
    for rec in records:
        grp = rec["_grp"]
        out_dir = BASE_DIR / grp
        out_dir.mkdir(parents=True, exist_ok=True)
        pid = str(rec["redcap_id"]).replace(";", "_")
        suffix = grp.replace("wsi_", "").upper()
        # package_id는 한 package의 여러 슬라이드가 공유해 충돌하므로 file_id(고유) 사용
        fid = str(rec["file_name"]).split("_")[0][:8]
        ext = ".tif" if grp == "wsi_if" else ".svs"
        dest = out_dir / f"{pid}_{suffix}_{fid}{ext}"
        if dest.exists() and dest.stat().st_size > 0:
            ok += 1
            continue
        url = DOWNLOAD_URL.format(package_id=rec["package_id"], file_name=rec["file_name"])
        try:
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(
                desc=dest.name, total=total, unit="iB", unit_scale=True, unit_divisor=1024
            ) as bar:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    bar.update(f.write(chunk))
            ok += 1
        except Exception as e:
            print(f"  실패: {dest.name} - {e}")
            if dest.exists():
                dest.unlink()
            fail += 1
    return ok, fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="선정 후 실제 다운로드 수행")
    args = ap.parse_args()

    if not MANIFEST.exists():
        print(f"오류: {MANIFEST} 없음. 먼저 fetch_kpmp_manifest.py 실행.")
        sys.exit(1)

    df = pd.read_csv(MANIFEST)
    df = df[df["access"] == "open"].copy()
    df["_grp"] = df.apply(classify, axis=1)
    pool = df[df["_grp"].notna()].copy()

    print("=" * 64)
    print("구조 균형 50GB 데이터셋 빌더 (KPMP AKI / MIL)")
    print("=" * 64)

    all_selected = []
    group_stats = {}
    for grp, cfg in STAIN_PLAN.items():
        target_bytes = int(TARGET_TOTAL_GB * cfg["ratio"] * GiB)
        df_grp = pool[pool["_grp"] == grp]
        if df_grp.empty:
            print(f"[경고] {cfg['label']}({grp}) 대상 파일 0개")
            group_stats[grp] = {"n": 0, "bytes": 0, "pts": 0}
            continue
        sel, acc = select_patient_diverse(df_grp, target_bytes)
        for r in sel:
            r["_grp"] = grp
        all_selected.extend(sel)
        pts = len({r["redcap_id"] for r in sel})
        group_stats[grp] = {"n": len(sel), "bytes": acc, "pts": pts,
                            "avail_gb": df_grp["file_size"].sum() / GiB,
                            "avail_pts": df_grp["redcap_id"].nunique()}

    total_bytes = sum(g["bytes"] for g in group_stats.values())
    total_gb = total_bytes / GiB

    # 1) stain별 개수/용량/비율
    print("\n[1] Stain 구성")
    print(f"{'그룹':<10}{'라벨':<11}{'파일':>6}{'용량GB':>9}{'비율':>8}{'환자':>6}{'가용GB':>9}")
    ratios = {}
    for grp, cfg in STAIN_PLAN.items():
        s = group_stats[grp]
        ratio = s["bytes"] / total_bytes if total_bytes else 0
        ratios[grp] = ratio
        print(f"{grp:<10}{cfg['label']:<11}{s['n']:>6}{s['bytes']/GiB:>9.2f}{ratio*100:>7.1f}%"
              f"{s['pts']:>6}{s.get('avail_gb',0):>9.1f}")
    print(f"{'합계':<21}{len(all_selected):>6}{total_gb:>9.2f}{'100.0%':>8}")

    # 2) 환자 커버리지
    uniq_pat = len({r["redcap_id"] for r in all_selected})
    print(f"\n[2] 환자 커버리지: 선정 {uniq_pat}명 / AKI 코호트 {df['redcap_id'].nunique()}명")

    # 3) 강제규칙 검증
    print("\n[3] 강제규칙 검증")
    all_pass = True
    for name, ok in validate_rules(ratios):
        print(f"   {'PASS' if ok else 'FAIL'}  {name}")
        all_pass = all_pass and ok
    print(f"   => {'모든 규칙 충족' if all_pass else '규칙 위반 있음(쿼터 조정 필요)'}")
    print(f"   총 용량 {total_gb:.2f}GB (허용 45-60GB): "
          f"{'OK' if 45 <= total_gb <= 60 else 'RANGE 위반'}")

    # 4) etiology(판독) 분포 - 선정된 환자 기준
    sel_pids = {str(r["redcap_id"]) for r in all_selected}
    pat_lvl = df[df["redcap_id"].astype(str).isin(sel_pids)].drop_duplicates("redcap_id")
    print("\n[4] 선정 환자 판독 카테고리 분포(primary_adjudicated_category)")
    print(pat_lvl["primary_adjudicated_category"].value_counts(dropna=False).to_string())

    # 선정 매니페스트 저장
    sel_df = pd.DataFrame(all_selected)
    cols = ["_grp", "redcap_id", "workflow_type", "data_format", "file_size",
            "primary_adjudicated_category", "kdigo_stage", "package_id", "file_name"]
    sel_df[cols].to_csv(SELECTED_OUT, index=False, encoding="utf-8")
    print(f"\n선정 매니페스트 저장: {SELECTED_OUT} ({len(sel_df)}건)")

    if args.download:
        print("\n[다운로드 시작]")
        ok, fail = download(all_selected)
        print(f"\n다운로드 완료: 성공 {ok} / 실패 {fail}")
    else:
        print("\n(dry-run) 실제 다운로드하려면 --download 옵션을 붙여 재실행하세요.")


if __name__ == "__main__":
    main()
