"""
KPMP 균형 데이터셋 무결성 검증 + 최종 보고

selected_manifest.csv 대비 실제 다운로드 파일을 검증한다.
 - 파일 존재/개수 (stain별)
 - 매직넘버: SVS/IF tif 모두 TIFF (II*\\0=49492A00 또는 MM\\0*=4D4D002A)
 - 0바이트/비정상적으로 작은 파일
 - 용량/비율 분포 + 8개 강제규칙 재검증
 - 환자 커버리지, etiology(판독) 분포
"""
from pathlib import Path
import pandas as pd

BASE = (Path(__file__).resolve().parents[2] / "data/raw")
SEL = (Path(__file__).resolve().parents[2] / "selected_manifest.csv")
MANI = (Path(__file__).resolve().parents[2] / "manifest_aki_full.csv")
GiB = 1024 ** 3
LABEL = {"wsi_he": "H&E", "wsi_pas": "PAS", "wsi_mt": "Trichrome",
         "wsi_silver": "Silver", "wsi_if": "IF"}
ORDER = ["wsi_he", "wsi_pas", "wsi_mt", "wsi_silver", "wsi_if"]
# 표준 TIFF(II*\0 / MM\0*) + BigTIFF(II+\0 / MM\0+). 대용량 SVS는 BigTIFF(2b) 사용.
TIFF_MAGICS = (b"\x49\x49\x2a\x00", b"\x4d\x4d\x00\x2a",
               b"\x49\x49\x2b\x00", b"\x4d\x4d\x00\x2b")


def dest_path(r):
    pid = str(r["redcap_id"]).replace(";", "_"); suf = r["_grp"].replace("wsi_", "").upper()
    fid = str(r["file_name"]).split("_")[0][:8]
    ext = ".tif" if r["_grp"] == "wsi_if" else ".svs"
    return BASE / r["_grp"] / f"{pid}_{suf}_{fid}{ext}"


def main():
    df = pd.read_csv(SEL)
    print("=" * 64)
    print(" KPMP AKI 병리 WSI 균형 데이터셋 — 최종 검증 보고")
    print("=" * 64)

    # ---- 무결성 스캔 ----
    missing, zero, badmagic, ok = [], [], [], []
    rows_by_grp = {g: [] for g in ORDER}
    for _, r in df.iterrows():
        p = dest_path(r)
        rows_by_grp[r["_grp"]].append((r, p))
        if not p.exists():
            missing.append(p.name); continue
        sz = p.stat().st_size
        if sz == 0:
            zero.append(p.name); continue
        with open(p, "rb") as f:
            head = f.read(4)
        if head not in TIFF_MAGICS:
            badmagic.append((p.name, head.hex())); continue
        ok.append((r, p, sz))

    # ---- [1] stain별 개수/용량/비율 ----
    print("\n[1] Stain 구성 (실측)")
    print(f"{'Stain':<11}{'파일':>7}{'목표':>6}{'용량GB':>9}{'비율':>8}{'환자':>6}")
    total_bytes = sum(sz for _, _, sz in ok)
    ratios = {}
    goal = {"wsi_he": 140, "wsi_pas": 108, "wsi_mt": 70, "wsi_silver": 32, "wsi_if": 21}
    for g in ORDER:
        items = [(r, p, sz) for (r, p, sz) in ok if r["_grp"] == g]
        b = sum(sz for _, _, sz in items)
        pts = len({r["redcap_id"] for r, _, _ in items})
        ratios[g] = b / total_bytes if total_bytes else 0
        print(f"{LABEL[g]:<11}{len(items):>7}{goal[g]:>6}{b/GiB:>9.2f}{ratios[g]*100:>7.1f}%{pts:>6}")
    n_ok = len(ok)
    print(f"{'합계':<11}{n_ok:>7}{'371':>6}{total_bytes/GiB:>9.2f}{'100.0%':>8}")

    # ---- [2] 무결성 ----
    print("\n[2] 무결성")
    print(f"  정상(TIFF 매직넘버 확인): {n_ok}/{len(df)}")
    print(f"  누락: {len(missing)}  0바이트: {len(zero)}  매직넘버 이상: {len(badmagic)}")
    for n in missing[:10]:
        print(f"    [누락] {n}")
    for n in zero[:10]:
        print(f"    [0byte] {n}")
    for n, h in badmagic[:10]:
        print(f"    [badmagic {h}] {n}")

    # ---- [3] 강제규칙 ----
    he, pas, tri, sil, iff = (ratios[k] for k in ORDER)
    checks = [
        ("H&E 30-40%", 0.30 <= he <= 0.40), ("PAS 20-30%", 0.20 <= pas <= 0.30),
        ("Trichrome 15-25%", 0.15 <= tri <= 0.25), ("Silver 10-15%", 0.10 <= sil <= 0.15),
        ("IF 5-10%", 0.05 <= iff <= 0.10), ("PAS <= H&E", pas <= he),
        ("Silver+IF >= 15%", sil + iff >= 0.15), ("Trichrome >= 15%", tri >= 0.15),
    ]
    print("\n[3] 강제규칙 검증 (실측 비율 기준)")
    for name, passed in checks:
        print(f"   {'PASS' if passed else 'FAIL'}  {name}")
    print(f"   총 용량 {total_bytes/GiB:.2f}GB (45-60GB): "
          f"{'OK' if 45 <= total_bytes/GiB <= 60 else 'RANGE 위반'}")

    # ---- [4] 환자 커버리지 + etiology ----
    sel_pids = {str(r["redcap_id"]) for r, _, _ in ok}
    full = pd.read_csv(MANI)
    cohort = full["redcap_id"].nunique()
    print(f"\n[4] 환자 커버리지: {len(sel_pids)}명 / AKI 코호트 {cohort}명")
    patlvl = full[full["redcap_id"].astype(str).isin(sel_pids)].drop_duplicates("redcap_id")
    print("    판독(primary_adjudicated_category) 분포:")
    for k, v in patlvl["primary_adjudicated_category"].value_counts(dropna=False).items():
        kk = str(k)[:55]
        print(f"      {v:>3}  {kk}")


if __name__ == "__main__":
    main()
