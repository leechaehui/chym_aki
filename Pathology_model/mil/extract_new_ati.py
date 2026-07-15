"""
신규 ATI 환자(미처리 raw) 40x 패치 좌표 추출 → patches_manifest.csv 에 append.
기존 72명은 건드리지 않고(중복 제거), 신규만 추가. 이후 embed_patches --mags 40 --topk 100 로 임베딩.
"""
import re, sys, json
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path("c:/team/chym_aki/Pathology_model")))
from mil.patch_extract import extract_slide_patches

ROOT = Path("D:/cdss_core")
MANI = ROOT / "patches_manifest.csv"
ROOTS = [Path("D:/cdss_core/data_raw"), Path("D:/chym_aki_data")]
ATI25 = ("28-10051,28-12178,28-12263,28-12545,28-12663,31-10000,31-10061,31-10063,31-10298,"
         "31-10340,31-10572,31-10645,933-10053,933-10182,933-10281,933-10373,933-10441,"
         "933-10566,933-10661,32-10419,32-10459,33-10376,34-11008,34-11049,34-11093").split(",")
STAINS = ("HE", "PAS", "MT")


def main():
    # 인벤토리: pid -> stain -> (path, slide_id)
    inv = {}
    for root in ROOTS:
        if not root.exists():
            continue
        for f in root.rglob("*.svs"):
            m = re.match(r"(\d+-\d+)_([A-Za-z]+)_([0-9a-fA-F]+)", f.name)
            if not m:
                continue
            pid, st, fid = m.group(1), m.group(2).upper(), m.group(3)
            if pid in ATI25 and st in STAINS:
                inv.setdefault(pid, {}).setdefault(st, (str(f), fid))

    new_rows = []
    for pid in ATI25:
        for st, (path, fid) in inv.get(pid, {}).items():
            try:
                rows = extract_slide_patches(path, st, pid, fid, mags=[40], min_tissue_frac=0.5)
                new_rows.extend(rows)
                print(f"  {pid}/{st}: {len(rows)} tiles", flush=True)
            except Exception as e:
                print(f"  [FAIL] {pid}/{st}: {e}", flush=True)

    new_df = pd.DataFrame(new_rows)
    print(f"\n신규 40x 패치 {len(new_df)}행, 환자 {new_df['patient_id'].nunique()}, "
          f"(stain별 {new_df.groupby('stain').size().to_dict()})", flush=True)

    old = pd.read_csv(MANI, dtype={"magnification": str})
    new_df["magnification"] = new_df["magnification"].astype(str)
    # 기존 컬럼 순서에 맞춤
    new_df = new_df[[c for c in old.columns if c in new_df.columns]]
    merged = pd.concat([old, new_df], ignore_index=True)
    merged = merged.drop_duplicates(
        subset=["patient_id", "slide_id", "stain", "magnification", "tile_x", "tile_y"])
    merged.to_csv(MANI, index=False, encoding="utf-8")
    print(f"patches_manifest 갱신: {len(old)} -> {len(merged)} 행 "
          f"(환자 {old['patient_id'].nunique()} -> {merged['patient_id'].nunique()})", flush=True)


if __name__ == "__main__":
    main()
