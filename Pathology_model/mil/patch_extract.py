"""
Phase B — 512 multi-scale 패치 좌표 추출 (계보 #6 완비, 픽셀 저장 없이 좌표만)

배율 10/20/30/40x 좌표를 모두 생성(브라이트필드 SVS). 배율 X의 목표 MPP =
base_mpp * 40/X (슬라이드별 base 0.253/0.263 자동 반영). 실제 패치는 '소스레벨에서
read_size 픽셀 읽어 512로 resize'(진짜 리샘플, 단순확대 금지). IF는 MPP 없어 단일 native.

tissue 판정은 썸네일 마스크로 white/empty만 제거(빠름). blur/out-of-focus/artifact는
패치 픽셀이 필요하므로 embed 단계에서 추가 필터.

출력 컬럼(패치 계보):
 patient_id, slide_id, stain, slide_path, magnification, source_level, read_size,
 out_size(=512), base_mpp, target_mpp, tile_x, tile_y  (tile_x/y=source_level 픽셀좌표)
"""
import numpy as np
import tifffile
import zarr

DEFAULT_MAGS = [10, 20, 30, 40]
OUT_SIZE = 512


def _base_mpp(tif):
    try:
        desc = tif.pages[0].description or ""
        for part in desc.replace("\n", "|").split("|"):
            if "MPP" in part and "=" in part:
                return float(part.split("=")[1].strip())
    except Exception:
        pass
    return float("nan")


def _tissue_mask(thumb, sat_thresh=0.10, val_thresh=0.95):
    import cv2
    if thumb.ndim == 2:
        thumb = np.stack([thumb] * 3, -1)
    hsv = cv2.cvtColor(thumb[..., :3], cv2.COLOR_RGB2HSV).astype(np.float32)
    return (hsv[..., 1] / 255.0 > sat_thresh) & (hsv[..., 2] / 255.0 < val_thresh)


def _plan_for_mag(shapes, base_mpp, X, out_size):
    """배율 X에 대한 (source_level, read_size, target_mpp). 항상 다운샘플(업샘플 금지).
    downsample 기반 선택 + 2% 허용오차(피라미드 downsample의 부동소수점 오차 흡수)."""
    downs = [shapes[0][0] / sh[0] for sh in shapes]   # L0 기준 각 레벨 downsample
    target_ds = 40.0 / X                              # 40x(L0) 기준 목표 downsample
    cand = [i for i, d in enumerate(downs) if d <= target_ds * 1.02]
    if not cand:
        cand = [0]
    src = max(cand, key=lambda i: downs[i])           # target 이하에서 가장 거친 레벨
    read_size = int(round(out_size * target_ds / downs[src]))
    target_mpp = base_mpp * target_ds
    return src, max(read_size, 1), target_mpp


def extract_slide_patches(slide_path, stain, patient_id, slide_id,
                          mags=DEFAULT_MAGS, out_size=OUT_SIZE, min_tissue_frac=0.5):
    with tifffile.TiffFile(slide_path) as t:
        s = t.series[0]
        shapes = [lv.shape for lv in s.levels]
        base_mpp = _base_mpp(t)
        thumb = np.asarray(zarr.open(s.levels[-1].aszarr(), mode="r")[:])
    mask = _tissue_mask(thumb)
    th, tw = mask.shape
    H0, W0 = shapes[0][0], shapes[0][1]

    # IF 등 MPP 미상 -> 단일 native 스케일(레벨0, 512)
    if np.isnan(base_mpp):
        plans = [("native", 0, out_size, float("nan"))]
    else:
        plans = []
        for X in mags:
            src, rs, tmpp = _plan_for_mag(shapes, base_mpp, X, out_size)
            plans.append((X, src, rs, tmpp))

    rows = []
    for mag, src, read_size, tmpp in plans:
        Hs, Ws = shapes[src][0], shapes[src][1]
        ny, nx = Hs // read_size, Ws // read_size
        for iy in range(ny):
            for ix in range(nx):
                y, x = iy * read_size, ix * read_size
                # source 좌표 -> 썸네일 마스크 영역 tissue 비율 (전체 슬라이드 비율 동일)
                fy0 = (y / Hs); fy1 = ((y + read_size) / Hs)
                fx0 = (x / Ws); fx1 = ((x + read_size) / Ws)
                my0, my1 = int(fy0 * th), max(int(fy0 * th) + 1, int(fy1 * th))
                mx0, mx1 = int(fx0 * tw), max(int(fx0 * tw) + 1, int(fx1 * tw))
                ts = float(mask[my0:my1, mx0:mx1].mean())   # 조직 밀도(Top-K 점수)
                if ts < min_tissue_frac:
                    continue
                rows.append({
                    "patient_id": patient_id, "slide_id": slide_id, "stain": stain,
                    "slide_path": slide_path, "magnification": mag,
                    "source_level": src, "read_size": read_size, "out_size": out_size,
                    "base_mpp": base_mpp,
                    "target_mpp": round(float(tmpp), 4) if not np.isnan(tmpp) else "",
                    "tile_x": x, "tile_y": y, "tissue_score": round(ts, 4),
                })
    return rows


def _main():
    import argparse
    import pandas as pd
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from mil.cdss_paths import data_root
        ROOT = data_root()
    except Exception:
        ROOT = Path("c:/team/chym_aki")

    ap = argparse.ArgumentParser()
    ap.add_argument("--mags", default="10,20,30,40")
    ap.add_argument("--out-size", type=int, default=512)
    ap.add_argument("--min-tissue", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "patches_manifest.csv"))
    args = ap.parse_args()
    mags = [int(m) for m in args.mags.split(",")]

    root = ROOT
    df = pd.read_csv(root / "split_manifest.csv")
    if args.limit:
        df = df.head(args.limit)
    all_rows = []
    for i, r in enumerate(df.itertuples(), 1):
        p = str(root / r.slide_path)
        try:
            all_rows.extend(extract_slide_patches(p, r.stain, r.patient_id, r.file_id,
                                                  mags, args.out_size, args.min_tissue))
        except Exception as e:
            print(f"  [실패] {r.slide_path}: {e}", flush=True)
        if i % 30 == 0:
            print(f"  {i}/{len(df)} 슬라이드, 누적 패치 {len(all_rows)}", flush=True)
    out = pd.DataFrame(all_rows)
    out.to_csv(args.out, index=False, encoding="utf-8")
    print(f"\n패치 좌표 {len(out)}개 -> {args.out}")
    print("배율별 패치 수:")
    print(out.groupby("magnification").size().to_string())
    print("\n배율×stain (패치 수):")
    print(out.groupby(["magnification", "stain"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    _main()
