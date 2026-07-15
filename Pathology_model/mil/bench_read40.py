"""40x 읽기: Top-K 흩어진 순서 vs 공간(y,x) 정렬 순서 속도 비교."""
import sys, time
sys.path.insert(0, "backend")
import numpy as np, pandas as pd, tifffile, zarr
from mil.embed_patches import read_region

df = pd.read_csv("patches_manifest.csv", dtype={"magnification": str})
# Top-K처럼 tissue_score 정렬 후 한 (환자,stain,슬라이드)에서 120개
g = df[(df.stain == "SILVER") & (df.magnification == "40")].sort_values("tissue_score", ascending=False)
sp = g.iloc[0].slide_path
sub = g[g.slide_path == sp].head(120)
scattered = [(int(r.tile_x), int(r.tile_y)) for r in sub.itertuples()]          # Top-K 순서(흩어짐)
ordered = sorted(scattered, key=lambda c: (c[1], c[0]))                          # 공간 정렬
rs, os_ = int(sub.iloc[0].read_size), int(sub.iloc[0].out_size)

with tifffile.TiffFile(sp) as t:
    za = zarr.open(t.series[0].aszarr(level=int(sub.iloc[0].source_level)), mode="r")
    t0 = time.time(); [read_region(za, x, y, rs, os_) for (x, y) in scattered]; t_sc = time.time() - t0
with tifffile.TiffFile(sp) as t:
    za = zarr.open(t.series[0].aszarr(level=int(sub.iloc[0].source_level)), mode="r")
    t0 = time.time(); [read_region(za, x, y, rs, os_) for (x, y) in ordered]; t_or = time.time() - t0
n = len(scattered)
print(f"scattered(Top-K순): {t_sc:.2f}s ({t_sc/n*1000:.0f}ms/패치)")
print(f"spatial-sorted    : {t_or:.2f}s ({t_or/n*1000:.0f}ms/패치)  -> {t_sc/max(t_or,1e-6):.1f}x 빠름")
