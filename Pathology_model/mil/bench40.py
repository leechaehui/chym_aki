"""40x read(L0) vs norm vs encode 비용 측정 -> 병목 특정."""
import sys, time
sys.path.insert(0, "backend")
import numpy as np, pandas as pd, tifffile, zarr, torch
from mil.embed_patches import read_region
from mil.stain_norm import MacenkoNormalizer, ReinhardNormalizer
from mil.encoders import build_encoder

df = pd.read_csv("patches_manifest.csv", dtype={"magnification": str})
dev = "cuda"
model, dim, pp = build_encoder("ctranspath", dev)

for stain, Norm in [("HE", MacenkoNormalizer), ("SILVER", ReinhardNormalizer)]:
    g = df[(df.stain == stain) & (df.magnification == "40")]
    sp = g.iloc[0].slide_path
    sub = g[g.slide_path == sp].head(60)
    with tifffile.TiffFile(sp) as t:
        za = zarr.open(t.series[0].aszarr(level=int(sub.iloc[0].source_level)), mode="r")
        t0 = time.time()
        raw = [read_region(za, int(r.tile_x), int(r.tile_y), int(r.read_size), int(r.out_size))
               for r in sub.itertuples()]
        t_read = time.time() - t0
    nz = Norm().fit(raw[0])
    t0 = time.time(); normed = [nz.transform(p) for p in raw]; t_norm = time.time() - t0
    t0 = time.time()
    for i in range(0, len(normed), 64):
        with torch.no_grad():
            model(pp(np.stack(normed[i:i+64])))
    torch.cuda.synchronize(); t_enc = time.time() - t0
    n = len(raw)
    print(f"{stain}: {n}패치 | read {t_read:.2f}s({t_read/n*1000:.0f}ms) | "
          f"norm {t_norm:.2f}s({t_norm/n*1000:.0f}ms) | enc {t_enc:.2f}s({t_enc/n*1000:.0f}ms)", flush=True)
