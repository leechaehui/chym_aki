"""학습 stain-정규화 reference 동결 (train/serve 일치용, 1회 실행).

문제: CdssEngine(ctranspath, PAS/HE_CT) 은 학습 시 embed_patches.py 가
  read 512 -> Reinhard(PAS)/Macenko(HE) @512 -> encoder 로 임베딩을 만들었는데,
  reference(색 맞출 대상 패치)를 그때그때 fit 하고 버려서 서빙(feature_extraction.py)이
  같은 정규화를 재현할 수 없다(reference 부재). 결과: train/serve 색공간 스큐.

해결: 학습과 '동일한' 선택 규칙으로 reference 패치를 다시 골라 정규화 target 통계만
  동결한다. Reinhard=무작위성0, Macenko=seed0 고정 -> 결정적 재현.
  - PAS  -> Reinhard target {t_mean, t_std}
  - HE_CT-> Macenko  target {t_HE, t_maxC}   (HE reference 로 fit, 서빙 tag 는 HE_CT)

선택 규칙(embed_patches.main 과 동일):
  patches_manifest -> mags {10,40}(+IF native) -> stains {HE,PAS,MT}
  -> tissue_score desc 정렬 -> 각 stain 첫 행(=tissue 최고 패치)을 reference 로 fit.
  (top-k 값과 무관: 정렬 후 첫 행은 top-k>=1 이면 항상 동일 패치)

출력: Pathology_model/models/cdss_shadow/stain_ref.json
사용: python -m mil.build_stain_ref
"""
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Pathology_model
from mil.cdss_paths import data_root
from mil.stain_norm import ReinhardNormalizer, MacenkoNormalizer

MAGS = ["10", "40"]                 # data_LOCKED.mags
MAIN_STAINS = {"HE", "PAS", "MT"}   # embed_patches MAIN
NORM_BY_STAIN = {"HE": "macenko", "PAS": "reinhard", "MT": "reinhard"}
# 서빙(ctranspath) tag -> 정규화를 fit 할 원본 stain
SERVE_TAG_REF_STAIN = {"PAS": "PAS", "HE_CT": "HE"}


def read_region(za, x, y, read_size, out_size):
    """embed_patches.read_region 과 동일: source_level 에서 read_size 읽고 out_size 로 resize."""
    import cv2
    p = np.asarray(za[y:y + read_size, x:x + read_size])[..., :3]
    if p.shape[:2] != (read_size, read_size):
        p = np.pad(p, ((0, read_size - p.shape[0]), (0, read_size - p.shape[1]), (0, 0)),
                   mode="edge")
    if read_size != out_size:
        p = cv2.resize(p, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return p.astype(np.uint8)


def fit_reference(sel, stain):
    """embed_patches.fit_reference 와 동일: 해당 stain 첫 행 패치로 normalizer fit."""
    sub = sel[sel["stain"] == stain]
    if sub.empty:
        return None
    r = sub.iloc[0]
    with tifffile.TiffFile(r["slide_path"]) as t:
        za = zarr.open(t.series[0].aszarr(level=int(r["source_level"])), mode="r")
        ref = read_region(za, int(r["tile_x"]), int(r["tile_y"]),
                          int(r["read_size"]), int(r["out_size"]))
    if NORM_BY_STAIN[stain] == "macenko":
        try:
            return MacenkoNormalizer().fit(ref), ref, r
        except Exception:
            return ReinhardNormalizer().fit(ref), ref, r
    return ReinhardNormalizer().fit(ref), ref, r


def _serialize(norm):
    if isinstance(norm, MacenkoNormalizer):
        return {"method": "macenko",
                "t_HE": np.asarray(norm.t_HE, dtype=float).tolist(),
                "t_maxC": np.asarray(norm.t_maxC, dtype=float).tolist(),
                "Io": norm.Io, "beta": norm.beta, "alpha": norm.alpha}
    if isinstance(norm, ReinhardNormalizer):
        return {"method": "reinhard",
                "t_mean": np.asarray(norm.t_mean, dtype=float).tolist(),
                "t_std": np.asarray(norm.t_std, dtype=float).tolist()}
    raise TypeError(type(norm))


def main():
    import pandas as pd
    ROOT = data_root()
    man = ROOT / "patches_manifest.csv"
    print(f"[manifest] {man}", flush=True)
    pm = pd.read_csv(man, dtype={"magnification": str})

    sel = pm[(pm["magnification"].isin(MAGS)) |
             ((pm["stain"] == "IF") & (pm["magnification"] == "native"))].copy()
    sel = sel[sel["stain"].isin(MAIN_STAINS)]
    # embed_patches 의 top-k 정렬과 동일(정렬만 재현 — reference 는 top-k 값 무관)
    sel = sel.sort_values("tissue_score", ascending=False)

    out = {"_purpose": "CdssEngine(ctranspath) train/serve stain-norm reference (frozen)",
           "_source_manifest": str(man),
           "_select_rule": "mags{10,40}+MAIN{HE,PAS,MT}, tissue_score desc, stain 첫 행 reference",
           "refs": {}}

    for tag, ref_stain in SERVE_TAG_REF_STAIN.items():
        res = fit_reference(sel, ref_stain)
        if res is None:
            print(f"  [skip] {tag}: stain '{ref_stain}' 매니페스트에 없음", flush=True)
            continue
        norm, ref_img, row = res
        entry = _serialize(norm)
        entry["ref_stain"] = ref_stain
        entry["ref_from"] = {"slide_id": str(row.get("slide_id", "")),
                             "slide_path": str(row["slide_path"]),
                             "magnification": str(row["magnification"]),
                             "tile_x": int(row["tile_x"]), "tile_y": int(row["tile_y"]),
                             "tissue_score": float(row.get("tissue_score", float("nan")))}
        out["refs"][tag] = entry
        print(f"  [ok] {tag:6} <- {ref_stain} ({entry['method']}) "
              f"tissue={entry['ref_from']['tissue_score']:.4f} "
              f"slide={entry['ref_from']['slide_id']} tile=({row['tile_x']},{row['tile_y']})", flush=True)

    shadow = Path(__file__).resolve().parent.parent / "models" / "cdss_shadow"
    shadow.mkdir(parents=True, exist_ok=True)
    out_path = shadow / "stain_ref.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
