"""
Phase B(임베딩) — 512 multi-scale, encoder registry, embedding cache, 멀티프로세싱

좌표(patches_manifest)를 읽어:
 1) source_level에서 read_size 영역 읽고 out_size(512)로 resize(진짜 리샘플)
 2) per-stain 정규화 (H&E=Macenko, 그 외=Reinhard)
 3) encoder(resnet50/ctranspath/dinov2)로 임베딩
 4) (encoder, magnification, patient, stain) 캐시 .npy + index.csv (배율 병합)

병렬화: ProcessPoolExecutor(--workers). 각 워커는 자기 tifffile 핸들(동시읽기 세그폴트 회피)·
자기 BLAS(정규화 thread-unsafe 회피)·자기 모델(GPU)로 read+norm+encode 후 임베딩만 반환(IPC 경량).
워커 내부는 청크 스트리밍으로 메모리 억제. main은 CUDA 미사용(refs만 계산).

사용: python embed_patches.py --encoder ctranspath --mags 40 --topk 2000 --workers 4
"""
import os
# numpy/BLAS import 전에 스레드 1로 제한(멀티프로세싱 oversubscription 방지)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile
import zarr

sys.path.insert(0, str(Path("c:/team/chym_aki/Pathology_model")))
from mil.stain_norm import ReinhardNormalizer, MacenkoNormalizer
from mil.encoders import check_availability, EMBED_DIM

try:
    from mil.cdss_paths import data_root as _data_root, p as _p
    ROOT = _data_root()
    EMB_ROOT = _p("embeddings")
except Exception:
    ROOT = Path("c:/team/chym_aki")
    EMB_ROOT = ROOT / "data/embeddings"
SEED = 42
CHUNK = 128
ENC_SIZE = 224   # 정규화·인코딩을 인코더 입력해상도(224)에서 수행(512norm 불필요, norm ~5x↑)
NORM_BY_STAIN = {"HE": "macenko", "PAS": "reinhard", "MT": "reinhard",
                 "SILVER": "reinhard", "IF": "reinhard"}


def read_region(za, x, y, read_size, out_size):
    p = np.asarray(za[y:y + read_size, x:x + read_size])[..., :3]
    if p.shape[:2] != (read_size, read_size):
        p = np.pad(p, ((0, read_size - p.shape[0]), (0, read_size - p.shape[1]), (0, 0)),
                   mode="edge")
    if read_size != out_size:
        p = cv2.resize(p, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return p.astype(np.uint8)


def _safe_norm(p, norm):
    if norm is None:
        return p
    try:
        return norm.transform(p)
    except Exception:
        return p


def fit_reference(sel, stain):
    """stain 정규화 reference = 해당 stain 첫 행의 패치 (main에서 1회, CUDA 불필요)."""
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
            return MacenkoNormalizer().fit(ref)
        except Exception:
            return ReinhardNormalizer().fit(ref)
    return ReinhardNormalizer().fit(ref)


# ---- 워커 (별도 프로세스) ----
_W = {}


def _init_worker(encoder, refs, chunk, enc_size=224):
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    import torch
    from mil.encoders import build_encoder
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, dim, pp = build_encoder(encoder, dev, img_size=enc_size)
    _W.update(torch=torch, model=model, dim=dim, pp=pp, dev=dev, refs=refs, chunk=chunk)


def _process_group(task):
    pid, stain, mag, spath, slevel, rsize, osize, coords = task
    torch = _W["torch"]; model = _W["model"]; pp = _W["pp"]; chunk = _W["chunk"]
    norm = _W["refs"].get(stain)
    rs, os_ = int(rsize), int(osize)
    feats = []
    with tifffile.TiffFile(spath) as t:
        za = zarr.open(t.series[0].aszarr(level=int(slevel)), mode="r")
        for i in range(0, len(coords), chunk):       # 청크 스트리밍(메모리 억제)
            sub = coords[i:i + chunk]
            # 스펙: norm은 512 space 고정. read 512 → norm@512 → pp가 224로 다운샘플
            normed = [_safe_norm(read_region(za, x, y, rs, os_), norm) for (x, y) in sub]
            x = pp(np.stack(normed))
            with torch.no_grad():
                feats.append(model(x).cpu().numpy())
    arr = np.concatenate(feats, 0).astype(np.float32) if feats else \
        np.zeros((0, _W["dim"]), np.float32)
    return (str(pid), str(stain), str(mag), arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--mags", default="10")
    ap.add_argument("--stains", default="", help="콤마구분(비우면 전체). 예: HE,PAS,MT")
    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=64, help="워커 인코딩 배치(GPU 메모리)")
    ap.add_argument("--enc-size", type=int, default=224,
                    help="인코더 입력 해상도(예: 512). !=224면 '{encoder}_s{size}' 별도 디렉터리에 저장")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="처리 그룹 수 제한(테스트)")
    args = ap.parse_args()

    ok, reason = check_availability(args.encoder)
    if not ok:
        print(f"[중단] encoder '{args.encoder}' 사용 불가: {reason}"); sys.exit(2)
    mags = [str(m) for m in args.mags.split(",")]
    embed_dim = EMBED_DIM[args.encoder]
    enc_tag = args.encoder if args.enc_size == 224 else f"{args.encoder}_s{args.enc_size}"
    out_dir = EMB_ROOT / enc_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    def npy_path(pid, stain, mag):
        return out_dir / f"{str(pid).replace(';','_')}__{stain}__m{mag}.npy"

    pm = pd.read_csv(ROOT / "patches_manifest.csv", dtype={"magnification": str})
    sel = pm[(pm["magnification"].isin(mags)) |
             ((pm["stain"] == "IF") & (pm["magnification"] == "native"))].copy()
    if args.stains:                                  # MAIN: HE,PAS,MT (Silver=optional, IF 제외)
        keep = set(s.strip() for s in args.stains.split(","))
        sel = sel[sel["stain"].isin(keep)]
    if args.topk and "tissue_score" in sel.columns:
        before = len(sel)
        sel = (sel.sort_values("tissue_score", ascending=False)
                  .groupby(["patient_id", "stain", "magnification"], sort=False).head(args.topk).copy())
        print(f"Top-K={args.topk}: {before} -> {len(sel)} 패치", flush=True)

    refs = {s: fit_reference(sel, s) for s in NORM_BY_STAIN if (sel["stain"] == s).any()}

    target_keys = set(tuple(map(str, x)) for x in
                      sel[["patient_id", "stain", "magnification"]].drop_duplicates().values)
    skip = set()
    if not args.overwrite:
        skip = {k for k in target_keys if npy_path(*k).exists()}
    print(f"encoder={args.encoder}({embed_dim}d) workers={args.workers} mags={mags}+IFnative | "
          f"그룹대상 {len(target_keys)}개, 캐시skip {len(skip)}", flush=True)

    gcols = ["patient_id", "stain", "magnification", "slide_path", "source_level", "read_size", "out_size"]
    tasks = []
    for key, g in sel.groupby(gcols):
        pid, stain, mag, spath, slevel, rsize, osize = key
        if (str(pid), str(stain), str(mag)) in skip:
            continue
        coords = [(int(r.tile_x), int(r.tile_y)) for r in g.itertuples()]
        tasks.append((pid, stain, mag, spath, slevel, rsize, osize, coords))
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"임베딩 그룹 {len(tasks)}개 (멀티프로세싱)", flush=True)

    bags = {}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(args.encoder, refs, args.chunk, args.enc_size)) as ex:
        futs = [ex.submit(_process_group, t) for t in tasks]
        for f in as_completed(futs):
            pid, stain, mag, arr = f.result()
            if arr.shape[0] > 0:
                bags.setdefault((pid, stain, mag), []).append(arr)
            done += 1
            if done % 5 == 0:
                print(f"  {done}/{len(tasks)} 그룹", flush=True)

    for (pid, stain, mag), arrs in bags.items():
        np.save(npy_path(pid, stain, mag), np.concatenate(arrs, 0))

    index = []
    for (pid, stain, mag) in target_keys:
        p = npy_path(pid, stain, mag)
        if p.exists():
            index.append({"patient_id": pid, "stain": stain, "magnification": mag,
                          "encoder": args.encoder, "embed_dim": embed_dim,
                          "n_patches": int(np.load(p, mmap_mode="r").shape[0]),
                          "npy_path": str(p.relative_to(ROOT)), "norm_method": NORM_BY_STAIN[stain]})
    idx_path = out_dir / "index.csv"
    new_df = pd.DataFrame(index)
    if idx_path.exists():
        old = pd.read_csv(idx_path, dtype={"magnification": str})
        old["patient_id"] = old["patient_id"].astype(str)
        new_df["patient_id"] = new_df["patient_id"].astype(str)
        merged = pd.concat([old, new_df]).drop_duplicates(
            subset=["patient_id", "stain", "magnification"], keep="last")
    else:
        merged = new_df
    merged.to_csv(idx_path, index=False, encoding="utf-8")
    print(f"\n임베딩 {len(new_df)} 백(이번) / index 총 {len(merged)} -> {idx_path}", flush=True)


if __name__ == "__main__":
    main()
