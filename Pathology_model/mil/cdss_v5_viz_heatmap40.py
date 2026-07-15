"""
CDSS v5 — 40x 조밀 attention 히트맵 (레퍼런스 figure 재현 · 배치 생성)

40x 재학습 가중치(tag=vfind40, train_cdss_v5.py --mags 40)로 환자별 attention 을 그린다.
환자마다 '자신이 held-out 이던 fold' 모델을 써서 OOF(정직한) attention 을 산출한다.

각 figure: 4열 × stain행 = Original │ Attention(blocky) │ Attention(smooth) │ Top-16 patches.

격자: tile 간격 = read_size(정규격자), pitch = read_size*2**source_level(level0 px) → 1타일=1칸.
좌표 정렬: 임베딩은 (patient,stain,40) tissue_score 내림차순 head(100) → manifest 동일 정렬 매칭.
상위100 패치는 단일 슬라이드에서 나옴(확인됨) → 단일 썸네일에 오버레이.

사용:
  python mil/cdss_v5_viz_heatmap40.py                       # 기본 배치(8명)
  python mil/cdss_v5_viz_heatmap40.py --pids 30-10123,30-10034
  python mil/cdss_v5_viz_heatmap40.py --tag vfind40 --mc-fold 0   # 폴드 고정
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import tifffile
import zarr
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path("c:/team/chym_aki/Pathology_model")))
from mil.cdss_paths import data_root, p as _p
from mil.train import ROOT, load_bags
from mil.cdss_v5_model import CDSSv5Model
from mil.embed_patches import read_region

STAINS = ["HE", "PAS", "MT"]
MAG = "40"
THUMB_DIR = Path("D:/chym_aki_data/metadata/thumbnails")
OUT = Path("C:/team/chym_aki/data/eval/attention")
OUT.mkdir(parents=True, exist_ok=True)
ALPHA = 0.55
DEFAULT_PIDS = [  # AIN/ATI/DKD 다양하게(전부 40x 3-stain 완비)
    "30-10018", "30-10868", "30-10929",   # AIN
    "30-10034", "30-10044", "30-10125", "30-10631",  # ATI
    "30-10123",                            # DKD
]


def _resolve(slide_path):
    return ROOT / slide_path if not Path(slide_path).is_absolute() else Path(slide_path)


def get_patch(slide_path, source_level, tile_x, tile_y, read_size, out_size=256):
    with tifffile.TiffFile(_resolve(slide_path)) as t:
        za = zarr.open(t.series[0].aszarr(level=int(source_level)), mode="r")
        return read_region(za, int(tile_x), int(tile_y), int(read_size), int(out_size))


def slide_level0_hw(slide_path):
    with tifffile.TiffFile(_resolve(slide_path)) as t:
        s = t.series[0].levels[0].shape
    return int(s[0]), int(s[1])


def load_model(tag, fold, embed_dim, device):
    import torch
    m = CDSSv5Model(in_dim=embed_dim).to(device)
    wp = data_root() / "results" / "cdss_v5" / f"cdss_v5_fold{fold}_{tag}.pt"
    m.load_state_dict(torch.load(wp, map_location=device, weights_only=True))
    m.eval()
    return m


def render_patient(pid, model, idx40, manifest, device):
    import torch
    bags = load_bags([pid], idx40, keep=set(STAINS))
    if pid not in bags:
        print(f"[{pid}] no 40x bag — skip"); return None

    mats, lengths, valid = [], [], []
    for s in STAINS:
        v = bags[pid].get(s)
        if v is not None and v.shape[0] > 0:
            mats.append(torch.from_numpy(v).to(device)); lengths.append(v.shape[0]); valid.append(s)
    if not mats:
        print(f"[{pid}] no base stain — skip"); return None

    a_split = []
    with torch.no_grad():
        for s in valid:
            x = torch.from_numpy(bags[pid][s]).to(device)
            if s == "HE":
                proj, attn_a, attn_c, attn_i = model.proj_he, model.attn_ais_he, model.attn_cds_he, model.attn_ins_he
            elif s == "PAS":
                proj, attn_a, attn_c, attn_i = model.proj_pas, model.attn_ais_pas, model.attn_cds_pas, model.attn_ins_pas
            elif s == "MT":
                proj, attn_a, attn_c, attn_i = model.proj_mt, model.attn_ais_mt, model.attn_cds_mt, model.attn_ins_mt
            else:
                continue

            h = proj(x)
            if ATTN_HEAD == "ais":
                a = torch.softmax(attn_a(h), dim=0)
            elif ATTN_HEAD == "cds":
                a = torch.softmax(attn_c(h), dim=0)
            elif ATTN_HEAD == "ins":
                a = torch.softmax(attn_i(h), dim=0)
            else:
                a = (torch.softmax(attn_a(h), dim=0) + torch.softmax(attn_c(h), dim=0) + torch.softmax(attn_i(h), dim=0)) / 3.0
            
            a_split.append(a)
    # [EXPL1 수정] 염색별 vmin/vmax(PAS/MT를 인위적으로 빨갛게)를 제거하고 '전 염색 공통' 스케일 복원.
    # 기여/집중이 낮은 염색은 공통 스케일에서 어둡게 표시(honest). 자동 확대로 핫스팟 조작 금지.
    _allv = np.concatenate([a.cpu().numpy().reshape(-1) for a in a_split]) if a_split else np.zeros(1)
    G_VMIN, G_VMAX = float(np.percentile(_allv, 5)), float(np.percentile(_allv, 99))

    fig = plt.figure(figsize=(26, 6.8 * len(valid)))
    gs = gridspec.GridSpec(len(valid), 4, width_ratios=[1, 1, 1, 1.25])

    for ri, (stain, attn) in enumerate(zip(valid, a_split)):
        attn_np = attn.cpu().numpy().reshape(-1)
        
        # 전 염색 공통 스케일 사용(염색별 확대 금지) — 기여 낮은 염색은 어둡게.
        vmin, vmax = G_VMIN, G_VMAX
        def norm(x):
            return np.clip((x - vmin) / (vmax - vmin + 1e-8), 0, 1)

        n = len(attn_np)
        sub = manifest[(manifest.patient_id == pid) & (manifest.stain == stain)
                       & (manifest.magnification == MAG)]
        sub = sub.sort_values("tissue_score", ascending=False).head(n).copy()
        sub["attn"] = attn_np[:len(sub)]
        slide_id = sub.slide_id.iloc[0]
        sub = sub[sub.slide_id == slide_id]
        slide_path = sub.slide_path.iloc[0]
        rs = int(sub.read_size.iloc[0]); slv = int(sub.source_level.iloc[0]); pitch = rs * (2 ** slv)
        H0, W0 = slide_level0_hw(slide_path)

        tp = THUMB_DIR / f"{pid}_{stain}_{slide_id}_thumb.jpg"
        if not tp.exists():
            print(f"[{pid}/{stain}] thumb missing {tp}"); continue
        g = plt.imread(tp).astype(np.float32)
        g = g / 255.0 if g.max() > 1 else g
        th, tw = g.shape[:2]
        tissue = g[..., :3].mean(2) < 0.92

        gw = int(np.ceil(W0 / pitch)); gh = int(np.ceil(H0 / pitch))
        grid = np.full((gh, gw), np.nan)
        for x, y, av in zip(sub.tile_x, sub.tile_y, sub.attn):
            gx = min(int(x // pitch), gw - 1); gy = min(int(y // pitch), gh - 1)
            grid[gy, gx] = av if np.isnan(grid[gy, gx]) else max(grid[gy, gx], av)
        filled = ~np.isnan(grid)
        grid0 = np.where(filled, grid, 0.0)

        # blocky
        block = cv2.resize(grid0, (tw, th), interpolation=cv2.INTER_NEAREST)
        bmask = cv2.resize(filled.astype(np.float32), (tw, th), interpolation=cv2.INTER_NEAREST) > 0.5
        bcolor = plt.cm.jet(norm(block))[..., :3]
        a_b = ALPHA * np.expand_dims(bmask & tissue, 2)
        blended_b = np.clip(g[..., :3] * (1 - a_b) + bcolor * a_b, 0, 1)

        # smooth (weighted gaussian)
        wgt = filled.astype(np.float32)
        wb = cv2.GaussianBlur(wgt, (0, 0), 3.0)
        vb = cv2.GaussianBlur(grid0, (0, 0), 3.0)
        sm = np.where(wb > 1e-4, vb / (wb + 1e-8), 0.0)
        sm_r = cv2.resize(sm, (tw, th), interpolation=cv2.INTER_LINEAR)
        cov = cv2.resize(np.clip(wb * 3.0, 0, 1), (tw, th), interpolation=cv2.INTER_LINEAR)
        scolor = plt.cm.jet(norm(sm_r))[..., :3]
        a_s = ALPHA * np.expand_dims(cov * tissue, 2)
        blended_s = np.clip(g[..., :3] * (1 - a_s) + scolor * a_s, 0, 1)

        # tissue bbox crop
        rows = np.where(tissue.any(1))[0]; cols = np.where(tissue.any(0))[0]
        if len(rows) and len(cols):
            my = max(1, int(0.04 * th)); mx = max(1, int(0.04 * tw))
            r0, r1 = max(0, rows[0] - my), min(th, rows[-1] + my)
            c0, c1 = max(0, cols[0] - mx), min(tw, cols[-1] + mx)
        else:
            r0, r1, c0, c1 = 0, th, 0, tw
        crop = lambda im: im[r0:r1, c0:c1]

        ax0 = plt.subplot(gs[ri, 0]); ax0.imshow(crop(g[..., :3])); ax0.axis('off')
        ax0.set_title(f"{stain} - Original Slide", fontsize=16)
        ax1 = plt.subplot(gs[ri, 1]); ax1.imshow(crop(blended_b)); ax1.axis('off')
        ax1.set_title(f"{stain} - Attention (blocky)", fontsize=16)
        ax2 = plt.subplot(gs[ri, 2]); ax2.imshow(crop(blended_s)); ax2.axis('off')
        ax2.set_title(f"{stain} - Attention (smooth)", fontsize=16)

        ax3 = plt.subplot(gs[ri, 3]); ax3.axis('off')
        ax3.set_title(f"{stain} - Top 16 Patches", fontsize=16, pad=25)
        top = sub.sort_values("attn", ascending=False).head(16)
        gi = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=gs[ri, 3], wspace=0.1, hspace=0.28)
        for i, (_, row) in enumerate(top.iterrows()):
            try:
                pim = get_patch(row.slide_path, row.source_level, row.tile_x, row.tile_y, row.read_size, 256)
                axp = plt.subplot(gi[i]); axp.imshow(pim); axp.axis('off')
                axp.set_title(f"{row.attn:.4f}", fontsize=9)
                axp.text(0.04, 0.96, f"#{i+1}", transform=axp.transAxes, color='black',
                         va='top', ha='left', fontweight='bold',
                         bbox=dict(facecolor='white', alpha=0.8, pad=1, edgecolor='none'))
            except Exception as e:
                print(f"[{pid}/{stain}] patch {i} fail: {e}")

    htag = "" if ATTN_HEAD == "avg" else f"_{ATTN_HEAD}"
    out = OUT / f"cdss_v5_heatmap40_{MODEL_TAG}{htag}_{pid}.png"
    plt.suptitle(f"{pid}  ·  40x (tag={MODEL_TAG})  ·  attn-head={ATTN_HEAD}  ·  OOF attention", fontsize=13, y=1.002)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[{pid}] saved -> {out}")
    return out


MODEL_TAG = "vfind40"
ATTN_HEAD = "avg"


def main():
    import torch
    global MODEL_TAG, ATTN_HEAD
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="vfind40")
    ap.add_argument("--pids", default="")
    ap.add_argument("--attn-head", default="avg", choices=["avg", "ais", "cds", "ins"],
                    help="ais=ATI 전용 attention 단독")
    ap.add_argument("--mc-fold", type=int, default=-1, help=">=0이면 모든 환자에 해당 fold 모델 고정")
    args = ap.parse_args()
    MODEL_TAG = args.tag
    ATTN_HEAD = args.attn_head
    device = "cuda" if torch.cuda.is_available() else "cpu"

    idx = pd.read_csv(_p("embeddings") / "ctranspath" / "index.csv", dtype={"magnification": str})
    embed_dim = int(idx["embed_dim"].iloc[0])
    idx40 = idx[idx["magnification"] == MAG]

    manifest = pd.read_csv("C:/team/chym_aki/patches_manifest.csv", dtype=str)
    for c in ["tile_x", "tile_y", "read_size", "source_level"]:
        manifest[c] = manifest[c].astype(int)
    manifest["tissue_score"] = manifest["tissue_score"].astype(float)

    sm = pd.read_csv("C:/team/chym_aki/split_manifest.csv").drop_duplicates("patient_id")
    sm["patient_id"] = sm["patient_id"].astype(str)
    pid2fold = sm.set_index("patient_id")["fold"].to_dict()

    pids = [x.strip() for x in args.pids.split(",") if x.strip()] or DEFAULT_PIDS

    model_cache = {}
    done = []
    for pid in pids:
        fold = args.mc_fold if args.mc_fold >= 0 else int(pid2fold.get(pid, 0))
        if fold not in (0, 1, 2, 3, 4):
            fold = 0  # held-out 없으면(fold -1) fold0 폴백
        if fold not in model_cache:
            model_cache[fold] = load_model(args.tag, fold, embed_dim, device)
        try:
            out = render_patient(pid, model_cache[fold], idx40, manifest, device)
            if out:
                done.append((pid, fold, out))
        except Exception as e:
            print(f"[{pid}] FAILED: {e}")

    print(f"\n=== batch done: {len(done)}/{len(pids)} ===")
    for pid, fold, out in done:
        print(f"  {pid} (fold{fold}) -> {out}")


if __name__ == "__main__":
    main()
