"""
STEP 5 — Explainability / Clinical validity (정성)

40x MAIN(HE/PAS/MT)로 StainAwareMIL을 전체 코호트에 학습(시각화용)한 뒤,
예시 환자의 HE 패치 attention을 슬라이드 썸네일에 오버레이 + top-attended 패치 저장.

한계(정직): 픽셀단위 병변 주석(lesion GT)이 없어 attention의 '정량적' 임상타당성
(IoU 등)은 불가. 병리과 검토용 정성 산출물 + stain contribution을 제공.

좌표 정렬: 임베딩은 (patient,stain,40)별 tissue_score 내림차순 Top-K 순서이므로 동일 재현.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import zarr
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Malgun Gothic"   # 한글 글리프(폰트문제 해결)
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

_PKG = Path(__file__).resolve().parents[1]          # Pathology_model
ROOT = _PKG.parent                                   # 리포 루트(머신 비의존)
sys.path.insert(0, str(_PKG))
from mil.embed_patches import read_region

OUT = ROOT / "data/eval/attention"
OUT.mkdir(parents=True, exist_ok=True)
ENC = "ctranspath"
MAIN = ["HE", "PAS", "MT"]
TOPK = 300


def main():
    import torch
    from mil.model import StainAwareMIL

    idx = pd.read_csv(ROOT / "data/embeddings" / ENC / "index.csv", dtype={"magnification": str})
    idx = idx[(idx["magnification"] == "40") & (idx["stain"].isin(MAIN))]
    embed_dim = int(idx["embed_dim"].iloc[0])
    sm = pd.read_csv(ROOT / "split_manifest.csv").drop_duplicates("patient_id")
    sm["patient_id"] = sm["patient_id"].astype(str)
    gold = sm[(sm["label_grade"] == "gold") & (sm["fold"] >= 0)]

    # bags (40x MAIN)
    def load(pid):
        d = {}
        for r in idx[idx["patient_id"].astype(str) == pid].itertuples():
            a = np.load(ROOT / r.npy_path)
            if a.shape[0] > 0:
                d[r.stain] = a.astype(np.float32)
        return d
    bags = {p: load(p) for p in gold["patient_id"] if load(p)}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    model = StainAwareMIL(in_dim=embed_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    g = gold[gold["patient_id"].isin(bags)]
    import torch.nn.functional as F
    print(f"viz용 학습: {len(g)}환자 (전체, 시각화 목적)", flush=True)
    for ep in range(40):
        model.train()
        for r in g.sample(frac=1, random_state=42 + ep).itertuples():
            bag = {s: torch.from_numpy(v).to(device) for s, v in bags[r.patient_id].items()}
            out = model(bag); loss = 0.0; nt = 0
            if not np.isnan(r.task_immune):
                loss = loss + F.binary_cross_entropy_with_logits(out["immune"],
                        torch.tensor(float(r.task_immune), device=device)); nt += 1
            if not np.isnan(r.task_chronic):
                loss = loss + F.binary_cross_entropy_with_logits(out["chronic"],
                        torch.tensor(float(r.task_chronic), device=device)); nt += 1
            if nt:
                opt.zero_grad(); loss.backward(); opt.step()

    # 예시 환자: AIN, ATI, DKD 각 1명(HE 단일 슬라이드 우선)
    pm = pd.read_csv(ROOT / "patches_manifest.csv", dtype={"magnification": str})
    pm40 = pm[pm["magnification"] == "40"]
    examples = []
    for cat in ["Acute Interstitial Nephritis", "Acute Tubular Injury", "Diabetic Kidney Disease"]:
        cand = gold[gold["primary_adjudicated_category"] == cat]["patient_id"].tolist()
        for pid in cand:
            sub_he = pm40[(pm40["patient_id"].astype(str) == pid) & (pm40["stain"] == "HE")]
            if pid in bags and "HE" in bags[pid] and sub_he["slide_path"].nunique() == 1:
                examples.append((pid, cat)); break

    model.eval()
    for pid, cat in examples:
        bag = {s: torch.from_numpy(v).to(device) for s, v in bags[pid].items()}
        with torch.no_grad():
            out = model(bag)
        contrib = out["stain_contrib"]

        # [EXPL1 수정] 이미지별 min-max 자동스케일(착시) 금지 → 염색 간 '공통 절대 스케일'.
        # per-stain softmax(합=1)를 fusion 기여도로 가중(attn*contrib)해, 기여 0인 염색은
        # 실제로 어둡게 표시. vmax 는 전 염색 공통 → 색이 염색 간 절대 비교 가능(honest).
        disp = {s: out["patch_attn"][s].cpu().numpy() * float(contrib.get(s, 0.0))
                for s in ["HE", "PAS", "MT"] if s in out["patch_attn"]}
        vmax = max((d.max() for d in disp.values() if len(d)), default=1e-8) or 1e-8

        for s in ["HE", "PAS", "MT"]:
            if s not in out["patch_attn"]: continue
            attn = disp[s]                                   # 기여 가중값(공통 스케일)
            sub = pm40[(pm40["patient_id"].astype(str) == pid) & (pm40["stain"] == s)]
            if len(sub) == 0: continue
            coords = sub.sort_values("tissue_score", ascending=False).head(TOPK)
            n = min(len(coords), len(attn))
            coords = coords.iloc[:n]; attn = attn[:n]
            spath = coords.iloc[0]["slide_path"]; slevel = int(coords.iloc[0]["source_level"])
            with tifffile.TiffFile(ROOT / spath if not Path(spath).is_absolute() else spath) as t:
                thumb = np.asarray(zarr.open(t.series[0].levels[-1].aszarr(), mode="r")[:])[..., :3]
                Hs = t.series[0].levels[slevel].shape[0]; Ws = t.series[0].levels[slevel].shape[1]
            th, tw = thumb.shape[:2]
            fig, ax = plt.subplots(1, 2, figsize=(14, 6))
            ax[0].imshow(thumb); ax[0].set_title(f"{pid} ({cat[:20]}) {s} thumbnail")
            xs = coords["tile_x"].to_numpy() / Ws * tw
            ys = coords["tile_y"].to_numpy() / Hs * th
            # 공통 vmin=0, vmax(전 염색) → 기여 낮은 염색은 어둡게(자동스케일로 빨갛게 만들지 않음)
            sc = ax[1].scatter(xs, ys, c=attn, cmap="jet", s=14, vmin=0.0, vmax=vmax)
            ax[1].imshow(thumb, alpha=0.45)
            ax[1].set_title(f"{s}  attention×contrib={contrib.get(s, 0):.3f}  (공통스케일; 색=실제 기여)")
            plt.colorbar(sc, ax=ax[1], fraction=0.04)
            ax[1].set_xlim(0, tw); ax[1].set_ylim(th, 0)
            fig.tight_layout(); fig.savefig(OUT / f"attn_{pid}_{s}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
            print(f"  {pid} ({cat}) {s}: contrib={contrib.get(s,0):.3f} disp_max={attn.max():.4f}/{vmax:.4f}")
    print(f"\n주의: lesion GT 없어 정량 임상타당성(IoU) 불가 — 정성 검토용. -> {OUT}")


if __name__ == "__main__":
    main()
