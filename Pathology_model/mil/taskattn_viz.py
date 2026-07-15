"""
Task-Specific Attention 병리 검증 — taskattn_viz (설명가능성 최우선 단계)

TaskAttentionMIL 을 전체 gold 코호트에 학습(시각화용)한 뒤, 대표 환자에 대해
task별(immune/chronic/stage3/ati_severity) attention 을 '독립적으로' WSI 썸네일에 오버레이.
핵심 질문: immune attention 이 실제 염증 영역에, chronic attention 이 섬유화/경화 영역에 집중하는가?

출력(환자별):
  - PNG: 4 task × 대표 stain(정규화 contribution 최대 single-slide MAIN) 오버레이
  - task별 raw/normalized stain contribution, attention entropy
  - Top-10 attended patches(좌표/stain/score)  → CSV + JSON 동시 저장

좌표 정렬(중요): 임베딩 npy 는 (patient,stain,mag) 단위로 tissue_score 내림차순. 단,
멀티슬라이드 stain 은 슬라이드 연결순서가 비결정적이라 좌표 신뢰 불가 → single-slide stain만 오버레이.

한계(정직): 픽셀 lesion GT 없음 → IoU 등 정량 임상타당성 불가. 병리과 정성 검토용.

사용: python taskattn_viz.py --silver-mode off --n-per-group 5
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import zarr
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path("c:/team/chym_aki/Pathology_model")))

ROOT = Path("c:/team/chym_aki")
OUT = ROOT / "Pathology_model/results/04_taskattn"
OUT.mkdir(parents=True, exist_ok=True)
ENC = "ctranspath"
MAG = "10"
MAIN = ["HE", "PAS", "MT"]
TASKS = ["immune", "chronic", "stage3", "ati_severity"]
GROUPS = {  # (라벨, 판독진단 필터) — 대표 환자 그룹
    "chronic_pos": ["Diabetic Kidney Disease", "Hypertensive Kidney Disease"],
    "immune_pos": ["Acute Interstitial Nephritis"],
    "neg_ctrl_ATI": ["Acute Tubular Injury"],
}
# task별 임상 우선 stain(오버레이 stain 선택 우선순위). immune=염증(HE), chronic=섬유화/사구체(PAS/MT)
PREF = {"immune": ["HE", "PAS", "MT"], "chronic": ["PAS", "MT", "HE"],
        "stage3": ["HE", "PAS", "MT"], "ati_severity": ["HE", "PAS", "MT"]}
# 그룹별 핵심 stain(이게 single-slide인 환자를 우선 선정 → 임상 관련 stain 오버레이 가능)
KEY_STAIN = {"immune_pos": "HE", "chronic_pos": "PAS", "neg_ctrl_ATI": "HE"}


def _entropy(a):
    n = len(a)
    if n <= 1:
        return None
    p = np.asarray(a, dtype=np.float64) + 1e-12
    return float(-(p * np.log(p)).sum() / np.log(n))


def main():
    import torch
    import torch.nn.functional as F
    from mil.model import STAINS, SILVER, TaskAttentionMIL

    ap = argparse.ArgumentParser()
    ap.add_argument("--silver-mode", default="off",
                    choices=["off", "consistency", "consistency_attn"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-per-group", type=int, default=5)
    args = ap.parse_args()

    idx = pd.read_csv(ROOT / "data/embeddings" / ENC / "index.csv", dtype={"magnification": str})
    idx = idx[idx["magnification"] == MAG]
    embed_dim = int(idx["embed_dim"].iloc[0])
    sm = pd.read_csv(ROOT / "split_manifest.csv").drop_duplicates("patient_id")
    sm["patient_id"] = sm["patient_id"].astype(str)
    gold = sm[(sm["label_grade"] == "gold") & (sm["fold"] >= 0)].copy()
    pm = pd.read_csv(ROOT / "patches_manifest.csv", dtype={"magnification": str})
    pm = pm[pm["magnification"] == MAG]

    def load(pid):
        d = {}
        for r in idx[idx["patient_id"].astype(str) == pid].itertuples():
            a = np.load(ROOT / r.npy_path)
            if a.shape[0] > 0:
                d[r.stain] = a.astype(np.float32)
        return d
    bags = {p: load(p) for p in gold["patient_id"] if load(p)}
    gold = gold[gold["patient_id"].isin(bags)]

    # ---- 시각화용 학습(전체 gold, 4 task) ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42); np.random.seed(42)
    model = TaskAttentionMIL(in_dim=embed_dim, silver_mode=args.silver_mode).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    def pw(col):
        yy = gold[col].dropna(); n1 = (yy == 1).sum(); n0 = (yy == 0).sum()
        return torch.tensor(n0 / max(n1, 1), device=device, dtype=torch.float32)
    PW = {"immune": pw("task_immune"), "chronic": pw("task_chronic"), "stage3": pw("task_stage3")}
    COLB = {"immune": "task_immune", "chronic": "task_chronic", "stage3": "task_stage3"}
    gold["ati_n"] = (gold["task_ati_severity"] - 1) / 2.0
    print(f"viz 학습: {len(gold)} gold 환자, silver={args.silver_mode}", flush=True)
    for ep in range(args.epochs):
        model.train()
        for r in gold.sample(frac=1, random_state=42 + ep).itertuples():
            bag = {s: torch.from_numpy(v).to(device) for s, v in bags[r.patient_id].items()}
            out = model(bag); loss = 0.0; nt = 0
            for t in ("immune", "chronic", "stage3"):
                yv = getattr(r, COLB[t])
                if not (yv != yv):
                    loss = loss + F.binary_cross_entropy_with_logits(
                        out[t], torch.tensor(float(yv), device=device), pos_weight=PW[t]); nt += 1
            sv = r.ati_n
            if not (sv != sv):
                loss = loss + F.mse_loss(torch.sigmoid(out["ati_severity"]),
                                         torch.tensor(float(sv), device=device)); nt += 1
            if nt:
                if out.get("silver_align") is not None:
                    loss = loss + 0.3 * (1.0 - F.cosine_similarity(
                        out["silver_align"].unsqueeze(0), out["silver_target"].unsqueeze(0)).squeeze())
                if out.get("main_attn_entropy") is not None:
                    loss = loss + 0.1 * (1.0 - out["main_attn_entropy"])
                opt.zero_grad(); loss.backward(); opt.step()

    # ---- single-slide stain 좌표 캐시 ----
    slide_n = pm.groupby(["patient_id", "stain"])["slide_path"].nunique()

    def coords_for(pid, stain, n_rows):
        """single-slide stain의 tissue_score 내림차순 head(n_rows) 좌표. 멀티슬라이드면 None."""
        if slide_n.get((pid, stain), 99) != 1:
            return None
        sub = pm[(pm["patient_id"].astype(str) == pid) & (pm["stain"] == stain)]
        sub = sub.sort_values("tissue_score", ascending=False).head(n_rows)
        return sub

    # ---- 대표 환자 선정 ----
    selected = []
    for grp, cats in GROUPS.items():
        cand = gold[gold["primary_adjudicated_category"].isin(cats)]["patient_id"].tolist()
        key = KEY_STAIN[grp]
        # 핵심 stain(immune→HE, chronic→PAS)이 single-slide인 환자 우선 → 임상 관련 오버레이 가능
        cand = [p for p in cand if any(s in bags[p] and slide_n.get((p, s), 99) == 1 for s in MAIN)]
        cand.sort(key=lambda p: 0 if (key in bags[p] and slide_n.get((p, key), 99) == 1) else 1)
        for pid in cand[:args.n_per_group]:
            selected.append((pid, grp))
        nkey = sum(1 for p in cand[:args.n_per_group]
                   if key in bags[p] and slide_n.get((p, key), 99) == 1)
        print(f"  {grp}: {len(cand[:args.n_per_group])}명 선정(핵심stain {key} 단일슬라이드 {nkey}명)", flush=True)

    # ---- 환자별 task attention 오버레이 + 기록 ----
    model.eval()
    all_top, summary = [], {}
    for pid, grp in selected:
        bag = {s: torch.from_numpy(v).to(device) for s, v in bags[pid].items()}
        with torch.no_grad():
            out = model(bag)
        sid = out["stain_ids"]                       # 모델 행 순서의 stain 라벨
        dx = gold[gold["patient_id"] == pid]["primary_adjudicated_category"].iloc[0]
        # stain별 행 인덱스 + 좌표
        stain_rows = {s: [i for i, x in enumerate(sid) if x == s] for s in set(sid)}
        stain_coords = {s: coords_for(pid, s, len(stain_rows[s])) for s in stain_rows}

        summary[pid] = {"group": grp, "dx": dx, "tasks": {}}
        # [EXPL1] task 간 '공통 스케일' — 각 task 서브플롯을 개별 자동스케일하지 않음(비교 가능·honest).
        _gvmax = max((out["task_attn"][t].cpu().numpy().max() for t in TASKS if t in out["task_attn"]),
                     default=1e-8) or 1e-8
        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        for ti, task in enumerate(TASKS):
            a = out["task_attn"][task].cpu().numpy()
            raw = out["task_stain_contrib"][task]
            norm = out["task_stain_contrib_norm"][task]
            ent = _entropy(a)
            # Top-10 attended (좌표 가능한 single-slide stain만)
            top = []
            for s, rows in stain_rows.items():
                co = stain_coords[s]
                if co is None or len(co) == 0:
                    continue
                m = min(len(rows), len(co))
                cc = co.iloc[:m]
                for j in range(m):
                    top.append({"patient_id": pid, "group": grp, "dx": dx, "task": task,
                                "stain": s, "tile_x": int(cc.iloc[j]["tile_x"]),
                                "tile_y": int(cc.iloc[j]["tile_y"]),
                                "attention": float(a[rows[j]])})
            top = sorted(top, key=lambda d: -d["attention"])[:10]
            for rk, d in enumerate(top):
                d["rank"] = rk + 1
            all_top.extend(top)
            summary[pid]["tasks"][task] = {
                "raw_contrib": {k: round(v, 4) for k, v in raw.items()},
                "norm_contrib": {k: round(v, 6) for k, v in norm.items()},
                "attn_entropy": None if ent is None else round(ent, 3),
                "top10": [{"stain": d["stain"], "x": d["tile_x"], "y": d["tile_y"],
                           "attention": round(d["attention"], 4)} for d in top]}

            # 오버레이 stain: 임상 우선순위(PREF) 중 single-slide 인 첫 stain, 없으면 normC 최대
            avail = [s for s in MAIN if s in stain_rows and stain_coords[s] is not None]
            ax = axes[ti // 2][ti % 2]
            if not avail:
                ax.set_title(f"{task}: single-slide MAIN 없음"); ax.axis("off"); continue
            best = next((s for s in PREF[task] if s in avail),
                        max(avail, key=lambda s: norm.get(s, 0.0)))
            co = stain_coords[best]; rows = stain_rows[best]
            m = min(len(rows), len(co)); co = co.iloc[:m]; av = a[[rows[j] for j in range(m)]]
            spath = co.iloc[0]["slide_path"]; slevel = int(co.iloc[0]["source_level"])
            sp = ROOT / spath if not Path(spath).is_absolute() else Path(spath)
            with tifffile.TiffFile(sp) as t:
                thumb = np.asarray(zarr.open(t.series[0].levels[-1].aszarr(), mode="r")[:])[..., :3]
                Hs = t.series[0].levels[slevel].shape[0]; Ws = t.series[0].levels[slevel].shape[1]
            th, tw = thumb.shape[:2]
            xs = co["tile_x"].to_numpy() / Ws * tw; ys = co["tile_y"].to_numpy() / Hs * th
            ax.imshow(thumb, alpha=0.45)
            scp = ax.scatter(xs, ys, c=av, cmap="jet", s=16, vmin=0.0, vmax=_gvmax)
            ax.set_xlim(0, tw); ax.set_ylim(th, 0)
            ax.set_title(f"{task} on {best} (normC={norm.get(best,0):.4f}, ent={ent:.2f})")
            plt.colorbar(scp, ax=ax, fraction=0.04)
        fig.suptitle(f"{pid} [{grp}] {dx}", fontsize=13)
        fig.tight_layout()
        fig.savefig(OUT / f"taskattn_{pid}.png", dpi=100, bbox_inches="tight"); plt.close(fig)
        print(f"  -> taskattn_{pid}.png ({grp}, {dx})", flush=True)

    pd.DataFrame(all_top).to_csv(OUT / "taskattn_top_patches.csv", index=False, encoding="utf-8")
    (OUT / "taskattn_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n저장: {OUT}\\taskattn_*.png, taskattn_top_patches.csv, taskattn_summary.json")
    print("주의: lesion GT 없음 → 정성 검토용(IoU 불가). 멀티슬라이드 stain은 좌표 제외.")


if __name__ == "__main__":
    main()
