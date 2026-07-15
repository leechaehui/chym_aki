"""
슬라이드별 구조 분석 추론.

사용법:
  # H&E
  python src/run_inference.py \
    --feat_dir   ~/WSI/features_phikon_512_he \
    --label_csv  ~/WSI/kpmp_data/kpmp_wsi_labels.csv \
    --tiv_xlsx   ~/WSI/kpmp_data/KPMP_TIV_Descriptor_Scores.xlsx \
    --stain      HE \
    --checkpoint ~/WSI/checkpoints_abmil_phikon_512_he/abmil_he_final.pt \
    --out        ~/WSI/inference_cache/he_analysis.json

  # MT
  python src/run_inference.py \
    --feat_dir   ~/WSI/features_phikon_512_mt \
    --label_csv  ~/WSI/kpmp_data/kpmp_wsi_labels.csv \
    --tiv_xlsx   ~/WSI/kpmp_data/KPMP_TIV_Descriptor_Scores.xlsx \
    --stain      TRI \
    --checkpoint ~/WSI/checkpoints_abmil_phikon_512_mt/abmil_mt_final.pt \
    --patch_dir  ~/WSI/patches_mt \
    --out        ~/WSI/inference_cache/mt_analysis.json
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent))
from mil_model import ABMIL, HE_TARGETS, MT_TARGETS


def load_tiv(xlsx_path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(xlsx_path)
    df = xl.parse("Data Table", header=None)
    df.columns = df.iloc[3]
    df = df.iloc[4:].reset_index(drop=True)

    out = pd.DataFrame()
    out["Participant ID"] = df["Participant ID"].astype(str)
    pct_map = {
        "fibrosisRatio": "Cortex: Interstitial Fibrosis: Percent",
        "atrophyRatio":  "Cortex: Tubular Atrophy: Common Type: Percent",
        "tubularInjury": "Cortex: Tubular Injury Other: Percent",
        "inflammation":  "Cortex: Interstitial Monuclear WBC: Percent",
    }
    for key, col in pct_map.items():
        out[key] = pd.to_numeric(df[col], errors="coerce")
    hya = pd.to_numeric(
        df["Cortex and Medulla: Arteriolar Hyalinosis: Arteriolar Hyalinosis: Presence"],
        errors="coerce")
    hya[hya == 999] = np.nan
    out["artHyalinosis"] = hya / 3.0 * 100.0
    # iftaRatio: 모델 타겟이 아니므로 actuals용으로만 파생
    out["iftaRatio"] = out[["fibrosisRatio", "atrophyRatio"]].max(axis=1)
    return out


def build_records(feat_dir: Path, label_csv: Path, tiv_df: pd.DataFrame,
                  stain_filter: str, targets: list) -> list:
    wsi = pd.read_csv(label_csv)
    wsi = wsi[wsi["Workflow Type"] == stain_filter].copy()
    wsi["slide_id"] = wsi["File Name"].apply(lambda x: Path(x).stem)
    wsi["Participant ID"] = wsi["Participant ID"].astype(str)
    merged = wsi.merge(tiv_df, on="Participant ID", how="left")

    records = []
    for _, row in merged.iterrows():
        feat_path = feat_dir / f"{row['slide_id']}.pt"
        if not feat_path.exists():
            continue
        actuals = {}
        for t in targets:
            v = row.get(t)
            actuals[t] = None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
        records.append({
            "slide_id":  row["slide_id"],
            "feat_path": feat_path,
            "actuals":   actuals,
        })
    return records


# ── MT 색상 분석 (Masson's Trichrome) ────────────────────────────────────────
def _blue_ratio(arr: np.ndarray) -> float:
    r, g, b = arr[:,:,0].astype(int), arr[:,:,1].astype(int), arr[:,:,2].astype(int)
    mask = (b - r > 30) & (b - g > 20) & (arr[:,:,2] > 140) & (arr[:,:,0] < 180)
    return float(mask.mean())


def compute_mt_color_metrics(patch_dir: Path) -> dict:
    patches = list(patch_dir.rglob("*.png")) + list(patch_dir.rglob("*.jpg"))
    if not patches:
        return {}
    blue_ratios = []
    for p in patches:
        try:
            blue_ratios.append(_blue_ratio(np.array(Image.open(p).convert("RGB"))))
        except Exception:
            continue
    if not blue_ratios:
        return {}
    br = np.array(blue_ratios)
    return {
        "collagenRatio": round(float(br.mean()) * 100, 1),
        "interstitialFibrosisColor": round(float((br > 0.08).mean()) * 100, 1),
        "tubularAtrophyColor": round(float((br > 0.15).mean()) * 100, 1),
    }


# ── 추론 ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def infer_slide(model, feat_path: Path, device):
    feat = torch.load(feat_path, map_location="cpu", weights_only=False).float()
    preds, attn = model(feat.to(device))
    return preds.cpu().numpy(), attn.cpu().numpy(), feat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat_dir",    required=True)
    parser.add_argument("--label_csv",   required=True)
    parser.add_argument("--tiv_xlsx",    required=True)
    parser.add_argument("--stain",       required=True, help="HE | TRI")
    parser.add_argument("--stain_filter",default=None)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--in_dim",      type=int, default=1024)
    parser.add_argument("--patch_dir",   default=None, help="MT 색상 분석용 패치 디렉토리")
    parser.add_argument("--out",         required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = MT_TARGETS if args.stain == "TRI" else HE_TARGETS
    print(f"device={device}  stain={args.stain}  targets={targets}")

    model = ABMIL(in_dim=args.in_dim, n_targets=len(targets)).to(device)
    ck    = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ck.get("model_state_dict", ck), strict=True)
    model.eval()

    stain_filter = args.stain_filter or (
        "TRI stain" if args.stain == "TRI" else "H&E stain")
    tiv_df  = load_tiv(Path(args.tiv_xlsx))
    records = build_records(Path(args.feat_dir), Path(args.label_csv),
                            tiv_df, stain_filter, targets)
    print(f"슬라이드: {len(records)}개  타겟: {targets}")

    slides = []
    for i, r in enumerate(records, 1):
        pred_np, attn_np, feat = infer_slide(model, r["feat_path"], device)
        preds = {t: round(float(pred_np[j]) * 100, 1) for j, t in enumerate(targets)}
        # iftaRatio: fibrosis/atrophy에서 파생
        preds["iftaRatio"] = round(max(
            preds.get("fibrosisRatio", 0) or 0,
            preds.get("atrophyRatio",  0) or 0), 1)

        # attention top-20
        coords_path = r["feat_path"].parent / (r["feat_path"].stem + ".coords.npy")
        attn_top = []
        if coords_path.exists():
            coords  = np.load(coords_path)
            top_idx = np.argsort(attn_np)[-20:][::-1]
            attn_top = [[int(coords[k,0]), int(coords[k,1]), round(float(attn_np[k]),5)]
                        for k in top_idx if k < len(coords)]

        # MT: 색상 분석
        color_metrics = {}
        if args.stain == "TRI" and args.patch_dir:
            pd_path = Path(args.patch_dir) / r["slide_id"]
            if pd_path.exists():
                color_metrics = compute_mt_color_metrics(pd_path)

        slides.append({
            "slide_id":    r["slide_id"],
            "preds":       preds,
            "actuals":     r["actuals"],
            "attn_top":    attn_top,
            "color_metrics": color_metrics,
        })
        fib = preds.get("fibrosisRatio", 0)
        print(f"  [{i}/{len(records)}] {r['slide_id']}"
              f"  IFTA_pred={preds['iftaRatio']:.1f}%  Fibrosis={fib:.1f}%")

    output = {"stain": args.stain, "n_slides": len(slides), "slides": slides}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
