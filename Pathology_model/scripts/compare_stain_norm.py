"""
Stain별 염색 정규화 비교 실험 (raw vs Reinhard vs Macenko)

각 stain 그룹에서 슬라이드를 샘플링해 타일을 추출하고, 정규화 방법별로
'슬라이드 간 색 일관성'을 측정한다. 정규화의 목적은 inter-slide 색 편차 감소이므로
지표가 낮을수록 좋다.

지표:
 - color_CV   : 슬라이드별 평균 LAB의 변동계수(채널평균). 낮을수록 일관적.
 - nmi_CV     : 슬라이드별 NMI 중앙값의 변동계수. 낮을수록 일관적.
 - macenko_fail: Macenko 분리 실패 타일 수(비-H&E/배경에서 발생 가능; 정보성 결과)

재현성: SEED 고정. 결과 -> stain_norm_report.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "Pathology_model")))
from mil.stain_norm import ReinhardNormalizer, MacenkoNormalizer, nmi
from mil.wsi_tiles import sample_tissue_tiles

ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "split_manifest.csv"
REPORT = ROOT / "stain_norm_report.json"

SEED = 42
TILE = 256
WORK_LEVEL = 1            # SVS downsample 4 (~10x); IF는 자동으로 최저레벨
STAINS = ["HE", "PAS", "MT", "SILVER", "IF"]


def lab_mean(rgb):
    import cv2
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    return lab.reshape(-1, 3).mean(0)


def cv_across(vals):
    """행=슬라이드, 열=채널. 채널별 CV(std/|mean|)의 평균."""
    a = np.array(vals, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    mu = np.abs(a.mean(0)) + 1e-6
    return float((a.std(0) / mu).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides-per-stain", type=int, default=0,
                    help="0 = 전체 슬라이드 사용")
    ap.add_argument("--tiles-per-slide", type=int, default=12)
    ap.add_argument("--out", default=str(REPORT))
    args = ap.parse_args()
    slides_per_stain = args.slides_per_stain if args.slides_per_stain > 0 else 10 ** 9
    TILES_PER_SLIDE = args.tiles_per_slide
    out_path = Path(args.out)

    df = pd.read_csv(SPLIT)
    rng = np.random.default_rng(SEED)
    report = {"config": {"seed": SEED,
                         "slides_per_stain": "all" if slides_per_stain > 10 ** 8 else slides_per_stain,
                         "tiles_per_slide": TILES_PER_SLIDE, "tile": TILE,
                         "work_level": WORK_LEVEL,
                         "metric_caveat": "color_CV는 Reinhard 목적함수와 거의 동일(순환); "
                                          "NMI_CV가 더 독립적. 최종판정은 다운스트림 MIL 필요."},
              "by_stain": {}}

    print("=" * 64)
    print(" Stain별 염색 정규화 비교 (raw vs Reinhard vs Macenko)")
    print("=" * 64)

    for stain in STAINS:
        sub = df[df["stain"] == stain]
        # gold 우선, 부족하면 보충 — 다양성 위해 환자 단위로 샘플
        pids = sub["patient_id"].unique()
        rng.shuffle(pids)
        chosen, paths = [], []
        for pid in pids:
            row = sub[sub["patient_id"] == pid].iloc[0]
            p = ROOT / row["slide_path"]
            if p.exists():
                chosen.append(pid); paths.append(p)
            if len(paths) >= slides_per_stain:
                break

        # 슬라이드별 타일 샘플
        print(f"\n[{stain}] 슬라이드 {len(paths)}개 타일 추출 중...", flush=True)
        slide_tiles = []
        for i, p in enumerate(paths, 1):
            if i % 10 == 0:
                print(f"   ...{i}/{len(paths)}", flush=True)
            try:
                tiles = sample_tissue_tiles(str(p), TILES_PER_SLIDE, TILE,
                                            WORK_LEVEL, seed=SEED)
            except Exception as e:
                print(f"  [{stain}] 타일 실패 {p.name}: {e}")
                tiles = []
            if tiles:
                slide_tiles.append(tiles)
        if len(slide_tiles) < 2:
            print(f"  [{stain}] 슬라이드 부족({len(slide_tiles)}) -> 스킵")
            continue

        # reference = 첫 슬라이드의 첫 타일
        ref = slide_tiles[0][0]
        reinhard = ReinhardNormalizer().fit(ref)
        macenko = None
        try:
            macenko = MacenkoNormalizer().fit(ref)
        except Exception as e:
            print(f"  [{stain}] Macenko reference fit 실패: {e}")

        res = {m: {"lab_means": [], "nmis": []} for m in ["raw", "reinhard", "macenko"]}
        mac_fail = 0
        for tiles in slide_tiles:
            # 슬라이드별 평균 통계 누적
            acc = {m: {"lab": [], "nmi": []} for m in res}
            for t in tiles:
                acc["raw"]["lab"].append(lab_mean(t)); acc["raw"]["nmi"].append(nmi(t))
                rn = reinhard.transform(t)
                acc["reinhard"]["lab"].append(lab_mean(rn)); acc["reinhard"]["nmi"].append(nmi(rn))
                if macenko is not None:
                    try:
                        mn = macenko.transform(t)
                        acc["macenko"]["lab"].append(lab_mean(mn)); acc["macenko"]["nmi"].append(nmi(mn))
                    except Exception:
                        mac_fail += 1
            for m in res:
                if acc[m]["lab"]:
                    res[m]["lab_means"].append(np.nanmean(acc[m]["lab"], 0))
                    res[m]["nmis"].append(np.nanmedian(acc[m]["nmi"]))

        stain_out = {"n_slides": len(slide_tiles), "macenko_fail_tiles": mac_fail}
        for m in ["raw", "reinhard", "macenko"]:
            if len(res[m]["lab_means"]) >= 2:
                stain_out[m] = {"color_CV": round(cv_across(res[m]["lab_means"]), 4),
                                "nmi_CV": round(cv_across(res[m]["nmis"]), 4)}
        # 최적(color_CV 최소)
        cand = {m: stain_out[m]["color_CV"] for m in ["raw", "reinhard", "macenko"]
                if m in stain_out}
        best = min(cand, key=cand.get) if cand else None
        stain_out["best_by_color_CV"] = best
        report["by_stain"][stain] = stain_out

        print(f"\n[{stain}] 슬라이드 {len(slide_tiles)}개, Macenko 실패타일 {mac_fail}")
        for m in ["raw", "reinhard", "macenko"]:
            if m in stain_out:
                print(f"   {m:9} color_CV={stain_out[m]['color_CV']:.4f}  nmi_CV={stain_out[m]['nmi_CV']:.4f}")
        print(f"   => 최적(color_CV 최소): {best}")

    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n리포트 -> {out_path.name}")


if __name__ == "__main__":
    main()
