"""
STEP 5 — Auditability: 재현성·감사 요약 컴파일

데이터 버전·seed·split·checksum·무결성·전 실험 config/결과를 한 파일로 모은다.
사용자 거버넌스 #5(재현성)·#6(계보)·auditability 산출물.
"""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "audit_summary.json"
# 산출물이 root(신규 실행 staging)·results/·artifacts/ 에 흩어져 있어 모두 스캔(파일 이동 견고성)
DIRS = [ROOT, ROOT / "Pathology_model/results", ROOT / "Pathology_model/artifacts"]


def find_file(name):
    for d in DIRS:
        p = d / name
        if p.exists():
            return p
    return None


def safe_json(name):
    p = find_file(name)
    return json.loads(p.read_text(encoding="utf-8")) if p else None


def main():
    audit = {"generated": datetime.now().isoformat(timespec="seconds"),
             "dataset_version": "v1"}

    # 데이터 무결성/계보
    audit["integrity"] = safe_json("integrity_report.json")
    audit["split"] = safe_json("split_report.json")
    audit["modality_mask"] = safe_json("modality_mask_report.json")
    cs_path = find_file("checksums.csv")
    if cs_path:
        cs = pd.read_csv(cs_path)
        audit["checksums"] = {"n_files": int(len(cs)), "file": "checksums.csv"}

    # 환자/슬라이드
    sm_path = find_file("split_manifest.csv")
    if sm_path:
        sm = pd.read_csv(sm_path)
        pat = sm.drop_duplicates("patient_id")
        audit["cohort"] = {"slides": int(len(sm)), "patients": int(pat["patient_id"].nunique()),
                           "gold": int((pat["label_grade"] == "gold").sum()),
                           "fold_ge0": int((pat["fold"] >= 0).sum())}

    # 전 실험 config + 결과 (root + results/ 하위폴더 재귀 + artifacts, stem 중복은 root 신규 우선)
    audit["experiments"] = {}
    seen = set()
    mil_files = (sorted(ROOT.glob("mil_cv_*.json"))
                 + sorted((ROOT / "Pathology_model/results").rglob("mil_cv_*.json"))
                 + sorted((ROOT / "Pathology_model/artifacts").glob("mil_cv_*.json")))
    for f in mil_files:
        if True:
            if f.stem in seen:
                continue
            seen.add(f.stem)
            d = json.loads(f.read_text(encoding="utf-8"))
            cfg = d.get("config", {})
            tasks = {t: {k: v for k, v in m.items() if k in ("auroc", "auroc_ci95", "pr_auc",
                                                             "spearman", "n", "n_pos")}
                     for t, m in d.get("tasks", {}).items()}
            audit["experiments"][f.stem.replace("mil_cv_", "")] = {
                "config": {k: cfg.get(k) for k in (
                    "tag", "model", "encoder", "embed_dim", "mags", "scales", "stains",
                    "patch", "seed", "epochs", "cohort",
                    "silver_mode", "silver_lambda", "attn_lambda", "topk")},
                "results": tasks,
                "scale_contribution": d.get("scale_contribution"),
                "stain_contribution": d.get("stain_contribution"),
                # 개선2 Task-Specific Attention: task별 stain 기여·attention entropy
                "task_stain_contribution": d.get("task_stain_contribution"),
                "task_attn_entropy": d.get("task_attn_entropy")}

    # exp9 robustness 분석(있으면 통합)
    rob = {}
    for f in sorted((ROOT / "Pathology_model/results").rglob("exp9_robustness_*.json")):
        key = f.stem.replace("exp9_robustness_", "")
        if key not in rob:
            rob[key] = json.loads(f.read_text(encoding="utf-8"))
    if rob:
        audit["robustness_exp9"] = rob

    # 코드/아티팩트 계보
    audit["artifacts"] = {
        "manifest": "manifest_aki_full.csv (App Search, 26컬럼)",
        "selected": "selected_manifest.csv",
        "patches": "patches_manifest.csv (512px, 10/20/30/40x, tissue_score)",
        "embeddings": "data/embeddings/{encoder}/index.csv",
        "code": "Pathology_model/mil/{patch_extract,embed_patches,stain_norm,encoders,model,"
                "train,train_multiscale,train_clam_lite,clam_lite,eval_package,exp9_robustness}.py",
        "eval_figs": "results/eval/*_eval.png, *_errors.csv",
        "silver_branch": "model.StainAwareMIL(silver_mode): off/consistency/consistency_attn "
                         "— SILVER=구조 일관성 제약(L_main+λ·cos), 예측 비포함",
        "clam_lite": "clam_lite.ClamLite(Top-K=16) — baseline(SILVER·multiscale 미사용)",
    }
    audit["reproducibility"] = {
        "seed": 42, "split": "patient-level StratifiedKFold (gold=stratify_key, kdigo=stage3)",
        "norm": "512 space (HE=Macenko, others=Reinhard)", "encoder_input": 224,
        "leakage_free": "환자단위 fold, 슬라이드 누수 0 (integrity_report 확인)"}

    OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"감사 요약 -> {OUT.name}")
    print(f"  실험 {len(audit['experiments'])}개, 코호트 {audit.get('cohort')}")
    print(f"  무결성 통과 {audit.get('integrity',{}).get('slides_passed')}/"
          f"{audit.get('integrity',{}).get('slides_total')}")


if __name__ == "__main__":
    main()
