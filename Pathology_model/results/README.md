# Pathology_model 실험 결과 구조 (보고서/정리용)

KPMP AKI WSI Multi-Stain MIL 연구의 실험 산출물을 단계별로 정리. 각 폴더 = 하나의 연구 단계.
모든 실험은 **AKI-only(83명) · 환자단위 5-fold · seed 42 · CTransPath 기본**.

| 폴더 | 단계 | 내용 | 핵심 결과 |
|---|---|---|---|
| `01_baseline_cv/` | 베이스라인 MIL | exp1(ResNet50)·exp2(CTransPath)·exp3(DINOv2)·exp5(40x)·exp6(멀티스케일)·exp10main + stain ablation. `eval/`(ROC/PR/calib), `diag/`(ati/kdigo 진단) | foundation>ImageNet, chronic=PAS·immune=HE |
| `02_silver_ablation/` | SILVER 역할 | exp7: silver off / consistency / consistency+attn | SILVER 일관성제약, 이 N에선 개선 미미 |
| `03_clamlite_baseline/` | CLAM-lite 기준선 | exp8 Top-K=8/16/32 | K=16 chronic 0.736(상단값) |
| `04_taskattn/` | Task-Specific Attention | exp11(off/cons) + exp9 robustness + task별 attention overlay PNG·top patches | immune 0.5→0.63↑, task별 attention 독립 |
| `05_diag_collapse/` | Attention Collapse 진단 | fold분해·entropy·top-patch·permutation·temperature·entropy sweep | **collapse 반증** — 범인 아님 |
| `06_data_limitation/` | 데이터 한계 규명 | repeated CV 20×5·pooled OOF·bootstrap CI·fold variance·UMAP | **Case B 확정** — chronic 0.736 광폭CI(양성10명) |
| `07_acquisition/` | 추가수집 타당성 | AKI 전체 진단분포·확보율·용량 시나리오 | AKI 소진(86명중 83), 추가양성 0 |
| `08_descriptor_prediction/` | **병리 Descriptor 예측(현 본실험)** | 5-head(immune·tubulitis·WBC%·fibrosis%·atrophy) descriptor 정량예측 | (진행중) |

## 라벨·코호트
- 진단 라벨: `artifacts/split_manifest.csv` (immune/chronic/stage3/kdigo)
- **병리 정량 라벨: `artifacts/descriptor_labels.csv`** (KPMP TIV Descriptor; 우리 83명 중 62명 보유)
- 통합 감사: `artifacts/audit_summary.json` (`mil/audit_summary.py`로 재생성, results/ 재귀 스캔)

## 재현
스크립트는 `mil/`·`scripts/`. 실행 시 매니페스트는 루트(`c:/dev/chym_aki/`)에 staging 후 도는 구조.
설계 전문은 메모리 `kpmp-mil-silver-clamlite-design.md` 참조.
