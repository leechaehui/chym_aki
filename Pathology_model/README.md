# Pathology_model — KPMP 신장병리 Multi-Stain/Multi-Scale MIL

오늘 실험(데이터 구축 → 인코더/배율/멀티스케일 비교 → 평가/거버넌스) 일체.
백엔드(EMR 앱)와 분리 보관.

## 폴더 구조
```
Pathology_model/
  mil/         MIL 핵심 패키지·분석 코드 (19개)
  scripts/     데이터 준비·다운로드·검증 (15개)
  artifacts/   매니페스트 + 거버넌스 리포트 (12개)
  results/     실험 결과 mil_cv_*.json, oof_*.csv + eval/diag 그림
  README.md
```
> 원본 WSI(`data/raw`, ~50GB)와 임베딩(`data/embeddings`)은 용량 문제로 `c:/dev/chym_aki/data/`에 그대로 둠.

## mil/ (핵심)
- `stain_norm.py` Reinhard/Macenko(512 space) · `wsi_tiles.py` 조직 타일
- `patch_extract.py` 512px 10/20/30/40x 좌표(+tissue_score)
- `encoders.py` registry(resnet50/ctranspath/dinov2; UNI/Virchow gated 제외)
- `embed_patches.py` per-stain 정규화→인코딩→캐시(멀티/단일프로세스, BLAS=1, Top-K)
- `model.py` `StainAwareMIL`(stain fusion) / `MultiScaleMIL`(scale late-fusion)
- `missing_modality.py` 0-fill 금지 fusion · `train.py`/`train_multiscale.py` 환자단위 CV
- `eval_package.py` ROC/PR/calibration/confusion · `attention_viz.py` attention heatmap
- `audit_summary.py` 감사요약 · `agg_*.py`/`diag_*.py` 집계·진단

## 실험 요약 (탐색적, gold ~55환자·양성 12~14 → CI 넓음)
- 인코더: foundation(CTransPath/DINOv2) > ImageNet(ResNet50)
- 배율: chronic·severity는 **40x**, immune은 **HE@10x** 우위 (task별 상이)
- 멀티스케일(10x+40x): immune↑(0.72), chronic↓ — "항상 최적" 아님
- stain: immune=HE, chronic=PAS/MT 신호. **MAIN=H&E/PAS/MT**, Silver=부록, IF 제외
- 핵심수치: `results/mil_cv_*.json`, 비교는 `mil/agg_scale.py`/`agg_compare.py`

## 거버넌스 (사용자 우선순위 1~14, STEP1~5 완료)
- 누수0 환자단위 split, gold라벨, 무결성371/371, 재현성(seed42)·계보, missing-modality
- `artifacts/audit_summary.json`(전 실험 config+결과), `integrity_report.json`, `checksums.csv`

## ⚠️ 재실행 시 경로 주의
- import 경로는 `c:/dev/chym_aki/Pathology_model`로 수정 완료(패키지 import OK).
- 단 스크립트의 `ROOT = c:/dev/chym_aki`가 매니페스트를 루트에서 찾도록 돼 있음 →
  매니페스트가 `Pathology_model/artifacts/`로 이동했으므로, **재실행하려면 각 스크립트의
  artifact 경로를 `Pathology_model/artifacts/`로 갱신 필요**(요청 시 일괄 수정 가능).
  `data/raw`·`data/embeddings` 참조는 그대로 유효(이동 안 함).

## 데이터 범위 메모
- 실제 추출 매니페스트(`manifest_aki_full.csv`)는 enrollment=**AKI only 95명**.
  사용자 의도가 **AKI+CKD**였다면 코호트 재정의/재추출 필요(미반영 상태).
