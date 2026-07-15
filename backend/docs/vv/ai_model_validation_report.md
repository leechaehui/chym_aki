# AKI 2-Stage 모델 Validation 리포트

_생성: 2026-06-19T05:54:46+00:00_

## 데이터셋

- **source**: AKI\preprocessing\final_dataset\test_final.csv
- **n_rows**: 6433
- **features**: 35
- **aki_prevalence**: 24.8%
- **stage_distribution**: {0: 4835, 1: 1237, 2: 168, 3: 193}
- **stage1_threshold**: 0.7
- **stage2_threshold**: 0.56

## Stage1 (LR) — Non-AKI vs AKI

- N=6433, positives=1598 (prevalence 24.8%)
- **AUROC 0.8907** · **AUPRC 0.7322** · Brier 0.1339 · ECE 0.1790

| 확률 bin | n | 평균예측 | 관측빈도 |
|---|---:|---:|---:|
| [0.0,0.1) | 2433 | 0.021 | 0.005 |
| [0.1,0.2) | 514 | 0.145 | 0.006 |
| [0.2,0.3) | 355 | 0.249 | 0.014 |
| [0.3,0.4) | 270 | 0.348 | 0.022 |
| [0.4,0.5) | 242 | 0.447 | 0.050 |
| [0.5,0.6) | 231 | 0.551 | 0.078 |
| [0.6,0.7) | 261 | 0.652 | 0.222 |
| [0.7,0.8) | 1303 | 0.716 | 0.844 |
| [0.8,0.9) | 294 | 0.849 | 0.207 |
| [0.9,1.0) | 530 | 0.978 | 0.609 |

## Stage2 (LGBM) — Stage1 vs Stage2+3 (AKI 내)

- N=1598, positives=361 (prevalence 22.6%)
- **AUROC 0.5777** · **AUPRC 0.2632** · Brier 0.2361 · ECE 0.2252

| 확률 bin | n | 평균예측 | 관측빈도 |
|---|---:|---:|---:|
| [0.0,0.1) | 252 | 0.045 | 0.119 |
| [0.1,0.2) | 126 | 0.142 | 0.143 |
| [0.2,0.3) | 75 | 0.249 | 0.213 |
| [0.3,0.4) | 50 | 0.344 | 0.160 |
| [0.4,0.5) | 37 | 0.443 | 0.324 |
| [0.5,0.6) | 1043 | 0.567 | 0.263 |
| [0.6,0.7) | 8 | 0.641 | 0.250 |
| [0.7,0.8) | 4 | 0.755 | 0.000 |
| [0.8,0.9) | 2 | 0.822 | 0.000 |
| [0.9,1.0) | 1 | 0.900 | 1.000 |

## Subgroup — 연령 분위(표준화 age tercile, Stage1 AKI)

```json
[
  {
    "subgroup": "Q1(낮음)",
    "n": 2088,
    "positives": 341,
    "prevalence": 0.1633,
    "auroc": 0.8885,
    "auprc": 0.6445
  },
  {
    "subgroup": "Q2(중간)",
    "n": 2176,
    "positives": 580,
    "prevalence": 0.2665,
    "auroc": 0.9076,
    "auprc": 0.7728
  },
  {
    "subgroup": "Q3(높음)",
    "n": 2169,
    "positives": 677,
    "prevalence": 0.3121,
    "auroc": 0.8784,
    "auprc": 0.7668
  }
]
```

## Subgroup — 성별 (Stage1 AKI)

```json
[
  {
    "subgroup": "F",
    "n": 2837,
    "positives": 645,
    "prevalence": 0.2274,
    "auroc": 0.8813,
    "auprc": 0.699
  },
  {
    "subgroup": "M",
    "n": 3596,
    "positives": 953,
    "prevalence": 0.265,
    "auroc": 0.8981,
    "auprc": 0.7579
  }
]
```

## Missing-data sensitivity (Stage1 AKI)

```json
{
  "n_missing_flag_cols": 36,
  "overall_auroc": 0.8907,
  "low_missing_auroc": 0.9497,
  "high_missing_auroc": 0.8428,
  "median_missing_count": 3.0,
  "delta_low_minus_high": 0.1069
}
```

## Feature ablation (Stage1 AKI AUROC drop)

```json
[
  {
    "feature_group": "약물(vasopressor/norepi)",
    "base_auroc": 0.8907,
    "ablated_auroc": 0.8669,
    "auroc_drop": 0.0238
  },
  {
    "feature_group": "신장(Cr/BUN)",
    "base_auroc": 0.8907,
    "ablated_auroc": 0.8863,
    "auroc_drop": 0.0043
  },
  {
    "feature_group": "소변량",
    "base_auroc": 0.8907,
    "ablated_auroc": 0.7422,
    "auroc_drop": 0.1485
  }
]
```

## Time-series validation (4.3)

```json
{
  "leakage_check": {
    "n_aki": 1598,
    "n_cutoff_before_onset": 1598,
    "n_leakage": 0,
    "passed": true
  },
  "label_consistency": {
    "n": 6433,
    "n_match": 6433,
    "match_rate": 1.0,
    "passed": true
  }
}
```

## Clinical validity — 단조성(배포 예측기, 4.2)

```json
[
  {
    "check": "creatinine_monotonic",
    "description": "Cr 상승(baseline 대비 배수↑) → 위험점수 비감소",
    "axis_values": [
      1.0,
      1.5,
      2.0,
      2.5,
      3.0,
      3.5
    ],
    "risk_scores": [
      13,
      58,
      77,
      77,
      91,
      91
    ],
    "passed": true,
    "note": "KDIGO: Cr 1.5/2.0/3.0배는 Stage 1/2/3 기준"
  },
  {
    "check": "egfr_monotonic",
    "description": "eGFR 저하 → 위험점수 비감소",
    "axis_values": [
      90.0,
      60.0,
      45.0,
      30.0,
      20.0,
      10.0
    ],
    "risk_scores": [
      13,
      13,
      13,
      13,
      34,
      58
    ],
    "passed": true,
    "note": "eGFR<30/<15 에서 가중"
  },
  {
    "check": "urine_output_monotonic",
    "description": "소변량 감소(핍뇨/무뇨) → 위험점수 비감소",
    "axis_values": [
      1.5,
      1.0,
      0.6,
      0.5,
      0.3,
      0.1
    ],
    "risk_scores": [
      13,
      13,
      13,
      13,
      47,
      68
    ],
    "passed": true,
    "note": "KDIGO 소변량 기준 <0.5(핍뇨)/<0.3(무뇨)"
  }
]
```

## Known failure cases

- Stage2(LGBM) 중증 판별 AUROC 0.578 — 우연(0.5) 대비 낮은 변별력. 중증도 분류는 보조 지표로만 사용, 단독 임상 판단 금지.
- Stage2 AUPRC 0.263 — 양성(Stage2+3) 희소로 정밀도 한계(고recall/저precision).
- '신장(Cr/BUN)' 제거 시 AUROC 변화 +0.0043 — 모델이 해당 신호를 거의 사용하지 않음.
- 결측 많은 환자 AUROC 0.843 (적은 환자 0.950) — 결측 많을수록 성능 저하 Δ0.107.
- Stage1 보정 불량(ECE 0.179) — 예측확률 0.7–0.9 구간이 실제 위험을 과대평가. 운영 시 확률값보다 위험등급(threshold 0.7) 사용 권장. 재보정(Platt/Isotonic) 필요.
- ICU/ER/ward subgroup 분석 불가 — 데이터셋에 careunit 컬럼 없음(4.1 요건 부분 미충족). 대체로 성별·연령분위·결측수준 subgroup 으로 검증함.
