# AI Draft 파이프라인 실행 결과 (작업지시서 11)

_Audio → STT → SOAP → Problem List → CDSS Risk → Timeline_

## 1. Transcript (STT)

> 어제부터 다리가 붓고 소변이 줄었어요. 메스껍고 기운이 없어요.

## 2. SOAP (evidence mapping)

- **S**: 어제부터 다리가 붓고 소변이 줄었어요. 메스껍고 기운이 없어요. (추출 증상: 부종, 핍뇨)
- **O**: Creatinine 4.2 mg/dL. eGFR 14.0 mL/min. K 6.3 mmol/L. 시간당 소변량 0.2 mL/kg/h. AKI 위험점수 100점

**A — 기저 진단: 당뇨병성 신증 · AKI 평가: AKI Stage 2-3 (고위험) · 체액 과부하/신기능 저하 가능성 — 임상 상관 필요**

| statement | evidence |
|---|---|
| 기저 진단: 당뇨병성 신증 | 환자 기록 진단: 당뇨병성 신증 |
| AKI 평가: AKI Stage 2-3 (고위험) | Cr 4.2배 상승 (KDIGO Stage 3 기준); 48h Cr 증가 3.2 mg/dL (≥0.3); Cr 절대값 4.2 mg/dL (≥4.0); Creatinine 4.2 mg/dL |
| 체액 과부하/신기능 저하 가능성 — 임상 상관 필요 | 어제부터 다리가 붓고 소변이 줄었어요. 메스껍고 기…; 어제부터 다리가 붓고 소변이 줄었어요. 메스껍고 기운이 없어요. |

**P — 신독성 약물 점검/중단, 전해질 응급 교정, 신대체요법 적응증 확인, 신장내과 긴급 협진 · 고칼륨혈증 교정(칼슘/인슐린-포도당/케이엑살레이트) 및 ECG 모니터 · 엄격한 수분 출납(I/O) 및 시간당 소변량 추적**

| statement | evidence |
|---|---|
| 신독성 약물 점검/중단, 전해질 응급 교정, 신대체요법 적응증 확인, 신장내과 긴급 협진 | Cr 4.2배 상승 (KDIGO Stage 3 기준); 48h Cr 증가 3.2 mg/dL (≥0.3) |
| 고칼륨혈증 교정(칼슘/인슐린-포도당/케이엑살레이트) 및 ECG 모니터 | K 6.3 mmol/L |
| 엄격한 수분 출납(I/O) 및 시간당 소변량 추적 | 시간당 소변량 0.2 mL/kg/h; 어제부터 다리가 붓고 소변이 줄었어요. 메스껍고 기운이 없어요. |

## 3. Problem List (A 기반)

| problem | source | confidence | evidence |
|---|---|---:|---|
| 당뇨병성 신증 | A | 0.65 | 환자 기록 진단: 당뇨병성 신증 |
| Acute kidney injury (N17) | A | 0.95 | Cr 4.2배 상승 (KDIGO Stage 3 기준); 48h Cr 증가 3.2 mg/dL (≥0.3); Cr 절대값 4.2 mg/dL (≥4.0); Creatinine 4.2 mg/dL |
| Fluid overload (E87.7) | A | 0.8 | 어제부터 다리가 붓고 소변이 줄었어요. 메스껍고 기…; 어제부터 다리가 붓고 소변이 줄었어요. 메스껍고 기운이 없어요. |

## 4. CDSS Risk Score breakdown

**risk_score = 0.96 → HIGH** (alert: modal, nephrology_trigger: True)

| component | value | weight | contribution | explanation |
|---|---:|---:|---:|---|
| creatinine_trend | 1.0 | 0.4 | 0.4 | Cr 1.0→4.2 (4.2배) |
| urine_output_drop | 1.0 | 0.3 | 0.3 | 0.2 mL/kg/h (무뇨) |
| diagnosis_risk_weight | 1.0 | 0.2 | 0.2 | AKI 모델 P(S2+3)=1.00,P(S1)=0.00, problems=3 [rule-based] |
| vitals_instability | 0.6 | 0.1 | 0.06 | K 6.3(고칼륨) |

score = 0.4×1.0 + 0.3×1.0 + 0.2×1.0 + 0.1×0.6

## 5. Timeline event record

| event_type | severity | title | payload |
|---|---|---|---|
| ai_draft_note | INFO | AI 진료초안 생성 (위험 HIGH) | {"draftId": "draft-1a2df3149de6", "riskScore": 0.96, "tier": "HIGH"} |
| AI_ALERT | CRITICAL | CDSS 고위험 경보 (risk 0.96) | {"riskScore": 0.96, "nephrologyTrigger": true} |

## 6. Validation report

**passed: True**

- ✅ `soap_validator` — errors=0, warnings=0
- ✅ `ap_evidence_validator` — errors=0, warnings=0
- ✅ `risk_score_validator` — errors=0, warnings=0
