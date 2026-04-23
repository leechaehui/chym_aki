# 🏥 CHYM 프로젝트 — 트랙 D: NLP 피처 파이프라인

## 📋 개요

MIMIC-IV `mimiciv_note.radiology` 테이블에서 AKI(급성 신손상) 예측용 NLP 피처를 추출하는 파이프라인입니다.

- **대상 코호트**: 42,210명 (ICU 입실 환자, 18세 이상, 24시간 이상 체류)
- **데이터 소스**: MIMIC-IV `mimiciv_note.radiology`
- **최종 산출물**: `nlp_keyword_features.csv` (42,210행 × 7컬럼)

---

## 📁 파일 구조

```
kidney_nlp/
├── radiology_nlp_text.csv       # DBeaver에서 추출한 방사선 판독문 원본
├── cohort_stay_ids.csv          # 코호트 전체 stay_id 목록 (42,210명)
├── nlp_code.py                  # NLP 전처리 메인 코드
└── nlp_keyword_features.csv     # 최종 NLP 피처 (산출물)
```

---

## 🗄️ 데이터 추출 (SQL)

### MIMIC-IV Note 테이블 구조

MIMIC-IV Note에는 총 4개 테이블이 있습니다.

| 테이블 | 내용 | NLP 사용 여부 |
|--------|------|--------------|
| `radiology` | 방사선 판독문 전체 텍스트 | ✅ 채택 |
| `radiology_detail` | 검사 코드/이름 메타데이터 | ❌ (텍스트 없음) |
| `discharge` | 퇴원 요약 | ❌ (퇴원 후 작성 → Leakage) |
| `discharge_detail` | 퇴원 요약 상세 | ❌ (퇴원 후 작성 → Leakage) |

> ⚠️ 간호 노트, 의사 노트는 MIMIC-IV에 없습니다. MIMIC-III에만 존재합니다.

### 커버리지 확인 결과

```
전체 코호트:       42,210명
노트 보유 환자:    25,953명 (61.5%)
전체 노트 수:      54,415개 (환자당 평균 2.1개)
노트 없는 환자:    15,161명 (38.5%) → 제로 벡터 처리
```

### NLP 텍스트 추출 쿼리

```sql
-- ⚠️ Leakage 방지: ICU 입실 후 24시간 이내 노트만 사용
SELECT
    r.note_id,
    r.subject_id,
    r.hadm_id,
    c.stay_id,
    r.charttime,
    TRIM(SUBSTRING(r.text FROM 'FINDINGS:([\s\S]*?)(?=\n[A-Z ]+:|$)'))   AS findings,
    TRIM(SUBSTRING(r.text FROM 'IMPRESSION:([\s\S]*?)(?=\n[A-Z ]+:|$)')) AS impression
FROM mimiciv_note.radiology r
JOIN cohort c
    ON  r.hadm_id   = c.hadm_id
    AND r.charttime >= c.icu_intime                        -- ICU 입실 이후
    AND r.charttime <= c.icu_intime + INTERVAL '24 시간 전'  -- 24시간 이내
ORDER BY c.stay_id, r.charttime;
-- 저장 파일명: radiology_nlp_text.csv
```

```sql
-- 코호트 전체 stay_id 추출 (제로 벡터 처리용)
SELECT stay_id FROM cohort;
-- 저장 파일명: cohort_stay_ids.csv
```

---

## 🐍 Python NLP 파이프라인

### 환경 요구사항

```bash
pip install pandas
```

### 실행 방법

```bash
python nlp_code.py
```

### 단계별 설명

#### 1단계 — 데이터 로드
DBeaver에서 추출한 CSV를 Python으로 불러옵니다.

```
입력: radiology_nlp_text.csv
결과: 54,415개 노트 / 27,049명 환자
```

```python
import pandas as pd
import re

CSV_PATH = 'radiology_nlp_text.csv'
df = pd.read_csv(CSV_PATH)
print(f"로드 완료: {len(df):,}개 노트 | {df['stay_id'].nunique():,}명 환자")
# 결과: 54,415개 노트 | 27,049명 환자
```

---

#### 2단계 — nlp_text 생성
방사선 판독문은 IMPRESSION / FINDINGS 두 섹션으로 나뉩니다.
IMPRESSION이 판독 결론이라 더 핵심 정보를 담고 있어 우선순위를 높게 설정했습니다.

```
우선순위: IMPRESSION → FINDINGS → NULL (둘 다 없으면)

섹션별 NULL 현황:
  findings 없음:   20,963개 (38.5%)
  impression 없음: 12,622개 (23.2%)
  둘 다 없음:       4,983개  (9.2%)
```

```python
def merge_text(row):
    imp = str(row['impression']).strip() if pd.notna(row['impression']) else ''
    fin = str(row['findings']).strip() if pd.notna(row['findings']) else ''
    return imp if imp else (fin if fin else None)

df['nlp_text'] = df.apply(merge_text, axis=1)
print(f"nlp_text NULL: {df['nlp_text'].isna().sum():,}개")
print(f"nlp_text 있음: {df['nlp_text'].notna().sum():,}개")
# 결과: NULL 5,020개 / 있음 49,395개
```

---

#### 3단계 — 텍스트 전처리
키워드 검색이 잘 되도록 텍스트를 정리합니다.

```
처리 내용:
  ___ 제거      → MIMIC-IV 익명화 태그 (환자 이름, 날짜 등)
  특수문자 제거  → 의미 없는 기호 정리
  소문자화      → 대소문자 구분 없이 키워드 검색

전처리 전: 'No evidence of pneumothorax following unsuccessful left\nsubclavian line placement attempt.'
전처리 후: 'no evidence of pneumothorax following unsuccessful left subclavian line placement attempt.'
```

```python
def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'_{2,}', ' ', text)              # MIMIC-IV de-id 태그 제거
    text = re.sub(r'[^\w\s\.\,\;\:\-]', ' ', text)  # 특수문자 정리
    text = re.sub(r'\s+', ' ', text)                 # 연속 공백 정리
    return text.lower().strip()                      # 소문자화

df['nlp_text_clean'] = df['nlp_text'].apply(clean_text)
```

---

#### 4단계 — 키워드 플래그 추출
54,415개 노트 각각에 대해 5개 키워드를 검사합니다.
키워드가 텍스트에 있으면 1, 없으면 0인 이진 피처입니다.

**키워드 선정 근거**

| 키워드 | 검출 (노트) | 임상적 근거 |
|--------|------------|------------|
| `kw_fluid_overload` | 27.2% | AKI로 인한 핍뇨 → 체액 축적 → 폐부종/흉수/복수. 흉부 X-ray에서 가장 흔한 AKI 이차 소견 |
| `kw_edema` | 19.2% | AKI로 신장이 수분 배출 못함 → 전신 부종. 방사선에서 연부조직 부종/복수로 확인 |
| `kw_hydronephrosis` | 0.5% | 요로 폐색으로 소변 역류 → 신후성(Post-renal) AKI의 직접 원인 |
| `kw_renal_abnormal` | 0.1% | 신장 크기/에코 변화. 에코 증가는 신장 실질 손상 신호 |
| `kw_aki_mention` | 0.0% | 방사선과 의사가 판독문에 AKI 직접 기재. 드물지만 가장 확실한 신호 |

> ⚠️ `kw_oliguria`, `kw_anuria`는 제외했습니다.
> 방사선 판독문에는 소변량 정보가 없어 검출율이 0%이기 때문입니다.
> 소변량은 `outputevents` 수치 피처로 대체합니다.

```python
AKI_KEYWORDS = {
    # 수신증 — 요로 폐색으로 소변 역류 → 신후성(Post-renal) AKI의 직접 원인
    'kw_hydronephrosis': r'\b(hydronephrosis|hydroureter|pelviectasis)\b',

    # AKI 직접 언급 — 드물지만 가장 확실한 신호
    'kw_aki_mention':    r'\b(acute kidney injury|aki|acute renal failure|acute tubular necrosis|atn)\b',

    # 신장 구조 이상 — 에코 증가는 신장 실질 손상 신호
    # 신전성/신성 AKI 진행 시 초음파에서 관찰
    'kw_renal_abnormal': r'\b(renal enlargement|cortical thinning|echogenicity|nephrolithiasis)\b',

    # 부종 — AKI로 수분 배출 못함 → 전신 부종
    # 방사선에서 연부조직 부종/복수로 확인 가능
    'kw_edema':          r'\b(edema|anasarca|fluid overload|volume overload|pitting)\b',

    # 체액 과부하 — AKI로 인한 핍뇨 → 체액 축적
    # → 폐부종/흉수/복수. 흉부 X-ray에서 가장 흔한 AKI 이차 소견
    'kw_fluid_overload': r'\b(pulmonary edema|pleural effusion|ascites|vascular congestion|interstitial edema)\b',
}

for key, pat in AKI_KEYWORDS.items():
    df[key] = df['nlp_text_clean'].apply(
        lambda t: int(bool(re.search(pat, t, re.IGNORECASE)))
    )
```

---

#### 5단계 — stay_id 단위 집계
한 환자가 여러 노트를 가질 수 있어서(평균 2.1개) 환자 단위로 압축합니다.
OR 집계 방식으로 여러 노트 중 하나라도 키워드가 나오면 1로 처리합니다.

```
54,415개 노트 → 27,049명으로 압축

노트 단위 → 환자 단위 비율 변화 (OR 집계로 올라감):
  kw_fluid_overload: 27.2% → 42.9%
  kw_edema:          19.2% → 31.5%
  kw_hydronephrosis:  0.5% →  1.0%
  kw_renal_abnormal:  0.1% →  0.3%
  kw_aki_mention:     0.0% →  0.1%
```

```python
agg = {k: 'max' for k in AKI_KEYWORDS}
agg['nlp_text_clean'] = lambda x: ' '.join(x[x != ''])

patient_df = df.groupby('stay_id').agg(agg).reset_index()
patient_df.rename(columns={'nlp_text_clean': 'nlp_text_combined'}, inplace=True)

# 6단계에서 쓸 컬럼 목록 정의
out_cols = ['stay_id', 'nlp_text_combined'] + list(AKI_KEYWORDS.keys())
```

---

#### 6단계 — 코호트 42,210명 기준 맞추기 (제로 벡터)
노트 없는 15,161명을 키워드 전부 0으로 채워 42,210명으로 맞춥니다.
XGBoost에 넣을 때 AKI 레이블(42,210명)과 행 수가 일치해야 하기 때문입니다.

```
cohort_stay_ids.csv (42,210명)
        ↓ LEFT JOIN
patient_df (27,049명)
        ↓ NaN → 0
final_df (42,210명) → nlp_keyword_features.csv
```

```python
cohort_df = pd.read_csv('cohort_stay_ids.csv')

# LEFT JOIN → 노트 없는 환자는 NaN
final_df = cohort_df.merge(patient_df[out_cols], on='stay_id', how='left')

# NaN → 0 (제로 벡터)
for k in AKI_KEYWORDS:
    final_df[k] = final_df[k].fillna(0).astype(int)

# nlp_text_combined NaN → 빈 문자열
final_df['nlp_text_combined'] = final_df['nlp_text_combined'].fillna('')

final_df.to_csv('nlp_keyword_features.csv', index=False)
# 결과: shape (42210, 7) / 노트 없는 환자 15,161명 제로 벡터
```

---

## 📊 최종 산출물

**nlp_keyword_features.csv** (42,210행 × 7컬럼)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `stay_id` | int | ICU 입실 ID |
| `nlp_text_combined` | str | 전처리된 판독문 텍스트 (노트 없으면 빈 문자열) |
| `kw_fluid_overload` | 0/1 | 체액 과부하 키워드 (pulmonary edema, pleural effusion 등) |
| `kw_edema` | 0/1 | 부종 키워드 (edema, ascites 등) |
| `kw_hydronephrosis` | 0/1 | 수신증 키워드 |
| `kw_renal_abnormal` | 0/1 | 신장 구조 이상 키워드 |
| `kw_aki_mention` | 0/1 | AKI 직접 언급 키워드 |

---

## ⚠️ 주의사항

**Leakage 방지**
방사선 판독문 추출 시 반드시 아래 조건을 적용해야 합니다.
이 조건 없이 추출하면 AKI 발생 이후 노트가 피처에 섞여 데이터 리키지가 발생합니다.

```sql
AND r.charttime >= c.icu_intime
AND r.charttime <= c.icu_intime + INTERVAL '24 시간 전'
```

**노트 없는 환자 처리**
노트가 없는 15,161명(38.5%)은 키워드 피처를 전부 0으로 채웁니다.
XGBoost는 0값을 자연스럽게 처리하며, "노트 없음" 자체가 임상적 신호일 수 있습니다.
