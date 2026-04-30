import pandas as pd
import re

CSV_PATH = 'radiology_nlp_text.csv'
df = pd.read_csv(CSV_PATH)
print(f"로드 완료: {len(df):,}개 노트 | {df['stay_id'].nunique():,}명 환자")
# 결과: 54,415개 노트 | 27,049명 환자

def merge_text(row):
    imp = str(row['impression']).strip() if pd.notna(row['impression']) else ''
    fin = str(row['findings']).strip() if pd.notna(row['findings']) else ''
    return imp if imp else (fin if fin else None)

df['nlp_text'] = df.apply(merge_text, axis=1)
print(f"nlp_text NULL: {df['nlp_text'].isna().sum():,}개")
print(f"nlp_text 있음: {df['nlp_text'].notna().sum():,}개")
# 결과: NULL 5,020개 / 있음 49,395개

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'_{2,}', ' ', text)              # MIMIC-IV de-id 태그 제거
    text = re.sub(r'[^\w\s\.\,\;\:\-]', ' ', text)  # 특수문자 정리
    text = re.sub(r'\s+', ' ', text)                 # 연속 공백 정리
    return text.lower().strip()                      # 소문자화

df['nlp_text_clean'] = df['nlp_text'].apply(clean_text)

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

agg = {k: 'max' for k in AKI_KEYWORDS}
agg['nlp_text_clean'] = lambda x: ' '.join(x[x != ''])

patient_df = df.groupby('stay_id').agg(agg).reset_index()
patient_df.rename(columns={'nlp_text_clean': 'nlp_text_combined'}, inplace=True)

# 6단계에서 쓸 컬럼 목록 정의
out_cols = ['stay_id', 'nlp_text_combined'] + list(AKI_KEYWORDS.keys())

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
