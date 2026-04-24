-- Step 1. lab_features   크레아티닌, BUN, 전해질
-- Step 2. vital_features  활력징후 (혈압, 심박수, 호흡수)
-- Step 3. urine_features 소변량, 수액 균형
-- Step 4. numeric_features  세 개 통합

--모든 피처 추출 쿼리에 이 조건이 반드시 들어감
--JOIN aki_stage_final a ON c.stay_id = a.stay_id
-- WHERE charttime >= c.icu_intime
-- AND   charttime <= COALESCE(a.prediction_cutoff, c.icu_outtime)


-- step1 lab_features 혈액검사 피처
-- labevents에서 신기능 관련 혈액검사 수치를 가져와서 prediction_cutoff기준으로 집계

SELECT itemid, label
FROM mimiciv_hosp.d_labitems
WHERE label ILIKE '%creatinine%'
   OR label ILIKE '%urea nitrogen%'
   OR label ILIKE '%potassium%'
   OR label ILIKE '%bicarbonate%'
   OR label ILIKE '%lactate%'
   OR label ILIKE '%hemoglobin%'
ORDER BY label;

--위 쿼리를 확인하니까 같은 항목이 여러개 있는 경우가 있음
-- 예를 들어 Creatinine이 50912, 52546등 여러개 인걸 볼 수 있음
-- 어떤 걸 써야 하는지 실제 데이터 건수를 보고 결정해야함

SELECT
    d.itemid,
    d.label,
    COUNT(*) AS n_records
FROM mimiciv_hosp.labevents l
JOIN mimiciv_hosp.d_labitems d ON l.itemid = d.itemid
WHERE d.itemid IN (
    50912, 52546,        -- Creatinine
    51006, 52647,        -- Urea Nitrogen (BUN)
    50971, 52610, 50833, -- Potassium
    50882, 52039,        -- Bicarbonate
    50813, 52442, 50803, -- Lactate
    51222, 50811, 50855  -- Hemoglobin
)
AND l.valuenum IS NOT NULL
GROUP BY d.itemid, d.label
ORDER BY d.label, n_records DESC;

-- 52546, 50803, 52610, 52647, 50811은 데이터가 너무 적거나, 계산값, 실측값이 아니거나 해서 제외시킴

DROP TABLE IF EXISTS lab_features;

CREATE TABLE lab_features AS

WITH raw_labs AS (
    SELECT
        c.stay_id,
        a.prediction_cutoff,
        a.aki_label,
        l.charttime,
        l.itemid,
        l.valuenum
    FROM cohort c
    JOIN aki_stage_final a
        ON  c.stay_id = a.stay_id
    JOIN mimiciv_hosp.labevents l
        ON  c.subject_id = l.subject_id
        AND l.charttime >= c.icu_intime
        AND l.charttime <= COALESCE(
                a.prediction_cutoff,
                c.icu_outtime
            )
    WHERE l.itemid IN (
        50912,   -- Creatinine
        50882,   -- Bicarbonate
        50971,   -- Potassium
        51006,   -- Urea Nitrogen (BUN)
        51222,   -- Hemoglobin
        50813    -- Lactate
    )
    AND l.valuenum IS NOT NULL
)

SELECT
    stay_id,
    aki_label,

    -- 크레아티닌 (AKI 핵심 지표)
    MIN(CASE WHEN itemid = 50912
        THEN valuenum END)              AS cr_min,
    MAX(CASE WHEN itemid = 50912
        THEN valuenum END)              AS cr_max,
    AVG(CASE WHEN itemid = 50912
        THEN valuenum END)              AS cr_mean,
    MAX(CASE WHEN itemid = 50912
        THEN valuenum END)
    - MIN(CASE WHEN itemid = 50912
        THEN valuenum END)              AS cr_delta,

    -- BUN (신기능 보조 지표)
    MAX(CASE WHEN itemid = 51006
        THEN valuenum END)              AS bun_max,
    AVG(CASE WHEN itemid = 51006
        THEN valuenum END)              AS bun_mean,

    -- 칼륨 (신장이 못 걸러내면 올라감)
    MAX(CASE WHEN itemid = 50971
        THEN valuenum END)              AS potassium_max,
    AVG(CASE WHEN itemid = 50971
        THEN valuenum END)              AS potassium_mean,

    -- 중탄산염 (산염기 균형)
    MIN(CASE WHEN itemid = 50882
        THEN valuenum END)              AS bicarbonate_min,
    AVG(CASE WHEN itemid = 50882
        THEN valuenum END)              AS bicarbonate_mean,

    -- 헤모글로빈 (빈혈 → 산소 운반 감소)
    MIN(CASE WHEN itemid = 51222
        THEN valuenum END)              AS hemoglobin_min,
    AVG(CASE WHEN itemid = 51222
        THEN valuenum END)              AS hemoglobin_mean,

    -- 젖산 (조직 저산소 지표)
    MAX(CASE WHEN itemid = 50813
        THEN valuenum END)              AS lactate_max,
    AVG(CASE WHEN itemid = 50813
        THEN valuenum END)              AS lactate_mean,

    -- BUN/Creatinine 비율 (신전성 AKI 감별)
    CASE
        WHEN AVG(CASE WHEN itemid = 50912
             THEN valuenum END) > 0
        THEN AVG(CASE WHEN itemid = 51006
             THEN valuenum END)
           / AVG(CASE WHEN itemid = 50912
             THEN valuenum END)
        ELSE NULL
    END                                 AS bun_cr_ratio

FROM raw_labs
GROUP BY stay_id, aki_label;

-- 위 코드에 대한 설명! 

-- COALESCE(a.prediction_cutoff, c.icu_outtime)
-- AKI 있는 환자 : prediction_cutoff까지
-- AKI 없는 환자:  prediction_cutoff = NULL 
     --→ COALESCE가 icu_outtime으로 대체 , ICU전체 기간 데이터 사용

-- CASE WHEN itemid = 50912 THEN valuenum END
-- itemid별로 컬럼을 나누는 방법
--한 쿼리에서 여러 항목을 동시에 집계
--itemid가 50912(크레아티닌)인 행만 집계
--나머지는 NULL로 처리됨

-- cr_delta = MAX - MIN
-- 예측 구간 내 크레아티닌 최댓값 - 최솟값
-- = 얼마나 변화했는가
-- = AKI 예측에서 가장 중요한 단일 피처

-- bun_cr_ratio
-- BUN / Creatinine 비율
-- 20 이상: 신전성 AKI (혈류 부족)
-- 10 이하: 신성 AKI (신장 실질 손상)
-- → AKI 원인 감별에 도움

SELECT
    COUNT(*) AS n_total,
    COUNT(cr_min) AS n_has_creatinine,
    COUNT(bun_max) AS n_has_bun,
    COUNT(lactate_max) AS n_has_lactate,
    ROUND(AVG(cr_delta)::NUMERIC, 2) AS avg_cr_delta,
    ROUND(AVG(bun_cr_ratio)::NUMERIC, 1) AS avg_bun_cr_ratio
FROM lab_features;


-- 결과 분석
-- n_total           33,267명
-- 42,210명 중 33,267명만 lab_features에 있음
--→ 8,943명은 prediction_cutoff 이전 구간에
--  혈액검사 기록이 전혀 없음
--→ 이 환자들은 lab 피처가 전부 NULL
--→ 나중에 Python에서 결측값 처리

-- n_has_creatinine  33,166명  (99.7%)
-- n_has_bun         33,161명  (99.7%)
-- 크레아티닌/BUN 보유율 99.7%
-- 혈액검사 기록 있는 환자 중
--거의 전원이 크레아티닌, BUN 기록 있음
--→ 신기능 지표는 ICU에서 거의 항상 측정됨
--→ 결측 거의 없음 (좋은 신호)

-- n_has_lactate     17,098명  (51.4%)
-- 젖산 보유율 51.4%
-- 절반만 젖산 기록 있음
--→ 젖산은 모든 환자에게 측정하지 않음
--→ 쇼크, 패혈증 의심될 때만 측정
--→ 결측이 많아도 정보 자체는 의미 있음
  -- (젖산 측정했다는 것 자체가 위험 신호)
--→ Python에서 결측 여부를 별도 피처로 만들 예정

-- avg_cr_delta       0.15 mg/dL
-- 피처 구간 내 평균 크레아티닌 변화량 0.15 mg/dL
--→ AKI 기준이 0.3이므로 평균의 절반
--→ 일부 환자는 이미 0.3 이상 변화 있음
--→ 이게 AKI 예측의 핵심 신호

--avg_bun_cr_ratio   20.8
--평균 20.8 → 20 이상
--→ 신전성 AKI (혈류 부족으로 인한 AKI)가 많음
--→ ICU 환자의 전형적인 패턴
--→ 임상적으로 타당한 결과




--step2 vital_features (활력징후 피처)
-- chartevents에서 혈압, 심박수, 호흡수, 체온, 산소포화도 추출
-- 혈압 낮음  → 신장으로 가는 혈류 감소 → AKI 위험
--심박수 높음 → 쇼크 상태 가능성 → AKI 위험
--호흡수 높음 → 산증, 패혈증 신호 → AKI 위험
-- 혈액검사는 하루 1~2번이지만 활력징후는 1시간마다 기록됩니다. 그래서 더 촘촘하게 상태 변화를 추적할 수 있습니다.

SELECT
    d.itemid,
    d.label,
    COUNT(*) AS n_records
FROM mimiciv_icu.chartevents ce
JOIN mimiciv_icu.d_items d ON ce.itemid = d.itemid
WHERE
    d.label ILIKE '%heart rate%'
    OR d.label ILIKE '%blood pressure%'
    OR d.label ILIKE '%respiratory rate%'
    OR d.label ILIKE '%temperature%'
    OR d.label ILIKE '%spo2%'
    OR d.label ILIKE '%oxygen saturation%'
AND ce.valuenum IS NOT NULL
GROUP BY d.itemid, d.label
ORDER BY d.label, n_records DESC;

SELECT
    d.itemid,
    d.label,
    COUNT(*) AS n_records
FROM mimiciv_icu.chartevents ce
JOIN mimiciv_icu.d_items d ON ce.itemid = d.itemid
WHERE
    d.label ILIKE '%spo2%'
    OR d.label ILIKE '%o2 sat%'
    OR d.label ILIKE '%pulse ox%'
AND ce.valuenum IS NOT NULL
GROUP BY d.itemid, d.label
ORDER BY n_records DESC;


-- SpO2는 220277이 8,567,015건으로 압도적으로 많음 

DROP TABLE IF EXISTS vital_features;

CREATE TABLE vital_features AS

WITH raw_vitals AS (
    SELECT
        c.stay_id,
        a.prediction_cutoff,
        a.aki_label,
        ce.charttime,
        ce.itemid,
        ce.valuenum
    FROM cohort c
    JOIN aki_stage_final a
        ON  c.stay_id = a.stay_id
    JOIN mimiciv_icu.chartevents ce
        ON  ce.stay_id    = c.stay_id
        AND ce.charttime >= c.icu_intime
        AND ce.charttime <= COALESCE(
                a.prediction_cutoff,
                c.icu_outtime
            )
    WHERE ce.itemid IN (
        220045,   -- Heart Rate
        220210,   -- Respiratory Rate
        220050,   -- Arterial BP mean
        220179,   -- Non Invasive BP systolic
        220181,   -- Non Invasive BP mean
        223761,   -- Temperature Fahrenheit
        220277    -- SpO2
    )
    AND ce.valuenum IS NOT NULL

    -- 항목별 이상치 제거
    AND NOT (ce.itemid = 220045
             AND ce.valuenum NOT BETWEEN 20 AND 300)
    AND NOT (ce.itemid = 220210
             AND ce.valuenum NOT BETWEEN 4 AND 60)
    AND NOT (ce.itemid IN (220050, 220179, 220181)
             AND ce.valuenum NOT BETWEEN 20 AND 200)
    AND NOT (ce.itemid = 223761
             AND ce.valuenum NOT BETWEEN 86 AND 115)
    AND NOT (ce.itemid = 220277
             AND ce.valuenum NOT BETWEEN 50 AND 100)
)

SELECT
    stay_id,
    aki_label,

    -- 심박수
    MIN(CASE WHEN itemid = 220045
        THEN valuenum END)              AS hr_min,
    MAX(CASE WHEN itemid = 220045
        THEN valuenum END)              AS hr_max,
    AVG(CASE WHEN itemid = 220045
        THEN valuenum END)              AS hr_mean,

    -- 호흡수
    MAX(CASE WHEN itemid = 220210
        THEN valuenum END)              AS rr_max,
    AVG(CASE WHEN itemid = 220210
        THEN valuenum END)              AS rr_mean,

    -- 평균동맥압 (ABP 우선, 없으면 NIBP mean)
    MIN(COALESCE(
        CASE WHEN itemid = 220050
             THEN valuenum END,
        CASE WHEN itemid = 220181
             THEN valuenum END
    ))                                  AS map_min,
    AVG(COALESCE(
        CASE WHEN itemid = 220050
             THEN valuenum END,
        CASE WHEN itemid = 220181
             THEN valuenum END
    ))                                  AS map_mean,

    -- 수축기혈압
    MIN(CASE WHEN itemid = 220179
        THEN valuenum END)              AS sbp_min,
    AVG(CASE WHEN itemid = 220179
        THEN valuenum END)              AS sbp_mean,

    -- 체온 (화씨 → 섭씨 변환)
    MIN((CASE WHEN itemid = 223761
        THEN valuenum END - 32) * 5/9) AS temp_min,
    AVG((CASE WHEN itemid = 223761
        THEN valuenum END - 32) * 5/9) AS temp_mean,

    -- 산소포화도
    MIN(CASE WHEN itemid = 220277
        THEN valuenum END)              AS spo2_min,
    AVG(CASE WHEN itemid = 220277
        THEN valuenum END)              AS spo2_mean,

    -- 쇼크 지수 (심박수 / 수축기혈압)
    -- 1 이상이면 쇼크 상태 의심
    AVG(CASE WHEN itemid = 220045
        THEN valuenum END)
    / NULLIF(AVG(CASE WHEN itemid = 220179
        THEN valuenum END), 0)          AS shock_index

FROM raw_vitals
GROUP BY stay_id, aki_label;

-- 위에 코드를 설명하자면
-- 이상치를 제거함
-- 심박수      20~300     20 미만은 측정 오류
--호흡수       4~60      4 미만은 측정 오류
--혈압        20~200     20 미만은 오류
--체온(°F)   86~115     86°F(30°C) 미만은 오류
--SpO2       50~100     50 미만은 측정 오류

-- COALESCE로 ABP와 NIBP 통합
-- 동맥혈압(ABP)이 있으면 ABP 사용
--없으면 비침습혈압(NIBP mean) 사용
--→ 두 값을 합쳐서 MAP 결측을 최소화

-- 체온변환
-- (°F - 32) × 5/9 = °C
-- 예: 98.6°F → (98.6 - 32) × 5/9 = 37.0°C

-- 쇼크지수
--심박수 / 수축기혈압
--정상: 0.5~0.7
--1.0 이상: 쇼크 의심
--→ AKI의 강력한 예측 인자

-- NULLIF(..., 0)
-- 수축기혈압이 0이면 나누기 오류 발생
--NULLIF(값, 0) → 0이면 NULL 반환
--→ NULL / 숫자 = NULL (오류 방지)

SELECT
    COUNT(*) AS n_total,
    COUNT(hr_mean) AS n_has_hr,
    COUNT(map_min) AS n_has_map,
    COUNT(spo2_min) AS n_has_spo2,
    ROUND(AVG(map_min)::NUMERIC, 1) AS avg_map_min,
    ROUND(AVG(shock_index)::NUMERIC, 2) AS avg_shock_index
FROM vital_features;

-- 활력징후 보유율 99.9%
-- 거의 모든 환자에게 활력징후 기록 있음
-- ICU에서는 활력징후를 항상 모니터링하기 때문
-- 결측이 거의 없음 -> 좋은피처

-- avg_map_min 61.2 mmHg
-- 평균 최저 MAP 61.2
--정상 기준: 65 mmHg 이상
--→ 평균적으로 이미 위험 수준에 근접
--→ 많은 환자가 일시적으로 MAP 65 미만 경험
--→ 허혈성 손상 피처의 중요성 확인

-- avg_shock_index 0.71
-- 평균 0.71
--정상 기준: 0.5~0.7
--→ 평균이 정상 상한선에 걸쳐있음
--→ 일부 환자는 1.0 이상 (쇼크 상태)

-- 지금까지!
-- lab_features           33,267명    혈액검사 피처
-- vital_features         33,636명    활력징후 피처






--step3 urine_features (소변량 + 수액 균형 피처) 
-- outputevents + inputevents 활용
SELECT
    d.itemid,
    d.label,
    COUNT(*) AS n_records
FROM mimiciv_icu.inputevents ie
JOIN mimiciv_icu.d_items d ON ie.itemid = d.itemid
WHERE
    d.label ILIKE '%saline%'
    OR d.label ILIKE '%lactated%'
    OR d.label ILIKE '%dextrose%'
    OR d.label ILIKE '%water%'
    OR d.label ILIKE '%fluid%'
    OR d.label ILIKE '%crystalloid%'
AND ie.amount > 0
GROUP BY d.itemid, d.label
ORDER BY n_records DESC
LIMIT 20;

SELECT
    d.itemid,
    d.label,
    COUNT(*) AS n_records
FROM mimiciv_icu.inputevents ie
JOIN mimiciv_icu.d_items d ON ie.itemid = d.itemid
WHERE
    d.label ILIKE '%normal saline%'
    OR d.label ILIKE '%sodium chloride%'
    OR d.label ILIKE '%NS %'
    OR d.label ILIKE '%ringer%'
    OR d.label ILIKE '%bolus%'
    OR d.label ILIKE '%IV fluid%'
AND ie.amount > 0
GROUP BY d.itemid, d.label
ORDER BY n_records DESC
LIMIT 20;

SELECT
    d.itemid,
    d.label,
    COUNT(*) AS n_records
FROM mimiciv_icu.inputevents ie
JOIN mimiciv_icu.d_items d ON ie.itemid = d.itemid
WHERE ie.amount > 0
GROUP BY d.itemid, d.label
ORDER BY n_records DESC
LIMIT 30;

-- 결과 
--일반 수액 (crystalloid)
--  225158  NaCl 0.9% (생리식염수)  1,578,523건  ← 핵심
--  220949  Dextrose 5%            1,234,704건
--  225943  Solution                 698,484건
--  225828  LR (젖산링거액)           126,721건
--  225797  Free Water                97,226건

--승압제 (나중에 허혈성 피처에서 씀)
 -- 221906  Norepinephrine           459,800건
 --221749  Phenylephrine            209,374건

--신독성 약물
--  225798  Vancomycin               139,991건  ← 트랙 C에서 씀
--  221833  Hydromorphone            125,390건
--  221794  Furosemide (Lasix)       117,055건

DROP TABLE IF EXISTS urine_features;

CREATE TABLE urine_features AS

WITH
-- 소변량 집계 (outputevents)
urine_agg AS (
    SELECT
        c.stay_id,
        a.prediction_cutoff,

        -- 6시간 소변량 (마지막 6시간)
        SUM(CASE
            WHEN oe.charttime >= COALESCE(
                a.prediction_cutoff,
                c.icu_outtime
            ) - INTERVAL '6 hours'
            THEN oe.value ELSE 0
        END)                            AS urine_6h,

        -- 24시간 소변량
        SUM(CASE
            WHEN oe.charttime >= COALESCE(
                a.prediction_cutoff,
                c.icu_outtime
            ) - INTERVAL '24 hours'
            THEN oe.value ELSE 0
        END)                            AS urine_24h,

        -- 전체 구간 소변량
        SUM(oe.value)                   AS urine_total,

        -- 소변량이 0인 시간 비율
        COUNT(CASE WHEN oe.value = 0
              THEN 1 END) * 1.0
        / NULLIF(COUNT(*), 0)           AS urine_zero_ratio

    FROM cohort c
    JOIN aki_stage_final a
        ON  c.stay_id = a.stay_id
    JOIN mimiciv_icu.outputevents oe
        ON  oe.stay_id    = c.stay_id
        AND oe.charttime >= c.icu_intime
        AND oe.charttime <= COALESCE(
                a.prediction_cutoff,
                c.icu_outtime
            )
    WHERE oe.itemid IN (
        226559, 226560, 226561,
        226563, 226627, 226631, 226584
    )
    AND oe.value >= 0
    AND oe.value < 2000
    GROUP BY c.stay_id, a.prediction_cutoff
),

-- 수액 투여량 집계 (inputevents)
fluid_agg AS (
    SELECT
        c.stay_id,

        -- 24시간 수액 투여량
        SUM(CASE
            WHEN ie.starttime >= COALESCE(
                a.prediction_cutoff,
                c.icu_outtime
            ) - INTERVAL '24 hours'
            THEN ie.amount ELSE 0
        END)                            AS fluid_24h,

        -- 전체 구간 수액 투여량
        SUM(ie.amount)                  AS fluid_total

    FROM cohort c
    JOIN aki_stage_final a
        ON  c.stay_id = a.stay_id
    JOIN mimiciv_icu.inputevents ie
        ON  ie.stay_id     = c.stay_id
        AND ie.starttime  >= c.icu_intime
        AND ie.starttime  <= COALESCE(
                a.prediction_cutoff,
                c.icu_outtime
            )
    WHERE ie.itemid IN (
        225158,   -- NaCl 0.9%
        220949,   -- Dextrose 5%
        225943,   -- Solution
        225828,   -- LR
        225797    -- Free Water
    )
    AND ie.amount > 0
    AND ie.amountuom = 'mL'
    GROUP BY c.stay_id
)

-- 두 개 합치고 수액 균형 계산
SELECT
    c.stay_id,
    a.aki_label,

    -- 소변량 피처
    COALESCE(u.urine_6h, 0)             AS urine_6h,
    COALESCE(u.urine_24h, 0)            AS urine_24h,
    COALESCE(u.urine_total, 0)          AS urine_total,
    COALESCE(u.urine_zero_ratio, 0)     AS urine_zero_ratio,

    -- 수액 피처
    COALESCE(f.fluid_24h, 0)            AS fluid_24h,
    COALESCE(f.fluid_total, 0)          AS fluid_total,

    -- 수액 균형 (들어온 것 - 나간 것)
    -- 양수: 수액 많이 줬는데 소변 안 나옴 → 위험
    -- 음수: 소변이 더 많이 나옴 → 정상
    COALESCE(f.fluid_24h, 0)
    - COALESCE(u.urine_24h, 0)          AS fluid_balance_24h,

    COALESCE(f.fluid_total, 0)
    - COALESCE(u.urine_total, 0)        AS fluid_balance_total

FROM cohort             c
JOIN aki_stage_final    a ON c.stay_id = a.stay_id
LEFT JOIN urine_agg     u ON c.stay_id = u.stay_id
LEFT JOIN fluid_agg     f ON c.stay_id = f.stay_id;


-- 왜 6시간 24시간 두 가지를 모두를 보는가
-- 6시간 소변량
--  → 가장 최근 상태를 반영
--  → KDIGO 소변량 기준이 6시간 단위
--  → 급격한 변화를 빠르게 포착

--24시간 소변량
--  → 하루 전체 경향을 반영
--  → 일시적 감소 vs 지속적 감소 구별
--  → 더 안정적인 지표


-- 수액균형(fluid_balance)이 왜 중요한가 
--예시 1: 수액 균형 +2000mL
--  수액 2000mL 줬는데 소변 0mL
--  → 신장이 전혀 기능 안 함
--  → 매우 위험

--예시 2: 수액 균형 -500mL
--  수액 500mL 줬는데 소변 1000mL
 -- → 소변이 더 많이 나옴
--  → 정상 or 이뇨제 효과

-- COALESCE(..., 0)
--소변량이나 수액 기록이 없는 환자는 NULL
--COALESCE(NULL, 0) → 0으로 처리
--기록 없음 = 투여/배출 없음으로 간주

SELECT
    COUNT(*) AS n_total,
    COUNT(CASE WHEN urine_6h > 0
          THEN 1 END) AS n_has_urine,
    ROUND(AVG(urine_6h)::NUMERIC, 1)
        AS avg_urine_6h,
    ROUND(AVG(fluid_balance_24h)::NUMERIC, 1)
        AS avg_fluid_balance,
    SUM(CASE WHEN fluid_balance_24h > 1000
        THEN 1 ELSE 0 END) AS n_positive_balance
FROM urine_features;

-- 소변량 보유율 63.3%
-- 42,210명 중 26,735명만 소변량 기록 있음
--나머지 36.7%는 소변량 기록 없음
--→ 이유: Foley catheter 없는 환자
 --        또는 기록 누락
--→ 이 환자들은 urine_6h = 0으로 처리됨
--→ Python 전처리에서 결측 지시변수 추가 필요

--avg_urine_6h 338.1 mL
--6시간 평균 소변량 338.1 mL
--mL/kg/h로 환산하면 (체중 70kg 기준):
--338.1 / 70 / 6 = 약 0.81 mL/kg/h
--→ 정상 기준 0.5 이상 → 평균적으로 정상

-- avg_fluid_balance -829.8 mL
-- 평균 수액 균형이 -829.8 mL (음수)
--→ 소변이 수액보다 829mL 더 많이 나옴
--→ 평균적으로 정상 범위
--→ 신장이 잘 기능하고 있다는 의미

--n_positive_balance 1,948명
--수액 균형 +1000mL 이상인 환자 1,948명 (4.6%)
--→ 수액을 많이 줬는데 소변이 안 나오는 환자
--→ AKI 위험이 높은 환자군
--→ 이 피처가 AKI 예측에 유용할 것으로 예상

--cohort                 42,210명    코호트 확정
--aki_stage_final        42,210명    레이블 + prediction_cutoff
--lab_features           33,267명    혈액검사 피처
--vital_features         33,636명    활력징후 피처
--urine_features         42,210명    소변량 + 수액 균형 피처


-- step4 numeric_features 통합
-- 지금까지 만든 세 개의 피처 테이블을 stay_id 기준으로 하나로 합치는 작업
-- 지금 앞에서 만든 혈액검사 집계값, 활력징후 집계값, 소변량 집계값이다 따로 있으니까 나중에 python에서 세번 불러와서 합쳐야 하니까 번거로와서 합침

DROP TABLE IF EXISTS numeric_features;

CREATE TABLE numeric_features AS
SELECT
    c.stay_id,
    c.subject_id,
    c.hadm_id,
    c.age,
    c.gender,
    c.icu_intime,
    c.icu_outtime,
    c.icu_los_hours,
    c.first_careunit,

    -- AKI 레이블
    a.aki_label,
    a.aki_stage,
    a.aki_onset_time,
    a.prediction_cutoff,
    a.hours_to_aki,

    -- 혈액검사 피처 (lab_features)
    l.cr_min,
    l.cr_max,
    l.cr_mean,
    l.cr_delta,
    l.bun_max,
    l.bun_mean,
    l.potassium_max,
    l.potassium_mean,
    l.bicarbonate_min,
    l.bicarbonate_mean,
    l.hemoglobin_min,
    l.hemoglobin_mean,
    l.lactate_max,
    l.lactate_mean,
    l.bun_cr_ratio,

    -- 활력징후 피처 (vital_features)
    v.hr_min,
    v.hr_max,
    v.hr_mean,
    v.rr_max,
    v.rr_mean,
    v.map_min,
    v.map_mean,
    v.sbp_min,
    v.sbp_mean,
    v.temp_min,
    v.temp_mean,
    v.spo2_min,
    v.spo2_mean,
    v.shock_index,

    -- 소변량 + 수액 균형 피처 (urine_features)
    u.urine_6h,
    u.urine_24h,
    u.urine_total,
    u.urine_zero_ratio,
    u.fluid_24h,
    u.fluid_total,
    u.fluid_balance_24h,
    u.fluid_balance_total,

    -- 결측 여부 지시변수
    -- "이 피처가 없었다는 것 자체가 정보"
    CASE WHEN l.cr_min IS NULL
         THEN 1 ELSE 0 END              AS cr_missing,
    CASE WHEN l.lactate_max IS NULL
         THEN 1 ELSE 0 END              AS lactate_missing,
    CASE WHEN u.urine_6h IS NULL
         OR u.urine_6h = 0
         THEN 1 ELSE 0 END              AS urine_missing,
    CASE WHEN v.map_min IS NULL
         THEN 1 ELSE 0 END              AS map_missing

FROM cohort             c
JOIN aki_stage_final    a ON c.stay_id = a.stay_id
LEFT JOIN lab_features  l ON c.stay_id = l.stay_id
LEFT JOIN vital_features v ON c.stay_id = v.stay_id
LEFT JOIN urine_features u ON c.stay_id = u.stay_id;


-- 왜 left join을 쓸까?
-- JOIN (inner join): 양쪽 다 있어야 포함
--  → lab_features 없는 환자 제외됨
--  → 코호트 42,210명보다 줄어듦

--LEFT JOIN: 왼쪽(cohort)은 항상 포함
--  → lab 기록 없어도 포함
--  → 없는 피처는 NULL로 채움
--  → 42,210명 전원 유지

-- 결측 지시변수가 왜 필요한가?
--젖산(lactate)이 NULL인 두 가지 경우

--경우 1: 측정을 안 했음
  --→ 의사가 위험하다고 판단 안 함
  --→ 낮은 위험

--경우 2: 검사 지시를 냈는데 기록 누락
 -- → 실제로 측정했을 수도 있음

--경우 1이 더 많음
--→ "젖산을 측정했는가 안 했는가" 자체가
--  위험도를 나타내는 신호
--→ lactate_missing = 1 (측정 안 함)
 --  → 상대적으로 안정적인 환자
--→ lactate_missing = 0 (측정함)
 --  → 의사가 위험하다고 판단한 환자

SELECT
    COUNT(*) AS n_total,
    SUM(aki_label) AS n_aki,
    ROUND(AVG(cr_delta)::NUMERIC, 2)
        AS avg_cr_delta,
    ROUND(AVG(map_min)::NUMERIC, 1)
        AS avg_map_min,
    ROUND(AVG(fluid_balance_24h)::NUMERIC, 1)
        AS avg_fluid_balance,
    SUM(lactate_missing) AS n_no_lactate,
    SUM(cr_missing) AS n_no_cr
FROM numeric_features;



















