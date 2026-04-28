-- 1) MAP 원본 테이블
DROP TABLE IF EXISTS raw_map;

CREATE TABLE raw_map AS
SELECT
    a.stay_id,
    a.prediction_cutoff,
    ce.charttime,
    ce.itemid,
    ce.valuenum AS map
FROM cohort_death_excluded c
JOIN aki_stage_final_death_excluded a
    ON c.stay_id = a.stay_id
JOIN mimiciv_icu.chartevents ce
    ON ce.stay_id = c.stay_id
   AND ce.charttime >= c.icu_intime
   AND ce.charttime <= COALESCE(a.prediction_cutoff, c.icu_outtime)
WHERE ce.itemid IN (220052, 220181, 225312)
  AND ce.valuenum IS NOT NULL
  AND ce.valuenum BETWEEN 20 AND 300;
-- 2) 승압제 원본 테이블
DROP TABLE IF EXISTS raw_vasopressor;

CREATE TABLE raw_vasopressor AS
SELECT
    a.stay_id,
    a.prediction_cutoff,
    ie.itemid,
    ie.starttime,
    ie.endtime,
    ie.rate,
    ie.rateuom,
    ie.patientweight
FROM cohort_death_excluded c
JOIN aki_stage_final_death_excluded a
    ON c.stay_id = a.stay_id
JOIN mimiciv_icu.inputevents ie
    ON ie.stay_id = c.stay_id
   AND ie.starttime >= c.icu_intime
   AND ie.starttime <= COALESCE(a.prediction_cutoff, c.icu_outtime)
WHERE ie.itemid IN (
      221653, 221662, 221289, 221906, 221749, 229630, 222315
)
  AND ie.rate IS NOT NULL
  AND NOT (
        ie.itemid = 221749 
        AND ie.rateuom = 'mg/min'
  )
  AND NOT (
        ie.itemid = 222315 
        AND ie.rateuom = 'units/min'
  );

SELECT COUNT(*) FROM raw_map;			-- 2,536,820
SELECT COUNT(*) FROM raw_vasopressor;	-- 133,504

----------------------------------------------------
-- Step 1. lab raw table
DROP TABLE IF EXISTS raw_labs;

CREATE TABLE raw_labs AS
SELECT
    c.stay_id,
    c.subject_id,
    c.hadm_id,
    a.aki_label,
    a.aki_stage,
    a.aki_onset_time,
    a.prediction_cutoff,
    l.charttime,
    l.itemid,
    l.valuenum
FROM cohort_death_excluded c
JOIN aki_stage_final_death_excluded a
    ON c.stay_id = a.stay_id
JOIN mimiciv_hosp.labevents l
    ON l.hadm_id = c.hadm_id
   AND l.charttime >= c.icu_intime
   AND l.charttime <= COALESCE(a.prediction_cutoff, c.icu_outtime)
WHERE l.itemid IN (
        50912,   -- Creatinine
        50882,   -- Bicarbonate
        50971,   -- Potassium
        51006,   -- Urea Nitrogen / BUN
        51222,   -- Hemoglobin
        50813    -- Lactate
  )
  AND l.valuenum IS NOT NULL;
-- Step 2. vital raw table
DROP TABLE IF EXISTS raw_vitals;

CREATE TABLE raw_vitals AS
SELECT
    c.stay_id,
    a.aki_label,
    a.aki_stage,
    a.prediction_cutoff,
    ce.charttime,
    ce.itemid,
    ce.valuenum
FROM cohort_death_excluded c
JOIN aki_stage_final_death_excluded a
    ON c.stay_id = a.stay_id
JOIN mimiciv_icu.chartevents ce
    ON ce.stay_id = c.stay_id
   AND ce.charttime >= c.icu_intime
   AND ce.charttime <= COALESCE(a.prediction_cutoff, c.icu_outtime)
WHERE ce.itemid IN (
        220045,   -- Heart Rate
        220210,   -- Respiratory Rate
        220179,   -- Non Invasive BP systolic
        223761,   -- Temperature Fahrenheit
        220277    -- SpO2
  )
  AND ce.valuenum IS NOT NULL
  AND NOT (ce.itemid = 220045 AND ce.valuenum NOT BETWEEN 20 AND 300)
  AND NOT (ce.itemid = 220210 AND ce.valuenum NOT BETWEEN 4 AND 60)
  AND NOT (ce.itemid = 220179 AND ce.valuenum NOT BETWEEN 40 AND 300)
  AND NOT (ce.itemid = 223761 AND ce.valuenum NOT BETWEEN 86 AND 115)
  AND NOT (ce.itemid = 220277 AND ce.valuenum NOT BETWEEN 50 AND 100);
-- Step 3. urine raw table
DROP TABLE IF EXISTS raw_urine;

CREATE TABLE raw_urine AS
SELECT
    c.stay_id,
    a.aki_label,
    a.aki_stage,
    a.prediction_cutoff,
    oe.charttime,
    oe.itemid,
    oe.value AS urine_value,
    oe.valueuom
FROM cohort_death_excluded c
JOIN aki_stage_final_death_excluded a
    ON c.stay_id = a.stay_id
JOIN mimiciv_icu.outputevents oe
    ON oe.stay_id = c.stay_id
   AND oe.charttime >= c.icu_intime
   AND oe.charttime <= COALESCE(a.prediction_cutoff, c.icu_outtime)
WHERE oe.itemid IN (
        226559, 226560, 226561,
        226563, 226627, 226631, 226584
  )
  AND oe.value IS NOT NULL
  AND oe.value >= 0
  AND oe.value < 2000;
-- Step 4. fluid raw table
DROP TABLE IF EXISTS raw_fluid;

CREATE TABLE raw_fluid AS
SELECT
    c.stay_id,
    a.aki_label,
    a.aki_stage,
    a.prediction_cutoff,
    ie.starttime,
    ie.endtime,
    ie.itemid,
    ie.amount,
    ie.amountuom
FROM cohort_death_excluded c
JOIN aki_stage_final_death_excluded a
    ON c.stay_id = a.stay_id
JOIN mimiciv_icu.inputevents ie
    ON ie.stay_id = c.stay_id
   AND ie.starttime >= c.icu_intime
   AND ie.starttime <= COALESCE(a.prediction_cutoff, c.icu_outtime)
WHERE ie.itemid IN (
        225158,   -- NaCl 0.9%
        220949,   -- Dextrose 5%
        225943,   -- Solution
        225828,   -- LR
        225797    -- Free Water
  )
  AND ie.amount IS NOT NULL
  AND ie.amount > 0
  AND ie.amountuom = 'mL';

SELECT COUNT(*) FROM raw_labs;		-- 795,175
SELECT COUNT(*) FROM raw_vitals;	-- 10,004,740
SELECT COUNT(*) FROM raw_urine;		-- 1,222,630
SELECT COUNT(*) FROM raw_fluid;		-- 913,742

---------------------------------------------------
 -- 7. 전체 raw feature 통합 테이블
DROP TABLE IF EXISTS raw_all_features;

CREATE TABLE raw_all_features AS

SELECT
    stay_id,
    prediction_cutoff,
    charttime AS event_time,
    'map' AS source,
    itemid,
    map AS value
FROM raw_map

UNION ALL

SELECT
    stay_id,
    prediction_cutoff,
    starttime AS event_time,
    'vasopressor' AS source,
    itemid,
    rate AS value
FROM raw_vasopressor

UNION ALL

SELECT
    stay_id,
    prediction_cutoff,
    charttime AS event_time,
    'lab' AS source,
    itemid,
    valuenum AS value
FROM raw_labs

UNION ALL

SELECT
    stay_id,
    prediction_cutoff,
    charttime AS event_time,
    'vital' AS source,
    itemid,
    valuenum AS value
FROM raw_vitals

UNION ALL

SELECT
    stay_id,
    prediction_cutoff,
    charttime AS event_time,
    'urine' AS source,
    itemid,
    urine_value AS value
FROM raw_urine

UNION ALL

SELECT
    stay_id,
    prediction_cutoff,
    starttime AS event_time,
    'fluid' AS source,
    itemid,
    amount AS value
FROM raw_fluid;


/* =========================
   8. 확인용
========================= */
SELECT source, COUNT(*) AS row_count
FROM raw_all_features
GROUP BY source
ORDER BY row_count DESC;

SELECT COUNT(*) AS total_raw_rows
FROM raw_all_features;				-- 15,606,611

----------------------------------------------------
/* =========================================================
   Final 48h Feature Table
   - prediction_cutoff 기준 직전 48시간만 사용
   - stay_id당 1줄
========================================================= */

DROP TABLE IF EXISTS final_features_48h;

CREATE TABLE final_features_48h AS
WITH base AS (
    SELECT
        c.stay_id,
        c.subject_id,
        c.hadm_id,
        c.age,
        c.gender,
        a.aki_label,
        a.aki_stage,
        a.aki_onset_time,
        a.prediction_cutoff,

        -- AKI=1: prediction_cutoff
        -- AKI=0: icu_outtime
        COALESCE(a.prediction_cutoff, c.icu_outtime) AS index_time
    FROM cohort_death_excluded c
    JOIN aki_stage_final_death_excluded a
        ON c.stay_id = a.stay_id
),

window_data AS (
    SELECT
        r.*
    FROM raw_all_features r
    JOIN base b
        ON r.stay_id = b.stay_id
    WHERE r.event_time >= b.index_time - INTERVAL '48 hours'
      AND r.event_time <= b.index_time
),

feature_agg AS (
    SELECT
        stay_id,

        -- Creatinine
        MAX(CASE WHEN source = 'lab' AND itemid = 50912 THEN value END) AS creatinine_max,
        MIN(CASE WHEN source = 'lab' AND itemid = 50912 THEN value END) AS creatinine_min,
        AVG(CASE WHEN source = 'lab' AND itemid = 50912 THEN value END) AS creatinine_mean,

        -- BUN
        MAX(CASE WHEN source = 'lab' AND itemid = 51006 THEN value END) AS bun_max,
        AVG(CASE WHEN source = 'lab' AND itemid = 51006 THEN value END) AS bun_mean,

        -- Bicarbonate
        MIN(CASE WHEN source = 'lab' AND itemid = 50882 THEN value END) AS bicarbonate_min,
        AVG(CASE WHEN source = 'lab' AND itemid = 50882 THEN value END) AS bicarbonate_mean,

        -- Potassium
        MAX(CASE WHEN source = 'lab' AND itemid = 50971 THEN value END) AS potassium_max,
        AVG(CASE WHEN source = 'lab' AND itemid = 50971 THEN value END) AS potassium_mean,

        -- Hemoglobin
        MIN(CASE WHEN source = 'lab' AND itemid = 51222 THEN value END) AS hemoglobin_min,
        AVG(CASE WHEN source = 'lab' AND itemid = 51222 THEN value END) AS hemoglobin_mean,

        -- Lactate
        MAX(CASE WHEN source = 'lab' AND itemid = 50813 THEN value END) AS lactate_max,
        AVG(CASE WHEN source = 'lab' AND itemid = 50813 THEN value END) AS lactate_mean,

        -- Heart Rate
        MAX(CASE WHEN source = 'vital' AND itemid = 220045 THEN value END) AS hr_max,
        AVG(CASE WHEN source = 'vital' AND itemid = 220045 THEN value END) AS hr_mean,

        -- Respiratory Rate
        MAX(CASE WHEN source = 'vital' AND itemid = 220210 THEN value END) AS rr_max,
        AVG(CASE WHEN source = 'vital' AND itemid = 220210 THEN value END) AS rr_mean,

        -- SpO2
        MIN(CASE WHEN source = 'vital' AND itemid = 220277 THEN value END) AS spo2_min,
        AVG(CASE WHEN source = 'vital' AND itemid = 220277 THEN value END) AS spo2_mean,

        -- Temperature
        MAX(CASE WHEN source = 'vital' AND itemid = 223761 THEN value END) AS temp_max,
        AVG(CASE WHEN source = 'vital' AND itemid = 223761 THEN value END) AS temp_mean,

        -- SBP
        MIN(CASE WHEN source = 'vital' AND itemid = 220179 THEN value END) AS sbp_min,
        AVG(CASE WHEN source = 'vital' AND itemid = 220179 THEN value END) AS sbp_mean,

        -- MAP
        MIN(CASE WHEN source = 'map' THEN value END) AS map_min,
        AVG(CASE WHEN source = 'map' THEN value END) AS map_mean,

        -- Urine / Fluid / Vasopressor
        SUM(CASE WHEN source = 'urine' THEN value ELSE 0 END) AS urine_48h,
        SUM(CASE WHEN source = 'fluid' THEN value ELSE 0 END) AS fluid_48h,

        MAX(CASE WHEN source = 'fluid' THEN 1 ELSE 0 END) AS fluid_flag,
        MAX(CASE WHEN source = 'vasopressor' THEN 1 ELSE 0 END) AS vasopressor_flag
     
    FROM window_data
    GROUP BY stay_id
)

SELECT
    b.stay_id,
    b.subject_id,
    b.hadm_id,
    b.age,
    b.gender,
    b.aki_label,
    b.aki_stage,
    b.aki_onset_time,
    b.prediction_cutoff,
    b.index_time,

    f.creatinine_max,
    f.creatinine_min,
    f.creatinine_mean,
    f.bun_max,
    f.bun_mean,
    f.bicarbonate_min,
    f.bicarbonate_mean,
    f.potassium_max,
    f.potassium_mean,
    f.hemoglobin_min,
    f.hemoglobin_mean,
    f.lactate_max,
    f.lactate_mean,

    f.hr_max,
    f.hr_mean,
    f.rr_max,
    f.rr_mean,
    f.spo2_min,
    f.spo2_mean,
    f.temp_max,
    f.temp_mean,
    f.sbp_min,
    f.sbp_mean,
    f.map_min,
    f.map_mean,

    COALESCE(f.urine_48h, 0) AS urine_48h,
    COALESCE(f.fluid_48h, 0) AS fluid_48h,
    COALESCE(f.fluid_flag, 0) AS fluid_flag,
    COALESCE(f.vasopressor_flag, 0) AS vasopressor_flag

FROM base b
LEFT JOIN feature_agg f
    ON b.stay_id = f.stay_id;


/* =========================
   9. 확인용 쿼리
========================= */

-- label 분포 확인: 0과 1이 둘 다 나와야 함
SELECT
    aki_label,
    COUNT(*) AS n
FROM final_features_48h
GROUP BY aki_label
ORDER BY aki_label;

-- stay_id 중복 확인: 두 값이 같아야 함
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT stay_id) AS distinct_stay_id
FROM final_features_48h;		-- 41,409

-- 주요 변수 NULL 확인
SELECT
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (WHERE creatinine_max IS NULL) AS creatinine_null,
    COUNT(*) FILTER (WHERE bun_max IS NULL) AS bun_null,
    COUNT(*) FILTER (WHERE hr_mean IS NULL) AS hr_null,
    COUNT(*) FILTER (WHERE sbp_mean IS NULL) AS sbp_null,
    COUNT(*) FILTER (WHERE map_mean IS NULL) AS map_null,
    COUNT(*) FILTER (WHERE urine_48h IS NULL) AS urine_null,
    COUNT(*) FILTER (WHERE fluid_48h IS NULL) AS fluid_null,
    COUNT(*) FILTER (WHERE vasopressor_flag IS NULL) AS vaso_flag_null
FROM final_features_48h;

-- 결과 샘플
SELECT *
FROM final_features_48h
LIMIT 10;
-- 주요 변수 null 개수 확인
SELECT
    COUNT(*) AS total_rows,

    COUNT(*) FILTER (WHERE creatinine_max IS NULL) AS creatinine_null,
    COUNT(*) FILTER (WHERE bun_max IS NULL) AS bun_null,
    COUNT(*) FILTER (WHERE hr_mean IS NULL) AS hr_null,
    COUNT(*) FILTER (WHERE sbp_mean IS NULL) AS sbp_null,
    COUNT(*) FILTER (WHERE map_mean IS NULL) AS map_null,
    COUNT(*) FILTER (WHERE urine_48h IS NULL) AS urine_null,
    COUNT(*) FILTER (WHERE fluid_48h IS NULL) AS fluid_null,
    COUNT(*) FILTER (WHERE vasopressor_flag IS NULL) AS vaso_flag_null
FROM final_features_48h;
-- 값 범위 확인
SELECT
    MIN(creatinine_min) AS cr_min,
    MAX(creatinine_max) AS cr_max,

    MIN(hr_mean) AS hr_mean_min,
    MAX(hr_mean) AS hr_mean_max,

    MIN(sbp_mean) AS sbp_mean_min,
    MAX(sbp_mean) AS sbp_mean_max,

    MIN(map_mean) AS map_mean_min,
    MAX(map_mean) AS map_mean_max,

    MIN(urine_48h) AS urine_min,
    MAX(urine_48h) AS urine_max
FROM final_features_48h;
-- 48시간 잘 적용됐는지
SELECT
    MIN(event_time - prediction_cutoff) AS min_diff,
    MAX(event_time - prediction_cutoff) AS max_diff
FROM raw_all_features
WHERE event_time >= prediction_cutoff - INTERVAL '48 hours'
  AND event_time <= prediction_cutoff;