-- 1) MAP 원본 테이블
DROP TABLE IF EXISTS raw_map;

CREATE TABLE raw_map AS
SELECT
    a.stay_id,
    a.prediction_cutoff,
    ce.charttime,
    ce.itemid,
    ce.valuenum AS map
FROM cohort c
JOIN aki_stage_final a
    ON c.stay_id = a.stay_id
JOIN mimiciv_icu.chartevents ce
    ON ce.stay_id = c.stay_id
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
    EXTRACT( EPOCH FROM ( ie.endtime - ie.starttime ) ) / 3600.0 AS duration_hours,
    ie.patientweight
FROM cohort c
JOIN aki_stage_final a
    ON c.stay_id = a.stay_id
JOIN mimiciv_icu.inputevents ie
    ON ie.stay_id = c.stay_id
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
FROM cohort c
JOIN aki_stage_final a
    ON c.stay_id = a.stay_id
JOIN mimiciv_hosp.labevents l
    ON l.hadm_id = c.hadm_id
   AND l.charttime <= COALESCE(a.prediction_cutoff, c.icu_outtime)
WHERE l.itemid IN (
        50912,   -- Creatinine
        50882,   -- Bicarbonate
        50971,   -- Potassium
        51006,   -- Urea Nitrogen / BUN
        51222,   -- Hemoglobin
        50813,    -- Lactate
        50983    -- Sodium
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
FROM cohort c
JOIN aki_stage_final a
    ON c.stay_id = a.stay_id
JOIN mimiciv_icu.chartevents ce
    ON ce.stay_id = c.stay_id
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
FROM cohort c
JOIN aki_stage_final a
    ON c.stay_id = a.stay_id
JOIN mimiciv_icu.outputevents oe
    ON oe.stay_id = c.stay_id
   AND oe.charttime <= COALESCE(a.prediction_cutoff, c.icu_outtime)
WHERE oe.itemid IN (
        226559, 226560, 226561,
        226563, 226627, 226631, 226584
  )
  AND oe.value IS NOT NULL
  AND oe.value >= 0
  AND oe.value < 2000;

---------------------------------------------------
 -- 7. 전체 raw feature 통합 테이블
DROP TABLE IF EXISTS raw_all_features;

CREATE TABLE raw_all_features AS

/* =========================================
   MAP
========================================= */
SELECT
    stay_id,
    prediction_cutoff,
    charttime AS event_time,
    'map' AS source,
    itemid,
    map AS value,
    NULL::FLOAT AS duration_hours
FROM raw_map

UNION ALL


/* =========================================
   Vasopressor
========================================= */
SELECT
    stay_id,
    prediction_cutoff,
    starttime AS event_time,
    'vasopressor' AS source,
    itemid,
    rate AS value,
    duration_hours
FROM raw_vasopressor

UNION ALL


/* =========================================
   Laboratory
========================================= */
SELECT
    stay_id,
    prediction_cutoff,
    charttime AS event_time,
    'lab' AS source,
    itemid,
    valuenum AS value,
    NULL::FLOAT AS duration_hours
FROM raw_labs

UNION ALL


/* =========================================
   Vital signs
========================================= */
SELECT
    stay_id,
    prediction_cutoff,
    charttime AS event_time,
    'vital' AS source,
    itemid,
    valuenum AS value,
    NULL::FLOAT AS duration_hours
FROM raw_vitals

UNION ALL


/* =========================================
   Urine output
========================================= */
SELECT
    stay_id,
    prediction_cutoff,
    charttime AS event_time,
    'urine' AS source,
    itemid,
    urine_value AS value,
    NULL::FLOAT AS duration_hours
FROM raw_urine;
/* =========================
   8. 확인용
========================= */
SELECT source, COUNT(*) AS row_count
FROM raw_all_features
GROUP BY source
ORDER BY row_count DESC;

SELECT COUNT(*) AS total_raw_rows
FROM raw_all_features;				-- 15,421,029

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

        COALESCE(
            a.prediction_cutoff,
            c.icu_outtime
        ) AS index_time

    FROM cohort c

    JOIN aki_stage_final a
        ON c.stay_id = a.stay_id
),

window_data AS (

    SELECT
        r.*

    FROM raw_all_features r

    JOIN base b
        ON r.stay_id = b.stay_id

    WHERE r.event_time >= (
            b.index_time - INTERVAL '48 hours'
          )
      AND r.event_time <= b.index_time
),
map_duration AS (

SELECT
    stay_id,

    event_time,

    value AS map_value,

    LEAD(event_time) OVER (
        PARTITION BY stay_id
        ORDER BY event_time
    ) AS next_time

FROM window_data
WHERE source = 'map'

),

map_features AS (

SELECT
    stay_id,

    AVG(map_value) AS map_mean,

    MIN(map_value) AS map_min,

    SUM(
        CASE
            WHEN map_value < 65
             AND next_time IS NOT NULL
            THEN LEAST(
    EXTRACT(
        EPOCH FROM (
            next_time - event_time
        )
    ) / 3600.0,
    1.0
)
            ELSE 0
        END
    ) AS map_below65_hours

FROM map_duration
GROUP BY stay_id

),

feature_agg AS (

SELECT
    stay_id,

    /* Creatinine */
    MAX(
CASE
WHEN source='lab'
AND itemid=50912
THEN value
END
) AS creatinine_max,

MIN(
CASE
WHEN source='lab'
AND itemid=50912
THEN value
END
) AS creatinine_min,

(
MAX(
CASE
WHEN source='lab'
AND itemid=50912
THEN value
END
)
-
MIN(
CASE
WHEN source='lab'
AND itemid=50912
THEN value
END
)
) AS creatinine_delta,

    /* BUN */
    MAX(
        CASE
            WHEN source='lab'
             AND itemid=51006
            THEN value
        END
    ) AS bun_max,

    (
        MAX(
            CASE
                WHEN source='lab'
                 AND itemid=51006
                THEN value
            END
        )
        /
        NULLIF(
            MAX(
                CASE
                    WHEN source='lab'
                     AND itemid=50912
                    THEN value
                END
            ),
            0
        )
    ) AS bun_cr_ratio,

    /* Electrolyte */
    MAX(
        CASE
            WHEN source='lab'
             AND itemid=50971
            THEN value
        END
    ) AS potassium_max,
    AVG(
CASE
WHEN source='lab'
AND itemid=50971
THEN value
END
) AS potassium_mean,

    MIN(
        CASE
            WHEN source='lab'
             AND itemid=50882
            THEN value
        END
    ) AS bicarbonate_min,
    AVG(
CASE
WHEN source='lab'
AND itemid=50882
THEN value
END
) AS bicarbonate_mean,

    MIN(
        CASE
            WHEN source='lab'
             AND itemid=50983
            THEN value
        END
    ) AS sodium_min,

    MAX(
        CASE
            WHEN source='lab'
             AND itemid=50983
            THEN value
        END
    ) AS sodium_max,

    /* Hemoglobin */
    MIN(
        CASE
            WHEN source='lab'
             AND itemid=51222
            THEN value
        END
    ) AS hemoglobin_min,
    AVG(
CASE
WHEN source='lab'
AND itemid=51222
THEN value
END
) AS hemoglobin_mean,

/* Heart Rate */
MAX(
CASE
WHEN source='vital'
AND itemid=220045
THEN value
END
) AS hr_max,

AVG(
CASE
WHEN source='vital'
AND itemid=220045
THEN value
END
) AS hr_mean,

    /* Lactate */
    MAX(
        CASE
            WHEN source='lab'
             AND itemid=50813
            THEN value
        END
    ) AS lactate_max,

    AVG(
        CASE
            WHEN source='lab'
             AND itemid=50813
            THEN value
        END
    ) AS lactate_mean,

    /* SBP */
    MIN(
        CASE
            WHEN source='vital'
             AND itemid=220179
            THEN value
        END
    ) AS sbp_min,
    AVG(
CASE
WHEN source='vital'
AND itemid=220179
THEN value
END
) AS sbp_mean,

    /* Shock Index */
    (
        AVG(
            CASE
                WHEN source='vital'
                 AND itemid=220045
                THEN value
            END
        )
        /
        NULLIF(
            AVG(
                CASE
                    WHEN source='vital'
                     AND itemid=220179
                    THEN value
                END
            ),
            0
        )
    ) AS shock_index_mean,

    /* Respiratory Rate */
    MAX(
CASE
WHEN source='vital'
AND itemid=220210
THEN value
END
) AS rr_max,

AVG(
CASE
WHEN source='vital'
AND itemid=220210
THEN value
END
) AS rr_mean,

    /* SpO2 */
    MIN(
        CASE
            WHEN source='vital'
             AND itemid=220277
            THEN value
        END
    ) AS spo2_min,
    AVG(
CASE
WHEN source='vital'
AND itemid=220277
THEN value
END
) AS spo2_mean,

    /* Urine */
    SUM(
        CASE
            WHEN source='urine'
            THEN value
            ELSE 0
        END
    ) AS urine_output_sum,

    SUM(
        CASE
            WHEN source='urine'
             AND event_time >= (
                    prediction_cutoff
                    - INTERVAL '6 hours'
                 )
            THEN value
            ELSE 0
        END
    ) AS urine_output_6h,

    (
        (
            SUM(
                CASE
                    WHEN source='urine'
                    THEN value
                    ELSE 0
                END
            )
        ) / 48.0
    ) / 70.0
    AS urine_ml_kg_hr,

    CASE
        WHEN (
            (
                SUM(
                    CASE
                        WHEN source='urine'
                        THEN value
                        ELSE 0
                    END
                )
            ) / 48.0
        ) / 70.0 < 0.5
        THEN 1
        ELSE 0
    END AS oliguria_flag,

    /* Temperature */
MAX(
CASE
WHEN source='vital'
AND itemid=223761
THEN value
END
) AS temp_max,

AVG(
CASE
WHEN source='vital'
AND itemid=223761
THEN value
END
) AS temp_mean,
    /* Vasopressor */
    MAX(
        CASE
            WHEN source='vasopressor'
            THEN 1
            ELSE 0
        END
    ) AS vasopressor_flag,

    SUM(
        CASE
            WHEN source='vasopressor'
            THEN duration_hours
            ELSE 0
        END
    ) AS vasopressor_hours,

    MAX(
        CASE
            WHEN source='vasopressor'
             AND itemid=221906
            THEN value
        END
    ) AS norepi_dose_max
    

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

/* Hemodynamic */
mf.map_mean,
mf.map_min,
mf.map_below65_hours,

f.sbp_min,
f.sbp_mean,
f.shock_index_mean,

f.hr_max,
f.hr_mean,

f.rr_max,
f.rr_mean,

f.temp_max,
f.temp_mean,

/* Renal */
f.urine_output_sum,
f.urine_output_6h,
f.urine_ml_kg_hr,
f.oliguria_flag,

f.creatinine_min,
f.creatinine_max,
f.creatinine_delta,

f.bun_max,
f.bun_cr_ratio,

/* Shock */
f.lactate_max,
f.lactate_mean,

f.vasopressor_flag,
f.vasopressor_hours,
f.norepi_dose_max,

/* Electrolyte */
f.potassium_max,
f.potassium_mean,
f.bicarbonate_min,
f.bicarbonate_mean,
f.sodium_min,
f.sodium_max,

/* Oxygen */
f.hemoglobin_min,
f.hemoglobin_mean,
f.spo2_min,
f.spo2_mean


FROM base b

LEFT JOIN feature_agg f
ON b.stay_id = f.stay_id

LEFT JOIN map_features mf
ON b.stay_id = mf.stay_id;



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

SELECT
    aki_stage,
    COUNT(*) AS n
FROM final_features_48h
GROUP BY aki_stage
ORDER BY aki_stage;

-- stay_id 중복 확인: 두 값이 같아야 함
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT stay_id) AS distinct_stay_id
FROM final_features_48h;		-- 42,210


-- 주요 변수 NULL 확인
SELECT
    COUNT(*) AS total_rows,

    COUNT(*) FILTER (
        WHERE creatinine_max IS NULL
    ) AS creatinine_null,

    COUNT(*) FILTER (
        WHERE bun_max IS NULL
    ) AS bun_null,

    COUNT(*) FILTER (
        WHERE map_mean IS NULL
    ) AS map_null,

    COUNT(*) FILTER (
        WHERE sbp_min IS NULL
    ) AS sbp_null,

    COUNT(*) FILTER (
        WHERE urine_output_sum IS NULL
    ) AS urine_null,

    COUNT(*) FILTER (
        WHERE lactate_max IS NULL
    ) AS lactate_null,

    COUNT(*) FILTER (
    WHERE vasopressor_flag = 1
	) AS vaso_used

FROM final_features_48h;

-- 결과 샘플
SELECT *
FROM final_features_48h
LIMIT 10;

-- 값 범위 확인
SELECT


-- Creatinine
MIN(creatinine_max) AS cr_min,
MAX(creatinine_max) AS cr_max,

-- BUN
MIN(bun_max) AS bun_min,
MAX(bun_max) AS bun_max,

-- MAP
MIN(map_mean) AS map_mean_min,
MAX(map_mean) AS map_mean_max,

-- SBP
MIN(sbp_min) AS sbp_min_value,
MAX(sbp_min) AS sbp_max_value,

-- Shock Index
MIN(shock_index_mean) AS shock_index_min,
MAX(shock_index_mean) AS shock_index_max,

-- Urine
MIN(urine_output_sum) AS urine_min,
MAX(urine_output_sum) AS urine_max,

-- Lactate
MIN(lactate_max) AS lactate_min,
MAX(lactate_max) AS lactate_max,

-- Sodium
MIN(sodium_min) AS sodium_lowest,
MAX(sodium_max) AS sodium_highest


FROM final_features_48h;

-- 48시간 잘 적용됐는지
SELECT
    MIN(event_time - prediction_cutoff) AS min_diff,
    MAX(event_time - prediction_cutoff) AS max_diff
FROM raw_all_features
WHERE event_time >= prediction_cutoff - INTERVAL '48 hours'
  AND event_time <= prediction_cutoff;

/* =========================================
   final_features_48h 컬럼 확인
========================================= */

SELECT
    column_name,
    data_type
FROM information_schema.columns
WHERE table_name = 'final_features_48h'
ORDER BY ordinal_position;

select * from final_features_48h ffh;